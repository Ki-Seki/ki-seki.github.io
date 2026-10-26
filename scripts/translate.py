#!/usr/bin/env python3
"""Translate posts and moments with DeepSeek for the on-page translation button.

Each page's index.md, front matter included, is translated as a whole: English
pages into Simplified Chinese, Chinese pages into English. A translation is
reused for as long as the hash of its source file stays the same. The cache is
the translations.json published with the previous deployment (plus a local
data/translations.json, if any), so only new and changed pages reach the API.

Writes data/translations.json, which Hugo renders into each page's
translation.json, and static/translations.json, the cache for the next build.
Without DEEPSEEK_API_KEY it only reuses cached translations.

Environment:
  DEEPSEEK_API_KEY      API key; translation is skipped when unset
  DEEPSEEK_MODEL        model name (default: deepseek-flash)
  TRANSLATE_MAX_CHARS   most source characters sent in one run (default: 1000000)
  TRANSLATE_WORKERS     concurrent requests (default: 8)
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ("posts", "moments")
CACHE_URL = "https://ki-seki.github.io/translations.json"
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
MAX_CHARS = int(os.environ.get("TRANSLATE_MAX_CHARS", "1000000"))
WORKERS = int(os.environ.get("TRANSLATE_WORKERS", "8"))
# Requests per page. A reply that fails the structure check is usually a
# one-off slip (a stray quote) that a fresh request doesn't repeat.
ATTEMPTS = 3
# USD per million tokens at peak hours (cache-miss input, output), for the
# cost estimate only. See https://api-docs.deepseek.com/quick_start/pricing
PRICES = {"deepseek-flash": (0.3, 1.2), "deepseek-v4-pro": (1.32, 3.96)}

# Bump to retranslate every page after changing the prompt.
PROMPT_VERSION = 2
LANGUAGES = {"zh": "Simplified Chinese", "en": "English"}
PROMPT = """Translate the following Markdown file from a Hugo blog into {language}.

Keep the structure exactly as it is: front matter keys, Markdown syntax, Hugo shortcodes, HTML, math, code, URLs and file paths must not be changed or broken. Translate the text readers see, including front matter values such as the title, summary and captions, link text, and the captions and titles inside shortcodes. Inside shortcode parameters, write quotation marks as “ ” rather than ", which would end the parameter; in the front matter, keep the quotes around each value exactly as they are. Leave dates, tags and other values that aren't prose as they are. Keep people's names written in Latin letters exactly as they are: never turn a pinyin name into Chinese characters, which can't be told from the spelling.

Reply with the translated file only."""

FRONT_MATTER = re.compile(r"---\n(.*?)\n---\n(.*)", re.S)
# A shortcode closed in its own style: {{< ... >}} or {{% ... %}}
SHORTCODE = re.compile(r"\{\{<(.*?)>\}\}|\{\{%(.*?)%\}\}", re.S)
# A shortcode's name followed by parameters that are all named or all
# positional (Hugo rejects a mix), each value quoted, backquoted or bare
SHORTCODE_VALUE = r'(?:"(?:[^"\\]|\\.)*"|`[^`]*`|[^\s"`=]+)'
SHORTCODE_ARGS = re.compile(
    rf"\s*/?[\w.-]+(?:(?:\s+[\w-]+={SHORTCODE_VALUE})*|(?:\s+{SHORTCODE_VALUE})*)\s*", re.S
)
# A horizontal rule (three or more -, * or _) or a setext heading underline (=)
RULE_LINE = re.compile(r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|=+[ \t]*)$", re.M)


class OutOfBalance(Exception):
    pass


class Budget:
    """Source characters left to send in this run, shared by all requests,
    retries included."""

    def __init__(self, chars):
        self.left = chars
        self.lock = threading.Lock()

    def take(self, chars):
        with self.lock:
            if chars > self.left:
                return False
            self.left -= chars
            return True


def target_language(text):
    """Chinese pages go to English, everything else to Chinese; None if no text."""
    prose = re.sub(r"```.*?```|\{\{[<%].*?[%>]\}\}|\]\([^)]*\)|https?://\S+", " ", text, flags=re.S)
    han = len(re.findall(r"[一-鿿]", prose))
    words = len(re.findall(r"[A-Za-z]+", prose))
    if not han and not words:
        return None
    return "en" if han > words else "zh"


def quotes_balanced(front):
    """Whether every quoted front matter value is still valid YAML, e.g. a
    translated title hasn't gained an unescaped quote."""
    for value in re.findall(r"^[ \t]*(?:- |[\w-]+:[ \t]*)(['\"].*)$", front, re.M):
        value = value.rstrip()
        if not re.fullmatch(r"'(?:[^']|'')*'|\"(?:[^\"\\]|\\.)*\"", value):
            return False
    return True


def shortcodes_valid(body):
    """Whether every shortcode opener is closed in its own style, with
    parameters Hugo can parse."""
    found = [named or positional for named, positional in SHORTCODE.findall(body)]
    if len(found) != body.count("{{<") + body.count("{{%"):
        return False
    return all("{{" not in args and SHORTCODE_ARGS.fullmatch(args) for args in found)


def structure(text):
    """What a faithful translation keeps unchanged: front matter keys, valid
    quoting in front matter and shortcode parameters, and counts of
    shortcodes, code fences, display math, headings and rule lines."""
    match = FRONT_MATTER.match(text)
    if not match:
        return None
    front, body = match.groups()
    return (
        tuple(re.findall(r"^[ \t]*([\w-]+):", front, re.M)),
        quotes_balanced(front),
        # The quote each front matter value opens with: a translated title
        # wrapped in “ ” instead of " " would show the curly quotes
        tuple(re.findall(r"^[ \t]*(?:- |[\w-]+:[ \t]*)(['\"“”‘’]?)", front, re.M)),
        shortcodes_valid(body),
        body.count("{{<"),
        body.count("{{%"),
        body.count("```"),
        body.count("$$"),
        len(re.findall(r"^#{1,6} ", body, re.M)),
        # A stray --- after the last line turns that line into a heading
        len(RULE_LINE.findall(body)),
    )


def load_cache(source):
    try:
        if re.match(r"https?://", source):
            with urllib.request.urlopen(source, timeout=60) as response:
                return json.load(response)
        return json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"No cached translations from {source}: {error}")
        return {}


def request_translation(text, lang, key):
    """Stream one translation; return (translated text, finish_reason, usage)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT.replace("{language}", LANGUAGES[lang])},
            {"role": "user", "content": text},
        ],
        "thinking": {"type": "disabled"},
        "max_tokens": 384000,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    parts, finish, usage = [], None, {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    reply = "".join(parts).strip()
    # Unwrap a reply that came back as one fenced code block
    fenced = re.fullmatch(r"```[\w-]*\n(.*)\n```", reply, re.S)
    return (fenced.group(1) if fenced else reply) + "\n", finish, usage


def translate(page, key, stop, budget):
    """Translate a page in up to ATTEMPTS requests; return (entry or None,
    usage, note)."""
    total = {"prompt_tokens": 0, "completion_tokens": 0}
    note = ""
    for attempt in range(ATTEMPTS):
        if stop.is_set():
            return None, total, "stopped"
        if not budget.take(len(page["text"])):
            return None, total, "; ".join(filter(None, [note, "over TRANSLATE_MAX_CHARS"]))
        try:
            text, finish, usage = request_translation(page["text"], page["lang"], key)
        except urllib.error.HTTPError as error:
            if error.code == 402:
                stop.set()
                raise OutOfBalance() from error
            note = f"HTTP {error.code}"
            time.sleep(10 * (attempt + 1))
            continue
        except (OSError, ValueError) as error:
            note = str(error)
            time.sleep(10 * (attempt + 1))
            continue
        for name in total:
            total[name] += usage.get(name, 0)
        if finish != "stop":
            note = f"stopped early ({finish})"
        elif structure(text) != structure(page["text"]):
            note = "structure changed"
        else:
            return {"hash": page["hash"], "lang": page["lang"], "markdown": text}, total, ""
    return None, total, note


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", default=CACHE_URL, help="URL or file of published translations")
    parser.add_argument("--only", help="translate only pages whose path contains this text")
    parser.add_argument("--dry-run", action="store_true", help="list the pages that need translating")
    args = parser.parse_args()

    data_file = ROOT / "data" / "translations.json"
    cached = {}
    for source in (args.cache, data_file):
        if source == data_file and not data_file.exists():
            continue
        for entry in load_cache(str(source)).values():
            cached[entry["hash"]] = entry

    translations, pending = {}, []
    for section in SECTIONS:
        for path in sorted((ROOT / "content" / section).glob("*/index.md")):
            text = path.read_text(encoding="utf-8")
            match = FRONT_MATTER.match(text)
            if not match or re.search(r"^draft:[ \t]*true[ \t]*$", match.group(1), re.M):
                continue
            title = re.search(r"^title:[ \t]*(.*)$", match.group(1), re.M)
            lang = target_language(f"{title.group(1) if title else ''}\n{match.group(2)}")
            if not lang:
                continue
            name = f"{section}/{path.parent.name}"
            digest = hashlib.sha256(json.dumps([PROMPT_VERSION, MODEL, lang, text]).encode()).hexdigest()
            # Cached translations are checked again, so ones that fail a
            # newer check are redone
            entry = cached.get(digest)
            if entry and structure(entry.get("markdown", "")) == structure(text):
                translations[name] = entry
            else:
                pending.append({"name": name, "text": text, "lang": lang, "hash": digest})

    key = os.environ.get("DEEPSEEK_API_KEY", "")
    queue = [page for page in pending if not args.only or args.only in page["name"]]
    print(f"{len(translations)} cached, {len(pending)} new or changed, {len(queue)} to translate "
          f"({sum(len(page['text']) for page in queue)} characters; at most {MAX_CHARS} sent)")
    for page in queue:
        print(f"  {page['name']} -> {page['lang']}")
    if args.dry_run:
        return
    deferred = []
    if queue and not key:
        print("DEEPSEEK_API_KEY is not set; new and changed pages stay untranslated.")
        deferred = [(page["name"], "no API key") for page in queue]
        queue = []

    stop = threading.Event()
    budget = Budget(MAX_CHARS)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {pool.submit(translate, page, key, stop, budget): page for page in queue}
        for job in concurrent.futures.as_completed(jobs):
            page = jobs[job]
            try:
                entry, used, note = job.result()
            except OutOfBalance:
                entry, used, note = None, {}, "DeepSeek balance used up"
            for name in usage:
                usage[name] += used.get(name, 0)
            if entry:
                translations[page["name"]] = entry
                print(f"  translated {page['name']} ({used['completion_tokens']} tokens)")
            else:
                failed.append((page["name"], note))
                print(f"  not translated {page['name']}: {note}")

    output = json.dumps(dict(sorted(translations.items())), ensure_ascii=False, indent=1)
    for path in (data_file, ROOT / "static" / "translations.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")

    input_price, output_price = PRICES.get(MODEL, (0, 0))
    cost = (usage["prompt_tokens"] * input_price + usage["completion_tokens"] * output_price) / 1e6
    summary = [
        "## Translations",
        "",
        f"- Model: `{MODEL}`",
        f"- Pages with a translation: {len(translations)}",
        f"- Translated in this run: {len(queue) - len(failed)}",
        f"- Characters sent: {MAX_CHARS - budget.left} (limit {MAX_CHARS})",
        f"- Tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out"
        + (f" (about ${cost:.3f} at peak prices)" if cost else ""),
    ]
    for name, note in failed + deferred:
        summary.append(f"- Not translated: `{name}` ({note})")
    print("\n".join(summary))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    sys.exit(main())
