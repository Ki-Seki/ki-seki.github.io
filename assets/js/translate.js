// Adds a floating button that swaps a post or moment between its original
// text and the DeepSeek translation that the build publishes next to it as
// translation.json (see scripts/translate.py). The script tag carries the
// translation's URL (data-src) and language (data-lang).
//
// Readers whose browser prefers the translation's language get it on load.
// A reader's own switch is remembered for later pages.
(function () {
  const script = document.currentScript;
  const storageKey = "translate-lang";

  // Button label and tooltip by the language a click switches to
  const labels = { zh: "中", en: "EN" };
  const titles = {
    translation: { zh: "中文翻译", en: "English translation" },
    original: { zh: "查看原文", en: "Show the original" },
  };
  // Small note in the bottom-left corner while the translation is on screen
  const notes = { zh: "DeepSeek 翻译", en: "Translated by DeepSeek" };
  const failures = { zh: "翻译加载失败", en: "Translation unavailable" };

  // The language this reader last switched to, else the first of their
  // browser languages that is Chinese or English
  const preferredLanguage = () => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) {
        return saved;
      }
    } catch (error) {
      // Storage can be blocked; fall back to the browser languages
    }
    const languages = navigator.languages?.length ? navigator.languages : [navigator.language || ""];
    return languages.map((language) => language.toLowerCase().slice(0, 2)).find((language) => language === "zh" || language === "en");
  };

  const rememberLanguage = (language) => {
    try {
      localStorage.setItem(storageKey, language);
    } catch (error) {
      // Not remembered when storage is blocked
    }
  };

  const init = () => {
    const original = document.querySelector(".post-single .post-content");
    if (!original) {
      return;
    }

    const lang = script.dataset.lang;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "translate-toggle";
    button.textContent = labels[lang];
    button.title = titles.translation[lang];
    const note = document.createElement("span");
    note.className = "translate-note";
    note.textContent = notes[lang];
    note.hidden = true;
    document.body.append(note, button);

    const sourceLang = lang === "zh" ? "en" : "zh";
    const titleNode = [...(document.querySelector(".post-single .post-title")?.childNodes || [])]
      .find((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
    const originalTitle = titleNode?.textContent;
    const caption = document.querySelector(".post-single .entry-cover figcaption");
    const originalCaption = caption?.innerHTML;
    const chips = document.querySelector(".post-single .moment-meta--single");
    const originalChips = chips?.innerHTML;
    const toc = document.querySelector(".post-single .toc");
    const tocLinks = [...(toc?.querySelectorAll("a[href^='#']") || [])].map((link) => ({
      link,
      id: decodeURIComponent(link.hash.slice(1)),
      text: link.textContent,
    }));
    let translation = null;
    let showing = false;

    const headingText = (heading) => {
      const copy = heading.cloneNode(true);
      copy.querySelectorAll(".anchor").forEach((anchor) => anchor.remove());
      return copy.textContent.trim();
    };

    // Gives the translation's headings the IDs of the original's, so the table
    // of contents and in-page links keep their targets. Returns each heading's
    // translated text by ID, or null when the headings don't line up.
    const alignHeadings = (content) => {
      const selector = "h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]";
      const from = [...original.querySelectorAll(selector)];
      const to = [...content.querySelectorAll(selector)];
      if (from.length !== to.length || from.some((heading, index) => heading.tagName !== to[index].tagName)) {
        return null;
      }
      const texts = new Map();
      to.forEach((heading, index) => {
        const id = from[index].id;
        heading.id = id;
        heading.querySelector(".anchor")?.setAttribute("href", `#${id}`);
        texts.set(id, headingText(heading));
      });
      return texts;
    };

    const load = async () => {
      const response = await fetch(script.dataset.src);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      const content = document.createElement("div");
      content.className = "post-content";
      content.lang = lang === "zh" ? "zh-CN" : "en";
      content.innerHTML = data.content;
      // Embeds such as GeoGebra take focus once loaded, which would scroll the
      // page away from the button; load them only when scrolled into view.
      content.querySelectorAll("iframe").forEach((frame) => {
        frame.loading = "lazy";
      });
      return { content, title: data.title, caption: data.caption, chips: data.chips, headings: alignHeadings(content) };
    };

    // Where the reader is: the block at the top of the viewport, by its
    // position among the content's blocks, and how far down the screen it is.
    // Display math only becomes an element (mjx-container) once typeset, so it
    // isn't counted.
    const blocks = (content) => [...content.children].filter((block) => !block.matches("mjx-container"));
    const readingSpot = (content) => {
      const list = blocks(content);
      const index = list.findIndex((block) => block.getBoundingClientRect().bottom > 0);
      return index < 0 ? null : { index, top: list[index].getBoundingClientRect().top };
    };
    const restoreSpot = (content, spot) => {
      const block = spot && blocks(content)[spot.index];
      if (block) {
        window.scrollBy({ top: block.getBoundingClientRect().top - spot.top, behavior: "instant" });
      }
    };

    // Typesets math in content the first time it is shown, and wires up
    // admonitions in the translation as admonition.js did for the original.
    // The translation can be shown on load, before MathJax has typeset the
    // page, so the original may need typesetting when switched back to.
    // Resolves once math is rendered.
    const prepared = new WeakSet();
    const prepare = (content) => {
      if (prepared.has(content)) {
        return Promise.resolve();
      }
      prepared.add(content);
      if (content !== original) {
        content.querySelectorAll(".details .details-summary").forEach((summary) => {
          summary.addEventListener("click", () => summary.parentElement.classList.toggle("open"));
        });
      }
      if (window.MathJax?.startup?.promise) {
        return window.MathJax.startup.promise.then(() => window.MathJax.typesetPromise([content]));
      }
      if (window.renderMathInElement) {
        window.renderMathInElement(content, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
          ],
        });
      }
      return Promise.resolve();
    };

    const show = (translated) => {
      const [from, to] = translated ? [original, translation.content] : [translation.content, original];
      const spot = readingSpot(from);
      from.replaceWith(to);
      if (titleNode) {
        titleNode.textContent = translated ? translation.title : originalTitle;
      }
      if (caption && translation.caption) {
        caption.innerHTML = translated ? translation.caption : originalCaption;
      }
      if (chips && translation.chips) {
        chips.innerHTML = translated ? translation.chips : originalChips;
      }
      if (toc) {
        toc.hidden = translated && !translation.headings;
        for (const { link, id, text } of tocLinks) {
          link.textContent = (translated && translation.headings?.get(id)) || text;
        }
      }
      window.__siteMediaLightbox?.reload();
      restoreSpot(to, spot);
      prepare(to).then(() => restoreSpot(to, spot));
      note.hidden = !translated;
      button.textContent = labels[translated ? sourceLang : lang];
      button.title = translated ? titles.original[sourceLang] : titles.translation[lang];
      showing = translated;
    };

    const toggle = async () => {
      if (!translation) {
        button.disabled = true;
        button.textContent = "…";
        try {
          translation = await load();
        } catch (error) {
          // Left enabled, so a click retries
          console.error("Translation failed to load:", error);
          button.textContent = "!";
          button.title = failures[lang];
          return false;
        } finally {
          button.disabled = false;
        }
      }
      show(!showing);
      return true;
    };

    button.addEventListener("click", async () => {
      if (await toggle()) {
        rememberLanguage(showing ? lang : sourceLang);
      }
    });

    if (preferredLanguage() === lang) {
      toggle();
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
