// The site's reading language, for the DeepSeek translations the build
// publishes (see scripts/translate.py).
//
// A floating button opens a menu of three modes: 原文 (every page as
// written), English or 中文 (pages written in the other language show their
// translation). The mode is remembered and shared by every page; until a
// reader picks one, the browser language decides. The address shows the mode
// (no parameter for the original, ?lang=en or ?lang=zh), so a link carries
// it; a mode given that way applies to that page only, without changing the
// reader's own. On a post or moment the
// script tag carries the page's translation URL (data-src) and language
// (data-lang); on list pages (home, posts, tags, archive, moments) it carries
// the URL of the list translations (data-list).
//
// Translations are labelled where they appear: a line under a page's meta
// line and a tag on each translated list entry.
(function () {
  const script = document.currentScript;
  const storageKey = "translate-lang";
  const urlKey = "lang";

  const modes = ["original", "en", "zh"];
  // English and 中文 name themselves; "original" is said in the reader's
  // language (see originalLabel)
  const menuLabels = { original: { zh: "原文", en: "Original" }, en: "English", zh: "中文" };
  const buttonTitle = "Language · 语言";
  const failureTitle = "Translation unavailable · 翻译加载失败";
  // Material Icons "translate", drawn like the go-to-top arrow next to it
  const icon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
    + '<path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>'
    + "</svg>";
  // Labels on translated content, in the translation's language
  const attributions = { zh: "由 DeepSeek 翻译", en: "Translated by DeepSeek" };
  const tags = { zh: "DeepSeek 译", en: "Translated by DeepSeek" };

  // The mode this reader picked, else the first of their browser languages
  // that is Chinese or English, else the original
  const preferredMode = () => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (modes.includes(saved)) {
        return saved;
      }
    } catch (error) {
      // Storage can be blocked; fall back to the browser languages
    }
    return browserLanguage() || "original";
  };

  // The first of the reader's browser languages that is Chinese or English
  function browserLanguage() {
    const languages = navigator.languages?.length ? navigator.languages : [navigator.language || ""];
    return languages
      .map((language) => language.toLowerCase().slice(0, 2))
      .find((language) => language === "zh" || language === "en");
  }

  // "Original" in the reader's language: the mode's, else the browser's
  const originalLabel = (mode) => menuLabels.original[mode === "original" ? browserLanguage() || "en" : mode];

  const rememberMode = (mode) => {
    try {
      localStorage.setItem(storageKey, mode);
    } catch (error) {
      // Not remembered when storage is blocked
    }
  };

  // Shows the mode in the address bar, without reloading or adding to the
  // history: no parameter for the original, ?lang=en or ?lang=zh otherwise
  const showModeInURL = (mode) => {
    const url = new URL(location.href);
    if (mode === "original") {
      url.searchParams.delete(urlKey);
    } else {
      url.searchParams.set(urlKey, mode);
    }
    if (url.href !== location.href) {
      history.replaceState(history.state, "", url);
    }
  };

  // The mode a page starts in: the one its address asks for (?lang=en, zh or
  // original), left as given; else the reader's own, shown in the address
  const startMode = () => {
    const asked = new URLSearchParams(location.search).get(urlKey);
    if (modes.includes(asked)) {
      return asked;
    }
    const mode = preferredMode();
    showModeInURL(mode);
    return mode;
  };

  // The floating button and its menu of modes; onSelect gets the picked mode
  const createControl = (onSelect) => {
    const control = document.createElement("div");
    control.className = "translate-control";
    const menu = document.createElement("div");
    menu.className = "translate-menu";
    menu.setAttribute("role", "menu");
    menu.hidden = true;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "translate-toggle";
    button.setAttribute("aria-haspopup", "menu");
    button.setAttribute("aria-expanded", "false");

    const setOpen = (open) => {
      menu.hidden = !open;
      button.setAttribute("aria-expanded", String(open));
    };
    const items = modes.map((mode) => {
      const item = document.createElement("button");
      item.type = "button";
      item.setAttribute("role", "menuitemradio");
      const label = document.createElement("span");
      label.textContent = mode === "original" ? "" : menuLabels[mode];
      item.append(label);
      item.addEventListener("click", () => {
        setOpen(false);
        onSelect(mode);
      });
      menu.append(item);
      return item;
    });
    button.addEventListener("click", () => setOpen(menu.hidden));
    document.addEventListener("click", (event) => {
      if (!control.contains(event.target)) {
        setOpen(false);
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    });
    control.append(menu, button);
    document.body.append(control);

    return {
      // Marks the current mode in the menu, or that a translation failed to
      // load (the button then shows "!")
      show: (mode, failed = false) => {
        button.disabled = false;
        if (failed) {
          button.textContent = "!";
        } else {
          button.innerHTML = icon;
        }
        button.title = failed ? failureTitle : buttonTitle;
        button.setAttribute("aria-label", button.title);
        items.forEach((item, index) => item.setAttribute("aria-checked", String(modes[index] === mode)));
        items[modes.indexOf("original")].firstChild.textContent = originalLabel(mode);
      },
      loading: () => {
        button.disabled = true;
      },
    };
  };

  const fetchJSON = async (url) => {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  };

  // Swaps one part of the page between its original and a translation. The
  // original is read on the first swap, after the page scripts (local times,
  // for one) have run; undefined puts it back.
  const swappable = (read, write) => {
    let original;
    return (value) => {
      if (original === undefined) {
        original = read();
      }
      write(value ?? original);
    };
  };

  // A title's text, without the draft hint or anchor that may follow it
  const titleText = (element) => [...(element?.childNodes || [])]
    .find((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());

  const initPage = () => {
    const original = document.querySelector(".post-single .post-content");
    if (!original) {
      return;
    }

    const lang = script.dataset.lang;
    const titleNode = titleText(document.querySelector(".post-single .post-title"));
    const originalTitle = titleNode?.textContent;
    const caption = document.querySelector(".post-single .entry-cover figcaption");
    const originalCaption = caption?.innerHTML;
    const toc = document.querySelector(".post-single .toc");
    const tocLinks = [...(toc?.querySelectorAll("a[href^='#']") || [])].map((link) => ({
      link,
      id: decodeURIComponent(link.hash.slice(1)),
      text: link.textContent,
    }));
    let translation = null;
    let showing = false;
    let mode = startMode();

    // The line under the meta line while the page shows its translation
    const line = document.createElement("div");
    line.className = "translate-attribution";
    line.textContent = attributions[lang];
    line.hidden = true;
    (document.querySelector(".post-single .post-meta") || document.querySelector(".post-single .post-title"))
      .after(line);

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
      const data = await fetchJSON(script.dataset.src);
      const content = document.createElement("div");
      content.className = "post-content";
      content.lang = lang === "zh" ? "zh-CN" : "en";
      content.innerHTML = data.content;
      // Embeds such as GeoGebra take focus once loaded, which would scroll the
      // page away; load them only when scrolled into view.
      content.querySelectorAll("iframe").forEach((frame) => {
        frame.loading = "lazy";
      });
      return { content, title: data.title, caption: data.caption, headings: alignHeadings(content) };
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

    const swap = (translated) => {
      const [from, to] = translated ? [original, translation.content] : [translation.content, original];
      const spot = readingSpot(from);
      from.replaceWith(to);
      if (titleNode) {
        titleNode.textContent = translated ? translation.title : originalTitle;
      }
      if (caption && translation.caption) {
        caption.innerHTML = translated ? translation.caption : originalCaption;
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
      showing = translated;
    };

    let control;

    const show = async (translated) => {
      if (translated && !translation) {
        control.loading();
        try {
          translation = await load();
        } catch (error) {
          // A later pick from the menu retries
          console.error("Translation failed to load:", error);
          control.show(mode, true);
          return;
        }
      }
      control.show(mode);
      if (translation && translated !== showing) {
        swap(translated);
      }
      line.hidden = !showing;
    };

    control = createControl((picked) => {
      mode = picked;
      rememberMode(mode);
      showModeInURL(mode);
      show(mode === lang);
    });
    control.show(mode);
    if (mode === lang) {
      show(true);
    }
  };

  const initList = () => {
    // Each entry, by the path of the page it links to, with the parts that
    // switch: title and summary for posts, title and note for moments, plus a
    // label while it shows a translation (a moment's location and mood chips
    // stay as written, like tags)
    const entries = [...document.querySelectorAll("article.post-entry, article.first-entry, .archive-entry, article.moment-card")]
      .map((element) => {
        const link = element.querySelector("a.entry-link, a.moment-open");
        if (!link) {
          return null;
        }
        const parts = [];
        const title = titleText(element.querySelector(".entry-header h2, .archive-entry-title, .moment-title"));
        if (title) {
          const set = swappable(() => title.textContent, (text) => { title.textContent = text; });
          parts.push((item) => set(item?.title));
        }
        const summary = element.querySelector(".entry-content p");
        if (summary) {
          const set = swappable(() => summary.textContent, (text) => { summary.textContent = text; });
          parts.push((item) => set(item?.summary));
        }
        const content = element.querySelector(".moment-content");
        if (content) {
          const set = swappable(() => content.innerHTML, (html) => { content.innerHTML = html; });
          parts.push((item) => set(item?.content));
        }
        const meta = element.querySelector(".entry-footer, .archive-meta");
        if (content) {
          // A moment says so in a line under its note, as a translated page
          // does under its meta line, leaving the date and chips as they are
          const line = document.createElement("div");
          line.className = "translate-attribution";
          line.hidden = true;
          content.after(line);
          parts.push((item) => {
            line.textContent = item ? attributions[item.lang] : "";
            line.hidden = !item;
          });
        } else if (meta) {
          parts.push((item) => {
            meta.querySelector(".translate-tag")?.remove();
            if (item) {
              const tag = document.createElement("span");
              tag.className = "translate-tag";
              tag.textContent = `\u00a0·\u00a0${tags[item.lang]}`;
              meta.append(tag);
            }
          });
        }
        return { path: new URL(link.href, location.href).pathname, parts };
      })
      .filter(Boolean);
    if (!entries.length) {
      return;
    }

    let list = null;
    let mode = startMode();
    let control;

    const show = async () => {
      if (mode !== "original" && !list) {
        control.loading();
        try {
          list = await fetchJSON(script.dataset.list);
        } catch (error) {
          // A later pick from the menu retries
          console.error("Translations failed to load:", error);
          control.show(mode, true);
          return;
        }
      }
      control.show(mode);
      for (const { path, parts } of entries) {
        const item = mode !== "original" && list[path]?.lang === mode ? list[path] : undefined;
        parts.forEach((part) => part(item));
      }
      window.__siteMediaLightbox?.reload();
    };

    control = createControl((picked) => {
      mode = picked;
      rememberMode(mode);
      showModeInURL(mode);
      show();
    });
    control.show(mode);
    if (mode !== "original") {
      show();
    }
  };

  const init = () => (script.dataset.list ? initList() : initPage());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
