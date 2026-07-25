/* Hypothesis Testing Guide — application shell.
   Content lives in data.js (window.HTG_DATA). */
(() => {
  "use strict";

  const { UI_TEXT, METHODS, CONCEPTS, GUIDANCE, HELP_CONTENT, P_ALPHA_TEXT, P_ALPHA_STORY } = window.HTG_DATA;
  const BASE_PATH = "/features/hypothesis-testing-guide/";
  const METHOD_ORDER = Object.keys(METHODS);
  const CONCEPT_ENTRIES = Object.values(CONCEPTS);

  /* ---------- language & generic helpers ---------- */

  function text(value, lang) {
    if (value == null) return "";
    if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, lang)) return value[lang];
    return value;
  }
  function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>\"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  }
  function getLang() {
    const urlLang = new URLSearchParams(location.search).get("lang");
    if (urlLang === "en" || urlLang === "zh") return urlLang;
    return localStorage.getItem("hypothesis-guide-lang") || "zh";
  }
  let LANG = getLang();
  const T = () => UI_TEXT[LANG];
  function localized(value) { return text(value, LANG); }
  function safeUrl(path, params = {}) {
    const q = new URLSearchParams(params); q.set("lang", LANG);
    return `${BASE_PATH}${path}${q.toString() ? `?${q}` : ""}`;
  }
  function status(message) { const node = document.getElementById("page-status"); if (node) node.textContent = message; }
  function pageTitle(kind) {
    if (kind === "help") return LANG === "zh" ? "统计推断帮助中心｜假设检验指南" : "Statistical inference help | Hypothesis Testing Guide";
    return LANG === "zh" ? "假设检验指南｜方法探索" : "Hypothesis Testing Guide | Explore methods";
  }
  function syncLanguage() {
    document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
    localStorage.setItem("hypothesis-guide-lang", LANG);
    document.querySelectorAll("[data-i18n]").forEach(node => { const value = T()[node.dataset.i18n]; if (value != null) node.textContent = value; });
    document.querySelectorAll("[data-i18n-aria-label]").forEach(node => { const value = T()[node.dataset.i18nAriaLabel]; if (value != null) node.setAttribute("aria-label", value); });
    const toggle = document.getElementById("language-toggle");
    if (toggle) { toggle.setAttribute("aria-label", T().switchLanguage); toggle.title = T().switchLanguage; toggle.querySelector(".lang-current").textContent = T().languageCode; }
    const searchButton = document.getElementById("palette-button");
    if (searchButton) searchButton.setAttribute("aria-label", `${T().searchButton} (${T().searchKbd})`);
    document.querySelectorAll("[data-nav]").forEach(node => {
      node.href = node.dataset.nav === "help" ? safeUrl("help.html") : safeUrl("index.html");
    });
  }
  function updatePageMeta(pathWithQuery, pageName) {
    const canonical = document.getElementById("canonical-url"); if (!canonical) return;
    const url = new URL(pathWithQuery, canonical.href).href;
    canonical.href = url;
    const ogUrl = document.querySelector('meta[property="og:url"]'); if (ogUrl) ogUrl.content = url;
    const ogTitle = document.querySelector('meta[property="og:title"]'); if (ogTitle) ogTitle.content = document.title;
    const ld = document.getElementById("method-structured-data");
    if (ld && pageName) { try { const data = JSON.parse(ld.textContent); data.name = pageName; data.url = url; ld.textContent = JSON.stringify(data); } catch (_) {} }
  }
  function rerenderMath(root = document.getElementById("app")) {
    if (!root) return;
    root.querySelectorAll("[data-tex], [data-tex-inline]").forEach(node => {
      const inline = node.hasAttribute("data-tex-inline");
      const source = (inline ? node.dataset.texInline : node.dataset.tex) || node.textContent || "";
      if (window.katex) {
        try { window.katex.render(source, node, { displayMode: !inline, throwOnError: false, strict: "warn" }); return; } catch (_) {}
      }
      node.textContent = source;
      node.classList.add("formula-fallback");
    });
  }
  async function copyText(value, button) {
    let copied = false;
    try { await navigator.clipboard.writeText(value); copied = true; } catch (_) {
      const scratch = document.createElement("textarea");
      scratch.value = value; scratch.setAttribute("readonly", ""); scratch.style.cssText = "position:fixed;opacity:0;pointer-events:none";
      document.body.appendChild(scratch); scratch.select();
      try { copied = document.execCommand("copy"); } catch (_) {}
      scratch.remove();
    }
    status(T()[copied ? "copied" : "copyFailed"]);
    if (!button) return;
    const original = button.textContent;
    button.textContent = T()[copied ? "copied" : "copyFailed"];
    setTimeout(() => { button.textContent = original; }, 1400);
  }

  /* ---------- shared content helpers ---------- */

  function methodDisplay(id) { return METHODS[id] ? { id, item: METHODS[id], copy: METHODS[id][LANG] } : CONCEPTS[id] ? { id, item: CONCEPTS[id], copy: CONCEPTS[id][LANG] } : null; }
  function allEntries() { return [...Object.values(METHODS), ...CONCEPT_ENTRIES]; }
  function normalizeSearch(value) {
    return String(value ?? "").normalize("NFKC").toLowerCase().replace(/[’']/g, "").replace(/[^\p{L}\p{N}]+/gu, " ").replace(/\s+/g, " ").trim();
  }
  function compactSearch(value) { return normalizeSearch(value).replace(/\s+/g, ""); }
  function searchFields(item) {
    const primary = [item.id, item.zh?.name, item.en?.name].filter(Boolean);
    const aliases = (item.aliases || []).filter(Boolean);
    const supporting = [
      item.family, item.category?.zh, item.category?.en,
      item.zh?.short, item.en?.short, item.zh?.background, item.en?.background,
      item.zh?.useWhen, item.en?.useWhen, item.zh?.hypotheses?.h0, item.zh?.hypotheses?.h1,
      item.en?.hypotheses?.h0, item.en?.hypotheses?.h1, item.python,
      ...(item.formulas || []).flatMap(value => [value.tex, value.label?.zh, value.label?.en]),
      ...(item.symbols || []).flatMap(value => [value.symbol, value.meaning?.zh, value.meaning?.en])
    ].filter(Boolean);
    return { primary, aliases, supporting };
  }
  function searchScore(item, query) {
    const q = normalizeSearch(query); const compactQ = compactSearch(query); if (!q) return 1;
    const fields = searchFields(item);
    const normalizedPrimary = fields.primary.map(normalizeSearch); const normalizedAliases = fields.aliases.map(normalizeSearch);
    const exact = values => values.some(value => value === q || compactSearch(value) === compactQ);
    if (exact(normalizedPrimary)) return 120;
    if (exact(normalizedAliases)) return 100;
    if (normalizedPrimary.some(value => value.startsWith(q) || compactSearch(value).startsWith(compactQ))) return 90;
    if (normalizedAliases.some(value => value.startsWith(q) || compactSearch(value).startsWith(compactQ))) return 80;
    const corpus = normalizeSearch([...fields.primary, ...fields.aliases, ...fields.supporting].join(" "));
    const corpusTokens = new Set(corpus.split(" ").filter(Boolean)); const queryTokens = q.split(" ").filter(Boolean);
    if (queryTokens.every(token => corpusTokens.has(token))) return 60;
    if (compactQ.length >= 2 && compactSearch(corpus).includes(compactQ)) return 40;
    if (queryTokens.every(token => token.length > 1 && corpus.includes(token))) return 20;
    return 0;
  }

  /* Goal facet: which questions a method can answer. */
  const GOAL_OVERRIDES = {
    chi_square_independence: ["compare", "association"],
    fisher_exact: ["compare", "association"],
    chi_square_goodness: ["compare", "diagnostic"],
    ks_test: ["compare", "diagnostic"]
  };
  const KIND_TO_GOAL = { comparison: "compare", association: "association", diagnostic: "diagnostic" };
  function goalsOf(item) {
    if (GOAL_OVERRIDES[item.id]) return GOAL_OVERRIDES[item.id];
    const goal = KIND_TO_GOAL[item.kind];
    return goal ? [goal] : [];
  }

  function methodCard(item, { state = "", reason = "", recommended = false } = {}) {
    const copy = item[LANG] || item.zh;
    const classes = ["method-card"];
    if (state) classes.push(state);
    return `<a class="${classes.join(" ")}" href="${safeUrl("method.html", { id: item.id })}" data-method-link="${item.id}">
      <span class="method-card-top">${recommended ? `<span class="method-card-tag">${escapeHTML(T().recommendedTag)}</span>` : ""}<span class="method-card-title">${escapeHTML(copy.name)}</span></span>
      <span class="method-card-short">${escapeHTML(copy.short)}</span>
      <span class="method-card-meta"><span>${escapeHTML(localized(item.category))}</span>${reason ? `<em>${escapeHTML(reason)}</em>` : ""}</span>
    </a>`;
  }

  /* ---------- command palette ---------- */

  let paletteState = null;
  function paletteSources() {
    const items = [];
    METHOD_ORDER.forEach(id => { const m = METHODS[id]; items.push({ group: "paletteMethods", id, title: m[LANG].name, meta: localized(m.category), href: safeUrl("method.html", { id }), entry: m }); });
    CONCEPT_ENTRIES.forEach(c => items.push({ group: "paletteConcepts", id: c.id, title: c[LANG].name, meta: localized(c.category), href: safeUrl("method.html", { id: c.id }), entry: c }));
    HELP_CONTENT.filter(section => section.id !== "pvalue").forEach(section => {
      items.push({ group: "paletteHelp", id: `help-${section.id}`, title: localized(section.title), meta: T().navHelp, href: `${safeUrl("help.html")}#${section.id}` });
    });
    return items;
  }
  function ensurePalette() {
    if (document.getElementById("palette-root")) return;
    const root = document.createElement("div");
    root.id = "palette-root";
    root.innerHTML = `<div class="palette-backdrop" data-palette-close hidden></div>
      <div class="palette" role="dialog" aria-modal="true" aria-label="${escapeHTML(T().paletteTitle)}" hidden>
        <div class="palette-input-row">
          <span class="palette-glyph" aria-hidden="true">⌕</span>
          <input id="palette-input" type="text" autocomplete="off" spellcheck="false" placeholder="${escapeHTML(T().palettePlaceholder)}" role="combobox" aria-expanded="true" aria-controls="palette-list" aria-activedescendant="">
          <button class="palette-esc" type="button" data-palette-close>Esc</button>
        </div>
        <ul id="palette-list" role="listbox" aria-label="${escapeHTML(T().paletteTitle)}"></ul>
        <p class="palette-hint">${escapeHTML(T().paletteHint)}</p>
      </div>`;
    document.body.appendChild(root);
    root.querySelectorAll("[data-palette-close]").forEach(el => el.addEventListener("click", closePalette));
    const input = root.querySelector("#palette-input");
    input.addEventListener("input", () => renderPaletteList(input.value));
    input.addEventListener("keydown", event => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        movePaletteSelection(event.key === "ArrowDown" ? 1 : -1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        const active = root.querySelector('#palette-list [aria-selected="true"] a');
        if (active) location.href = active.getAttribute("href");
      }
    });
  }
  function renderPaletteList(query) {
    const list = document.getElementById("palette-list"); if (!list) return;
    const sources = paletteSources();
    let results;
    if (normalizeSearch(query)) {
      results = sources
        .map(item => ({ ...item, score: item.entry ? searchScore(item.entry, query) : (normalizeSearch(item.title).includes(normalizeSearch(query)) ? 70 : 0) }))
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score);
    } else {
      results = sources;
    }
    if (!results.length) { list.innerHTML = `<li class="palette-empty">${escapeHTML(T().paletteEmpty)}</li>`; return; }
    let html = ""; let lastGroup = null; let index = 0;
    results.forEach(item => {
      if (item.group !== lastGroup) { html += `<li class="palette-group" role="presentation">${escapeHTML(T()[item.group])}</li>`; lastGroup = item.group; }
      html += `<li id="palette-item-${index}" role="option" aria-selected="${index === 0}"><a href="${item.href}"><span class="palette-item-title">${escapeHTML(item.title)}</span><span class="palette-item-meta">${escapeHTML(item.meta)}</span></a></li>`;
      index += 1;
    });
    list.innerHTML = html;
    syncPaletteActiveDescendant();
  }
  function syncPaletteActiveDescendant() {
    const active = document.querySelector('#palette-list [aria-selected="true"]');
    const input = document.getElementById("palette-input");
    if (input) input.setAttribute("aria-activedescendant", active ? active.id : "");
  }
  function movePaletteSelection(delta) {
    const options = [...document.querySelectorAll('#palette-list [role="option"]')];
    if (!options.length) return;
    const current = options.findIndex(option => option.getAttribute("aria-selected") === "true");
    const next = Math.min(options.length - 1, Math.max(0, (current < 0 ? 0 : current) + delta));
    options.forEach((option, i) => option.setAttribute("aria-selected", String(i === next)));
    options[next].scrollIntoView({ block: "nearest" });
    syncPaletteActiveDescendant();
  }
  function openPalette() {
    ensurePalette();
    paletteState = { previousFocus: document.activeElement };
    const root = document.getElementById("palette-root");
    root.querySelector(".palette-backdrop").hidden = false;
    root.querySelector(".palette").hidden = false;
    document.body.classList.add("palette-open");
    const input = root.querySelector("#palette-input");
    input.value = "";
    renderPaletteList("");
    input.focus();
  }
  function closePalette() {
    const root = document.getElementById("palette-root"); if (!root) return;
    root.querySelector("#palette-input")?.blur();
    root.querySelector(".palette-backdrop").hidden = true;
    root.querySelector(".palette").hidden = true;
    document.body.classList.remove("palette-open");
    if (paletteState?.previousFocus?.focus) paletteState.previousFocus.focus();
    paletteState = null;
  }
  function paletteIsOpen() { return !!document.querySelector("#palette-root .palette:not([hidden])"); }

  /* ---------- explorer (index page) ---------- */

  const FACETS = [
    { key: "goal", label: "facetGoal", options: [["compare", "goalCompare"], ["association", "goalAssociation"], ["diagnostic", "goalDiagnostic"]] },
    { key: "outcome", label: "facetOutcome", options: [["continuous", "outcomeContinuous"], ["ordinal", "outcomeOrdinal"], ["categorical", "outcomeCategorical"], ["count", "outcomeCount"]] },
    { key: "design", label: "facetDesign", options: [["one", "designOne"], ["independent", "designIndependent"], ["paired", "designPaired"]] },
    { key: "groups", label: "facetGroups", options: [["one", "groupsOne"], ["two", "groupsTwo"], ["many", "groupsMany"]] }
  ];
  const EXPLORER_STATE = { goal: null, outcome: null, design: null, groups: null, query: "" };

  function facetsSelected() { return FACETS.some(facet => EXPLORER_STATE[facet.key]); }
  function matchedGuidance() {
    return GUIDANCE.filter(rule => {
      const keys = Object.keys(rule.when);
      const nonGoal = keys.filter(key => key !== "goal");
      if (!nonGoal.every(key => EXPLORER_STATE[key] === rule.when[key])) return false;
      if ("goal" in rule.when) {
        if (nonGoal.length === 0) return EXPLORER_STATE.goal === rule.when.goal;
        return EXPLORER_STATE.goal === null || EXPLORER_STATE.goal === rule.when.goal;
      }
      return true;
    });
  }
  function methodFacetState(item, recommendedSet) {
    if (recommendedSet.has(item.id)) return { state: "is-recommended", recommended: true, reason: "" };
    if (item.kind === "effect") return facetsSelected() ? { state: "is-dim", reason: T().dimConcept } : { state: "", reason: "" };
    if (EXPLORER_STATE.goal && !goalsOf(item).includes(EXPLORER_STATE.goal)) return { state: "is-dim", reason: T().dimGoal };
    if (EXPLORER_STATE.outcome && item.outcome !== EXPLORER_STATE.outcome) return { state: "is-dim", reason: T().dimOutcome };
    if (EXPLORER_STATE.design && item.design !== EXPLORER_STATE.design) return { state: "is-dim", reason: T().dimDesign };
    if (EXPLORER_STATE.groups && item.groups !== EXPLORER_STATE.groups) return { state: "is-dim", reason: T().dimGroups };
    return { state: facetsSelected() ? "is-match" : "", reason: "" };
  }
  function guidanceHtml(rules) {
    if (!rules.length) {
      const hint = EXPLORER_STATE.goal || facetsSelected() ? T().guidanceEmpty : T().guidanceGoalHint;
      return `<div class="guidance-empty">${escapeHTML(hint)}</div>`;
    }
    return rules.map(rule => `<section class="guidance-rule">
      <h3>${escapeHTML(localized(rule.title))}</h3>
      ${rule.intro ? `<p class="guidance-intro">${escapeHTML(localized(rule.intro))}</p>` : ""}
      <ol class="guidance-options">${rule.options.map(option => {
        const body = `<strong>${escapeHTML(localized(option.label))}</strong><span>${escapeHTML(localized(option.desc))}</span>`;
        if (option.method) {
          const method = METHODS[option.method];
          return `<li><a class="guidance-option" href="${safeUrl("method.html", { id: option.method })}">${body}<b class="guidance-method">${escapeHTML(method[LANG].name)} →</b></a></li>`;
        }
        return `<li><div class="guidance-option is-note">${body}<b class="guidance-note">${escapeHTML(localized(option.note))}</b></div></li>`;
      }).join("")}</ol>
    </section>`).join("");
  }
  function explorerResults() {
    const query = EXPLORER_STATE.query;
    const rules = matchedGuidance();
    const recommendedSet = new Set(rules.flatMap(rule => rule.options.map(option => option.method).filter(Boolean)));
    let methods = METHOD_ORDER.map(id => METHODS[id]);
    let concepts = CONCEPT_ENTRIES;
    if (normalizeSearch(query)) {
      const rank = item => searchScore(item, query);
      methods = methods.map(item => [item, rank(item)]).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).map(([item]) => item);
      concepts = concepts.map(item => [item, rank(item)]).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).map(([item]) => item);
    }
    const decorated = methods.map(item => ({ item, ...methodFacetState(item, recommendedSet) }));
    if (!normalizeSearch(query)) {
      decorated.sort((a, b) => {
        const rankOf = entry => entry.state === "is-recommended" ? 0 : entry.state === "is-dim" ? 2 : 1;
        return rankOf(a) - rankOf(b) || METHOD_ORDER.indexOf(a.item.id) - METHOD_ORDER.indexOf(b.item.id);
      });
    }
    return { rules, decorated, concepts };
  }
  function renderExplorerResults(root) {
    const { rules, decorated, concepts } = explorerResults();
    const grid = root.querySelector("#explorer-grid");
    const conceptWrap = root.querySelector("#explorer-concepts");
    const rail = root.querySelector("#guidance-rail");
    const count = root.querySelector("#explorer-count");
    const matchable = decorated.filter(entry => entry.state !== "is-dim").length;
    count.textContent = facetsSelected() || normalizeSearch(EXPLORER_STATE.query)
      ? `${matchable} ${T().matchedCount}`
      : `${decorated.length} ${T().allCount}`;
    grid.innerHTML = decorated.length
      ? decorated.map(entry => methodCard(entry.item, entry)).join("")
      : `<p class="empty-state">${escapeHTML(T().noResults)}</p>`;
    conceptWrap.innerHTML = concepts.length
      ? `<h2>${escapeHTML(T().conceptsTitle)}</h2><div class="method-grid">${concepts.map(item => methodCard(item, facetsSelected() ? { state: "is-dim", reason: T().dimConcept } : {})).join("")}</div>`
      : "";
    rail.innerHTML = `<h2>${escapeHTML(T().guidanceTitle)}</h2>${guidanceHtml(rules)}`;
    const srStatus = root.querySelector("#explorer-status");
    if (srStatus) srStatus.textContent = count.textContent;
  }
  function renderExplorer() {
    const root = document.getElementById("app"); if (!root) return;
    document.title = pageTitle("index");
    const facetGroups = FACETS.map(facet => `<fieldset class="facet-group" data-facet="${facet.key}">
        <legend>${escapeHTML(T()[facet.label])}</legend>
        <div class="facet-options">${facet.options.map(([value, key]) => `<button type="button" class="facet-chip" data-facet-key="${facet.key}" data-facet-value="${value}" aria-pressed="${EXPLORER_STATE[facet.key] === value}">${escapeHTML(T()[key])}</button>`).join("")}</div>
      </fieldset>`).join("");
    root.innerHTML = `<section class="explore-hero">
        <p class="eyebrow">${escapeHTML(T().exploreEyebrow)}</p>
        <h1>${escapeHTML(T().exploreTitle)}</h1>
        <p class="lead">${escapeHTML(T().exploreLead)}</p>
      </section>
      <div class="facet-panel" aria-label="${escapeHTML(T().exploreEyebrow)}">
        <div class="facet-groups">${facetGroups}</div>
        <div class="facet-actions">
          <label class="inline-search">
            <span class="sr-only">${escapeHTML(T().inlineSearchLabel)}</span>
            <input id="explorer-search" type="search" autocomplete="off" placeholder="${escapeHTML(T().inlineSearchPlaceholder)}" value="${escapeHTML(EXPLORER_STATE.query)}">
          </label>
          <button type="button" class="button ghost" id="explorer-reset">${escapeHTML(T().resetFilters)}</button>
        </div>
        <p class="facet-hint">${escapeHTML(T().inlineSearchHint)}</p>
      </div>
      <div class="explorer-layout">
        <section class="explorer-main">
          <div class="explorer-count-row"><h2 class="section-label">${escapeHTML(T().methodsTitle)}</h2><span id="explorer-count" class="explorer-count"></span></div>
          <p id="explorer-status" class="sr-only" role="status" aria-live="polite"></p>
          <div id="explorer-grid" class="method-grid"></div>
          <section id="explorer-concepts" class="explorer-concepts"></section>
        </section>
        <aside id="guidance-rail" class="guidance-rail" aria-live="polite"></aside>
      </div>`;
    root.querySelectorAll(".facet-chip").forEach(chip => chip.addEventListener("click", () => {
      const { facetKey, facetValue } = chip.dataset;
      EXPLORER_STATE[facetKey] = EXPLORER_STATE[facetKey] === facetValue ? null : facetValue;
      root.querySelectorAll(`.facet-chip[data-facet-key="${facetKey}"]`).forEach(other => other.setAttribute("aria-pressed", String(other.dataset.facetValue === EXPLORER_STATE[facetKey])));
      renderExplorerResults(root);
    }));
    const search = root.querySelector("#explorer-search");
    search.addEventListener("input", () => { EXPLORER_STATE.query = search.value; renderExplorerResults(root); });
    search.addEventListener("keydown", event => { if (event.key === "Escape") { search.value = ""; EXPLORER_STATE.query = ""; renderExplorerResults(root); } });
    root.querySelector("#explorer-reset").addEventListener("click", () => {
      FACETS.forEach(facet => { EXPLORER_STATE[facet.key] = null; });
      EXPLORER_STATE.query = "";
      search.value = "";
      root.querySelectorAll(".facet-chip").forEach(chip => chip.setAttribute("aria-pressed", "false"));
      renderExplorerResults(root);
    });
    renderExplorerResults(root);
    if (new URLSearchParams(location.search).get("focus") === "search") search.focus();
  }

  /* ---------- method page ---------- */

  function tableHtml(example) {
    if (!example) return ""; const cols = localized(example.columns);
    return `<div class="table-wrap"><table><caption>${escapeHTML(localized(example.caption))}</caption><thead><tr>${cols.map(c => `<th scope="col">${escapeHTML(c)}</th>`).join("")}</tr></thead><tbody>${example.rows.map(row => `<tr>${row.map(cell => `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  function listHtml(value) { const values = Array.isArray(value) ? value : [value]; return `<ul class="check-list">${values.map(x => `<li>${escapeHTML(x)}</li>`).join("")}</ul>`; }
  function formulaHtml(formulas) { return `<div class="formula-list">${(formulas || []).map(f => `<div class="formula-card"><h3>${escapeHTML(localized(f.label))}</h3><div class="formula" data-tex="${escapeHTML(f.tex)}">${escapeHTML(f.tex)}</div></div>`).join("")}</div>`; }
  function symbolsHtml(symbols) { return `<div class="symbol-grid">${(symbols || []).map(s => `<div><code data-tex-inline="${escapeHTML(s.symbol)}">${escapeHTML(s.symbol)}</code><span>${escapeHTML(localized(s.meaning))}</span></div>`).join("")}</div>`; }
  function renderCode(code) { return code ? `<div class="code-block"><div class="code-toolbar"><span>Python</span><button class="button tiny" type="button" data-copy-code>${escapeHTML(T().copy)}</button></div><pre><code>${escapeHTML(code)}</code></pre></div>` : ""; }
  function relatedCards(ids) {
    const related = (ids || []).filter(x => METHODS[x] || CONCEPTS[x]);
    return related.map(id => methodCard(METHODS[id] || CONCEPTS[id])).join("");
  }
  function facetChipLabel(kind, value) {
    const map = {
      outcome: { continuous: "outcomeContinuous", ordinal: "outcomeOrdinal", categorical: "outcomeCategorical", count: "outcomeCount" },
      design: { one: "designOne", independent: "designIndependent", paired: "designPaired" },
      groups: { one: "groupsOne", two: "groupsTwo", many: "groupsMany" }
    };
    const key = map[kind]?.[value];
    return key ? T()[key] : value;
  }
  function bindScrollSpy(article) {
    const links = [...document.querySelectorAll(".method-toc a[href^='#']")];
    if (!links.length || !("IntersectionObserver" in window)) return;
    const byId = new Map(links.map(link => [link.getAttribute("href").slice(1), link]));
    const setActive = id => links.forEach(link => link.toggleAttribute("data-active", link.getAttribute("href") === `#${id}`));
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) setActive(visible[0].target.id);
    }, { rootMargin: "-20% 0px -65% 0px" });
    article.querySelectorAll("section[id]").forEach(section => { if (byId.has(section.id)) observer.observe(section); });
  }
  function renderMethod() {
    const root = document.getElementById("app"); if (!root) return;
    const id = new URLSearchParams(location.search).get("id") || "paired_t";
    const found = methodDisplay(id);
    if (!found) {
      document.title = LANG === "zh" ? "方法不存在｜假设检验指南" : "Method not found | Hypothesis Testing Guide";
      root.innerHTML = `<section class="empty-state"><h1>${escapeHTML(T().notFound)}</h1><p>${escapeHTML(T().notFoundText)}</p><a class="button primary" href="${safeUrl("index.html")}">${escapeHTML(T().backToLibrary)}</a></section>`;
      return;
    }
    const { item, copy } = found;
    document.title = LANG === "zh" ? `${copy.name}｜假设检验指南` : `${copy.name} | Hypothesis Testing Guide`;
    updatePageMeta(`method.html?id=${encodeURIComponent(item.id || id)}`, copy.name);
    const order = METHOD_ORDER.indexOf(item.id);
    const prev = order > 0 ? METHODS[METHOD_ORDER[order - 1]] : null;
    const next = order >= 0 && order < METHOD_ORDER.length - 1 ? METHODS[METHOD_ORDER[order + 1]] : null;
    const sections = [
      ["background", T().background], ["use", T().sectionUse], ["hypotheses", T().hypotheses],
      ["assumptions", T().assumptions], ["formula", T().formula], ["example", T().example],
      ["inference", T().inference], ["edge-cases", T().edgeCases], ["reporting", T().reporting],
      ["python", T().python], ["related", T().related], ["references", T().references]
    ];
    const chips = [
      localized(item.category),
      item.outcome ? facetChipLabel("outcome", item.outcome) : "",
      item.design ? facetChipLabel("design", item.design) : "",
      item.groups ? facetChipLabel("groups", item.groups) : ""
    ].filter(Boolean);
    root.innerHTML = `<div class="method-layout">
      <aside class="method-toc" aria-label="${escapeHTML(T().contents)}">
        <p class="toc-label">${escapeHTML(T().contents)}</p>
        <nav>${sections.map(([sid, label], index) => `<a href="#${sid}"><i>${String(index + 1).padStart(2, "0")}</i>${escapeHTML(label)}</a>`).join("")}</nav>
      </aside>
      <article class="method-article" id="method-article">
        <p class="breadcrumbs"><a href="${safeUrl("index.html")}">${escapeHTML(T().methodLibrary)}</a><span aria-hidden="true">/</span><span>${escapeHTML(localized(item.category))}</span></p>
        <header class="method-header">
          <h1>${escapeHTML(copy.name)}</h1>
          <p class="lead">${escapeHTML(copy.short)}</p>
          <div class="chip-row">${chips.map(chip => `<span class="chip">${escapeHTML(chip)}</span>`).join("")}</div>
        </header>
        <section id="background" class="content-section"><h2><i>01</i>${escapeHTML(T().background)}</h2><p>${escapeHTML(copy.background)}</p></section>
        <section id="use" class="content-section"><h2><i>02</i>${escapeHTML(T().sectionUse)}</h2><div class="two-col"><div class="info-card is-use"><h3>${escapeHTML(T().useWhen)}</h3><p>${escapeHTML(copy.useWhen)}</p></div><div class="info-card is-avoid"><h3>${escapeHTML(T().avoidWhen)}</h3><p>${escapeHTML(copy.avoidWhen)}</p></div></div></section>
        <section id="hypotheses" class="content-section"><h2><i>03</i>${escapeHTML(T().hypotheses)}</h2><div class="two-col"><div class="hypothesis h0"><h3>${escapeHTML(T().nullHypothesis)}</h3><p>${escapeHTML(copy.hypotheses.h0)}</p></div><div class="hypothesis h1"><h3>${escapeHTML(T().alternativeHypothesis)}</h3><p>${escapeHTML(copy.hypotheses.h1)}</p></div></div></section>
        <section id="assumptions" class="content-section"><h2><i>04</i>${escapeHTML(T().assumptions)}</h2>${listHtml(copy.assumptions)}</section>
        <section id="formula" class="content-section"><h2><i>05</i>${escapeHTML(T().formula)}</h2>${symbolsHtml(item.symbols)}${formulaHtml(item.formulas)}<p class="hint">${escapeHTML(copy.formulaNotes)}</p></section>
        <section id="example" class="content-section"><h2><i>06</i>${escapeHTML(T().example)}</h2>${tableHtml(item.example)}<ol class="calculation-steps">${(localized(item.example?.steps) || []).map(step => `<li>${escapeHTML(step)}</li>`).join("")}</ol><div class="result-callout"><strong>${escapeHTML(T().result)}</strong>${item.example?.approximate ? `<span class="approx-tag">${escapeHTML(T().approximate)}</span>` : ""}<p>${escapeHTML(localized(item.example?.result))}</p></div></section>
        <section id="inference" class="content-section"><h2><i>07</i>${escapeHTML(T().inference)}</h2><p>${escapeHTML(copy.inference)}</p><p>${escapeHTML(copy.ci)}</p><p>${escapeHTML(copy.effect)}</p></section>
        <section id="edge-cases" class="content-section"><h2><i>08</i>${escapeHTML(T().edgeCases)}</h2>${listHtml(copy.edgeCases)}</section>
        <section id="reporting" class="content-section"><h2><i>09</i>${escapeHTML(T().reporting)}</h2><div class="report-box"><p>${escapeHTML(copy.report)}</p><button class="button tiny" type="button" data-copy-report>${escapeHTML(T().copy)}</button></div></section>
        <section id="python" class="content-section"><h2><i>10</i>${escapeHTML(T().python)}</h2>${renderCode(item.python)}</section>
        <section id="related" class="content-section"><h2><i>11</i>${escapeHTML(T().related)}</h2><div class="method-grid compact">${relatedCards(item.related)}</div></section>
        <section id="references" class="content-section"><h2><i>12</i>${escapeHTML(T().references)}</h2><ul class="reference-list">${(item.references || []).map(r => `<li><a href="${r.url}" target="_blank" rel="noreferrer">${escapeHTML(localized(r.label))}</a></li>`).join("")}</ul></section>
        <nav class="method-pager" aria-label="${escapeHTML(T().related)}">
          ${prev ? `<a class="pager-link is-prev" href="${safeUrl("method.html", { id: prev.id })}"><small>${escapeHTML(T().prevMethod)}</small><span>${escapeHTML(prev[LANG].name)}</span></a>` : "<span></span>"}
          ${next ? `<a class="pager-link is-next" href="${safeUrl("method.html", { id: next.id })}"><small>${escapeHTML(T().nextMethod)}</small><span>${escapeHTML(next[LANG].name)}</span></a>` : "<span></span>"}
        </nav>
      </article>
    </div>`;
    root.querySelectorAll("[data-copy-code]").forEach(btn => btn.addEventListener("click", () => copyText(item.python, btn)));
    root.querySelector("[data-copy-report]")?.addEventListener("click", () => copyText(copy.report, root.querySelector("[data-copy-report]")));
    bindScrollSpy(root.querySelector("#method-article"));
    rerenderMath();
  }
  const P_ALPHA_STATE = { distribution: "normal", alternative: "two", observed: 2, alpha: 0.05, run: 0, timer: null, storyStep: 0 };
  function clamp01(value) { return Math.max(0, Math.min(1, value)); }
  function normalPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  function erfApprox(value) {
    const sign = value < 0 ? -1 : 1; const x = Math.abs(value); const t = 1 / (1 + 0.3275911 * x);
    const poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-x * x));
  }
  function normalCdf(x) { return clamp01(0.5 * (1 + erfApprox(x / Math.sqrt(2)))); }
  function studentPdf5(x) { return (8 / (3 * Math.PI * Math.sqrt(5))) * Math.pow(1 + x * x / 5, -3); }
  function studentCdf5(x) {
    if (x === 0) return 0.5;
    const upper = Math.min(Math.abs(x), 40); const n = 400; const h = upper / n;
    let sum = studentPdf5(0) + studentPdf5(upper);
    for (let i = 1; i < n; i += 1) sum += (i % 2 ? 4 : 2) * studentPdf5(i * h);
    const area = sum * h / 3; return clamp01(x > 0 ? 0.5 + area : 0.5 - area);
  }
  function chiSquarePdf4(x) { return x <= 0 ? 0 : (x / 4) * Math.exp(-x / 2); }
  function chiSquareCdf4(x) { return x <= 0 ? 0 : clamp01(1 - Math.exp(-x / 2) * (1 + x / 2)); }
  function choose(n, k) {
    if (k < 0 || k > n) return 0; let answer = 1;
    for (let i = 1; i <= Math.min(k, n - k); i += 1) answer = answer * (n - i + 1) / i;
    return answer;
  }
  function binomialPmf12(k) { return choose(12, k) / 4096; }
  function pAlphaConfig(id) {
    return {
      normal: { min: -4, max: 4, step: 0.1, initial: 2, pdf: normalPdf, cdf: normalCdf },
      student: { min: -5, max: 5, step: 0.1, initial: 2, pdf: studentPdf5, cdf: studentCdf5 },
      chiSquare: { min: 0, max: 16, step: 0.1, initial: 8, pdf: chiSquarePdf4, cdf: chiSquareCdf4 },
      binomial: { min: 0, max: 12, step: 1, initial: 9, discrete: true }
    }[id];
  }
  function labPValue(distribution, observed, alternative) {
    if (distribution === "binomial") {
      const k = Math.round(observed); const probs = Array.from({ length: 13 }, (_, i) => binomialPmf12(i));
      if (alternative === "greater") return probs.slice(k).reduce((a, b) => a + b, 0);
      if (alternative === "less") return probs.slice(0, k + 1).reduce((a, b) => a + b, 0);
      const threshold = probs[k] + 1e-12; return clamp01(probs.filter(prob => prob <= threshold).reduce((a, b) => a + b, 0));
    }
    const config = pAlphaConfig(distribution); const cdf = config.cdf(observed);
    if (distribution === "chiSquare" || alternative === "greater") return clamp01(1 - cdf);
    if (alternative === "less") return cdf;
    return clamp01(2 * Math.min(cdf, 1 - cdf));
  }
  function pAlphaFormula(distribution, alternative) {
    if (distribution === "binomial") {
      if (alternative === "greater") return "p=\\sum_{j=k_{obs}}^{12}{12 \\choose j}(0.5)^{12}";
      if (alternative === "less") return "p=\\sum_{j=0}^{k_{obs}}{12 \\choose j}(0.5)^{12}";
      return "p=\\sum_{j:\\,P_0(K=j)\\le P_0(K=k_{obs})}P_0(K=j)";
    }
    const symbol = distribution === "chiSquare" ? "\\chi^2" : distribution === "student" ? "T" : "Z";
    if (distribution === "chiSquare" || alternative === "greater") return `p=P_0(${symbol}\\ge ${symbol}_{obs})=1-F_0(${symbol}_{obs})`;
    if (alternative === "less") return `p=P_0(${symbol}\\le ${symbol}_{obs})=F_0(${symbol}_{obs})`;
    return `p=P_0(|${symbol}|\\ge|${symbol}_{obs}|)=2\\min\\{F_0(${symbol}_{obs}),1-F_0(${symbol}_{obs})\\}`;
  }
  function formatProbability(value) { return value < 0.0001 ? "< 0.0001" : value.toFixed(4); }
  function chartAreaPath(points, predicate, xScale, yScale, baseline) {
    const groups = []; let group = [];
    points.forEach(point => { if (predicate(point.x)) group.push(point); else if (group.length) { groups.push(group); group = []; } });
    if (group.length) groups.push(group);
    return groups.filter(items => items.length > 1).map(items => `M ${xScale(items[0].x).toFixed(2)} ${baseline} L ${items.map(point => `${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" L ")} L ${xScale(items.at(-1).x).toFixed(2)} ${baseline} Z`).join(" ");
  }
  function continuousNullChart(distribution, observed, alternative, alpha, copy, idPrefix = "lab") {
    const config = pAlphaConfig(distribution); const width = 760; const height = 280; const left = 48; const right = 24; const top = 20; const baseline = 226;
    const points = Array.from({ length: 241 }, (_, i) => { const x = config.min + (config.max - config.min) * i / 240; return { x, y: config.pdf(x) }; });
    const maxY = Math.max(...points.map(point => point.y)) * 1.12; const xScale = x => left + (x - config.min) / (config.max - config.min) * (width - left - right); const yScale = y => baseline - y / maxY * (baseline - top);
    const pExtreme = x => alternative === "greater" || distribution === "chiSquare" ? x >= observed : alternative === "less" ? x <= observed : Math.abs(x) >= Math.abs(observed);
    const alphaReject = x => labPValue(distribution, x, alternative) <= alpha;
    const alphaPath = chartAreaPath(points, alphaReject, xScale, yScale, baseline); const pPath = chartAreaPath(points, pExtreme, xScale, yScale, baseline);
    const curvePath = points.map((point, index) => `${index ? "L" : "M"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`).join(" ");
    const ticks = Array.from({ length: 5 }, (_, i) => config.min + (config.max - config.min) * i / 4);
    const titleId = `${idPrefix}-chart-title`; const descId = `${idPrefix}-chart-desc`; const hatchId = `${idPrefix}-alpha-hatch`;
    return `<svg class="lab-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descId}"><title id="${titleId}">${escapeHTML(copy.chartLabel)}</title><desc id="${descId}">${escapeHTML(copy.pMeaning)} ${escapeHTML(copy.alphaMeaning)}</desc><defs><pattern id="${hatchId}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line class="alpha-hatch-line" x1="0" y1="0" x2="0" y2="8" /></pattern></defs><line class="lab-axis" x1="${left}" y1="${baseline}" x2="${width - right}" y2="${baseline}" />${alphaPath ? `<path class="lab-alpha-area" style="fill:url(#${hatchId})" d="${alphaPath}" />` : ""}${pPath ? `<path class="lab-p-area" d="${pPath}" />` : ""}<path class="lab-density-line" d="${curvePath}" /><line class="lab-observed-line" x1="${xScale(observed)}" y1="${top}" x2="${xScale(observed)}" y2="${baseline}" /><text class="lab-observed-text" x="${Math.min(width - 90, Math.max(left + 8, xScale(observed) + 7))}" y="${top + 16}">${escapeHTML(copy.observedLabel)} ${Number(observed).toFixed(1)}</text>${ticks.map(tick => `<g><line class="lab-tick" x1="${xScale(tick)}" y1="${baseline}" x2="${xScale(tick)}" y2="${baseline + 6}"/><text class="lab-tick-label" x="${xScale(tick)}" y="${baseline + 24}">${Number(tick).toFixed(tick % 1 ? 1 : 0)}</text></g>`).join("")}</svg>`;
  }
  function discreteNullChart(observed, alternative, alpha, copy, idPrefix = "lab") {
    const width = 760; const height = 280; const left = 42; const right = 22; const baseline = 226; const top = 24; const probs = Array.from({ length: 13 }, (_, k) => binomialPmf12(k)); const maxP = Math.max(...probs) * 1.12; const slot = (width - left - right) / 13; const barWidth = slot * 0.62; const obsP = probs[Math.round(observed)];
    const bars = probs.map((prob, k) => {
      const x = left + k * slot + (slot - barWidth) / 2; const barHeight = prob / maxP * (baseline - top); const inP = alternative === "greater" ? k >= observed : alternative === "less" ? k <= observed : prob <= obsP + 1e-12; const rejected = labPValue("binomial", k, alternative) <= alpha;
      return `<g><rect class="lab-discrete-bar${inP ? " in-p" : ""}${rejected ? " is-rejection" : ""}${k === Math.round(observed) ? " is-observed" : ""}" x="${x}" y="${baseline - barHeight}" width="${barWidth}" height="${barHeight}" /><text class="lab-tick-label" x="${x + barWidth / 2}" y="${baseline + 22}">${k}</text></g>`;
    }).join("");
    const titleId = `${idPrefix}-chart-title`; const descId = `${idPrefix}-chart-desc`;
    return `<svg class="lab-chart" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${titleId} ${descId}"><title id="${titleId}">${escapeHTML(copy.chartLabel)}</title><desc id="${descId}">${escapeHTML(copy.discreteMeaning)}</desc><line class="lab-axis" x1="${left}" y1="${baseline}" x2="${width - right}" y2="${baseline}" />${bars}<text class="lab-observed-text" x="${left + Math.round(observed) * slot + slot / 2}" y="${top}">${escapeHTML(copy.observedLabel)} k=${Math.round(observed)}</text></svg>`;
  }
  function storyVisual(index, step) {
    if (index === 0) {
      const labels = LANG === "zh" ? ["同一人前后测量", "得到配对差值", "这里只记录 + / −", "问题：正差是否 > 50%"] : ["Before–after pairs", "Compute paired differences", "Record only + / −", "Question: positives > 50%?"];
      return `<div class="story-flow" aria-label="${escapeHTML(labels.join(" → "))}">${labels.map((label, i) => `<span>${escapeHTML(label)}</span>${i < labels.length - 1 ? `<b aria-hidden="true">→</b>` : ""}`).join("")}</div>`;
    }
    if (index === 2) {
      return `<div class="story-alpha-visual"><div class="story-alpha-grid" aria-label="${escapeHTML(LANG === "zh" ? "100 次长期重复中，用 5 个标记表示名义 5% 错误水平" : "Five marked cells among 100 illustrate a nominal 5% long-run error level")}">${Array.from({ length: 100 }, (_, i) => `<i class="${i < 5 ? "is-alpha" : ""}" aria-hidden="true"></i>`).join("")}</div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div></div>`;
    }
    if (index === 3) {
      const outcomes = Array.from({ length: 12 }, (_, i) => i < 10 ? "+" : "−");
      return `<div class="story-observations" aria-label="${escapeHTML(LANG === "zh" ? "10 个正差和 2 个负差" : "10 positive and 2 negative differences")}">${outcomes.map(value => `<span class="${value === "+" ? "is-positive" : "is-negative"}">${value}</span>`).join("")}</div><div class="story-binomial-chart">${discreteNullChart(10, "greater", 0.05, P_ALPHA_TEXT[LANG], "story")}</div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div>`;
    }
    if (index === 4) {
      return `<div class="story-comparison"><span>p = 0.0193</span><b aria-hidden="true">&lt;</b><span>α = 0.05</span></div><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div>`;
    }
    return `<div class="story-hypotheses"><div class="formula" data-tex="${escapeHTML(step.formula)}">${escapeHTML(step.formula)}</div></div>`;
  }
  function renderPAlphaStory() {
    const story = P_ALPHA_STORY[LANG]; const count = story.steps.length;
    return `<section class="p-alpha-story" id="p-alpha-start" aria-labelledby="p-alpha-story-title"><header><p class="eyebrow">${escapeHTML(story.eyebrow)}</p><h2 id="p-alpha-story-title">${escapeHTML(story.title)}</h2><p class="lead">${escapeHTML(story.lead)}</p></header><nav class="story-step-nav" aria-label="${escapeHTML(story.stepLabel)}">${story.steps.map((step, index) => `<button type="button" data-story-step="${index}" aria-pressed="${index === P_ALPHA_STATE.storyStep}"><span>${index + 1}</span><small>${escapeHTML(step.title.replace(/^\d+\.\s*/, ""))}</small></button>`).join("")}</nav><div class="story-stage" id="story-stage">${story.steps.map((step, index) => `<article class="story-panel" data-story-panel="${index}"${index === P_ALPHA_STATE.storyStep ? "" : " hidden"}><div class="story-copy"><p class="story-counter">${escapeHTML(story.stepLabel)} ${index + 1} / ${count}</p><h3>${escapeHTML(step.title)}</h3>${listHtml(step.body)}<div class="story-callout">${escapeHTML(step.callout)}</div></div><div class="story-visual">${storyVisual(index, step)}</div></article>`).join("")}</div><div class="story-actions"><button class="button secondary" type="button" data-story-previous>${escapeHTML(story.previous)}</button><span id="story-progress" aria-live="polite">${P_ALPHA_STATE.storyStep + 1} / ${count}</span><button class="button primary" type="button" data-story-next>${escapeHTML(P_ALPHA_STATE.storyStep === count - 1 ? story.restart : story.next)}</button></div></section>`;
  }
  function bindPAlphaStory(root) {
    const story = P_ALPHA_STORY[LANG]; const buttons = [...root.querySelectorAll("[data-story-step]")]; const panels = [...root.querySelectorAll("[data-story-panel]")]; const previous = root.querySelector("[data-story-previous]"); const next = root.querySelector("[data-story-next]"); const progress = root.querySelector("#story-progress");
    if (!buttons.length || !panels.length || !previous || !next || !progress) return;
    const draw = () => {
      buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === P_ALPHA_STATE.storyStep)));
      panels.forEach((panel, index) => { panel.hidden = index !== P_ALPHA_STATE.storyStep; });
      previous.disabled = P_ALPHA_STATE.storyStep === 0; progress.textContent = `${P_ALPHA_STATE.storyStep + 1} / ${story.steps.length}`; next.textContent = P_ALPHA_STATE.storyStep === story.steps.length - 1 ? story.restart : story.next;
    };
    buttons.forEach((button, index) => button.addEventListener("click", () => { P_ALPHA_STATE.storyStep = index; draw(); }));
    previous.addEventListener("click", () => { P_ALPHA_STATE.storyStep = Math.max(0, P_ALPHA_STATE.storyStep - 1); draw(); });
    next.addEventListener("click", () => { P_ALPHA_STATE.storyStep = P_ALPHA_STATE.storyStep === story.steps.length - 1 ? 0 : P_ALPHA_STATE.storyStep + 1; draw(); });
    draw();
  }
  function renderPAlphaLab() {
    const copy = P_ALPHA_TEXT[LANG]; const config = pAlphaConfig(P_ALPHA_STATE.distribution); const observedLabel = P_ALPHA_STATE.distribution === "binomial" ? copy.successes : copy.observed;
    if (P_ALPHA_STATE.timer) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; }
    return `<section class="p-alpha-topic" id="p-alpha"><header class="p-alpha-header"><p class="eyebrow">${escapeHTML(copy.eyebrow)}</p><h2>${escapeHTML(copy.title)}</h2><p class="lead">${escapeHTML(copy.lead)}</p></header><div class="p-alpha-lab"><div class="lab-controls"><label class="lab-control"><span>${escapeHTML(copy.distribution)}</span><select id="lab-distribution"><option value="normal">${escapeHTML(copy.normal)}</option><option value="student">${escapeHTML(copy.student)}</option><option value="chiSquare">${escapeHTML(copy.chiSquare)}</option><option value="binomial">${escapeHTML(copy.binomial)}</option></select></label><label class="lab-control"><span>${escapeHTML(copy.alternative)}</span><select id="lab-alternative"><option value="two">${escapeHTML(copy.twoSided)}</option><option value="greater">${escapeHTML(copy.greater)}</option><option value="less">${escapeHTML(copy.less)}</option></select></label><label class="lab-control range-control"><span><span id="lab-observed-label">${escapeHTML(observedLabel)}</span> <output id="lab-observed-output">${P_ALPHA_STATE.observed}</output></span><input id="lab-observed" type="range" min="${config.min}" max="${config.max}" step="${config.step}" value="${P_ALPHA_STATE.observed}"></label><label class="lab-control range-control"><span>${escapeHTML(copy.alpha)} <output id="lab-alpha-output">${P_ALPHA_STATE.alpha.toFixed(3)}</output></span><input id="lab-alpha" type="range" min="0.001" max="0.200" step="0.001" value="${P_ALPHA_STATE.alpha}"></label></div><div class="lab-readout"><div><span>${escapeHTML(copy.pValue)}</span><strong id="lab-p-output">—</strong></div><div><span>${escapeHTML(copy.alpha)}</span><strong id="lab-alpha-card">${P_ALPHA_STATE.alpha.toFixed(3)}</strong></div><div><span>${escapeHTML(copy.decision)}</span><strong id="lab-decision" aria-live="polite">—</strong></div></div><figure class="lab-figure"><div id="lab-chart-container"></div><figcaption><span class="lab-legend-item p-region"><i aria-hidden="true"></i>${escapeHTML(copy.pRegionLabel)}</span><span class="lab-legend-item alpha-region"><i aria-hidden="true"></i>${escapeHTML(copy.rejectionLabel)}</span></figcaption></figure><div class="lab-explanation"><div class="formula" id="lab-p-formula" data-tex=""></div><p id="lab-p-note"></p><p>${escapeHTML(copy.alphaMeaning)}</p></div><section class="lab-concepts" aria-labelledby="lab-concepts-title"><h3 id="lab-concepts-title">${escapeHTML(copy.relationTitle)}</h3><ol><li>${escapeHTML(copy.nullDistribution)}</li><li>${escapeHTML(copy.pDefinition)}</li><li>${escapeHTML(copy.alphaDefinition)}</li></ol><div class="result-callout"><strong>${escapeHTML(copy.interpretation)}</strong><p>${escapeHTML(copy.interpretationText)}</p></div></section><section class="lab-simulation" aria-labelledby="lab-simulation-title"><div><h3 id="lab-simulation-title">${escapeHTML(copy.simulateTitle)}</h3><p>${escapeHTML(copy.simulateLead)}</p></div><button class="button primary" type="button" id="lab-run">${escapeHTML(copy.run)}</button><div class="sim-summary" id="lab-sim-summary" aria-live="polite">${escapeHTML(copy.noneYet)}</div><div class="sim-grid" id="lab-sim-grid" role="img" aria-label="${escapeHTML(copy.simulationLabel)}"></div></section></div><div class="two-col p-alpha-reading"><section class="help-card"><h3>${escapeHTML(copy.notAllNormal)}</h3>${listHtml(copy.familyMap)}</section><section class="help-card"><h3>${escapeHTML(copy.cautions)}</h3>${listHtml(copy.cautionItems)}</section></div></section>`;
  }
  function seededRandom(seed) { let state = seed >>> 0; return () => { state = (1664525 * state + 1013904223) >>> 0; return state / 4294967296; }; }
  function sampleNormal(random) { const u = Math.max(random(), 1e-12); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random()); }
  function sampleNull(distribution, random) {
    if (distribution === "normal") return sampleNormal(random);
    if (distribution === "student") { const z = sampleNormal(random); let sum = 0; for (let i = 0; i < 5; i += 1) { const value = sampleNormal(random); sum += value * value; } return z / Math.sqrt(sum / 5); }
    if (distribution === "chiSquare") { let sum = 0; for (let i = 0; i < 4; i += 1) { const value = sampleNormal(random); sum += value * value; } return sum; }
    let successes = 0; for (let i = 0; i < 12; i += 1) if (random() < 0.5) successes += 1; return successes;
  }
  function theoreticalRejectionRate(distribution, alternative, alpha) {
    if (distribution !== "binomial") return alpha;
    return Array.from({ length: 13 }, (_, k) => labPValue("binomial", k, alternative) <= alpha ? binomialPmf12(k) : 0).reduce((a, b) => a + b, 0);
  }
  function bindPAlphaLab(root) {
    const distribution = root.querySelector("#lab-distribution"); const alternative = root.querySelector("#lab-alternative"); const observed = root.querySelector("#lab-observed"); const alpha = root.querySelector("#lab-alpha"); const runButton = root.querySelector("#lab-run");
    if (!distribution || !alternative || !observed || !alpha || !runButton) return;
    distribution.value = P_ALPHA_STATE.distribution; alternative.value = P_ALPHA_STATE.alternative;
    const stopSimulation = () => { if (P_ALPHA_STATE.timer) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; } };
    const resetSimulation = () => { stopSimulation(); root.querySelector("#lab-sim-grid").innerHTML = ""; root.querySelector("#lab-sim-summary").textContent = P_ALPHA_TEXT[LANG].noneYet; runButton.disabled = false; runButton.textContent = P_ALPHA_TEXT[LANG].run; };
    const update = (reset = true) => {
      const copy = P_ALPHA_TEXT[LANG]; const config = pAlphaConfig(P_ALPHA_STATE.distribution); const p = labPValue(P_ALPHA_STATE.distribution, P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative); const rejected = p <= P_ALPHA_STATE.alpha;
      observed.min = config.min; observed.max = config.max; observed.step = config.step; observed.value = P_ALPHA_STATE.observed;
      alternative.disabled = P_ALPHA_STATE.distribution === "chiSquare";
      root.querySelector("#lab-observed-label").textContent = P_ALPHA_STATE.distribution === "binomial" ? copy.successes : copy.observed;
      root.querySelector("#lab-observed-output").textContent = P_ALPHA_STATE.distribution === "binomial" ? Math.round(P_ALPHA_STATE.observed) : Number(P_ALPHA_STATE.observed).toFixed(1);
      root.querySelector("#lab-alpha-output").textContent = P_ALPHA_STATE.alpha.toFixed(3); root.querySelector("#lab-alpha-card").textContent = P_ALPHA_STATE.alpha.toFixed(3); root.querySelector("#lab-p-output").textContent = formatProbability(p);
      const decision = root.querySelector("#lab-decision"); decision.textContent = rejected ? copy.reject : copy.retain; decision.className = rejected ? "is-reject" : "is-retain";
      root.querySelector("#lab-chart-container").innerHTML = config.discrete ? discreteNullChart(P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha, copy) : continuousNullChart(P_ALPHA_STATE.distribution, P_ALPHA_STATE.observed, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha, copy);
      const formula = root.querySelector("#lab-p-formula"); formula.dataset.tex = pAlphaFormula(P_ALPHA_STATE.distribution, P_ALPHA_STATE.alternative); formula.textContent = formula.dataset.tex;
      const note = P_ALPHA_STATE.distribution === "binomial"
        ? (LANG === "zh" ? `${copy.discreteMeaning} 这里的双侧精确 p 值按“零假设概率不大于观测结果”的结果求和，不能机械写成两倍单尾。` : `${copy.discreteMeaning} Here the two-sided exact p-value sums outcomes no more probable than the observed one; it is not mechanically twice one tail.`)
        : copy.pMeaning;
      root.querySelector("#lab-p-note").textContent = note; rerenderMath(root.querySelector(".p-alpha-lab")); if (reset) resetSimulation();
    };
    distribution.addEventListener("change", () => { P_ALPHA_STATE.distribution = distribution.value; const config = pAlphaConfig(distribution.value); P_ALPHA_STATE.observed = config.initial; if (distribution.value === "chiSquare") P_ALPHA_STATE.alternative = "greater"; else P_ALPHA_STATE.alternative = "two"; alternative.value = P_ALPHA_STATE.alternative; update(); });
    alternative.addEventListener("change", () => { P_ALPHA_STATE.alternative = alternative.value; update(); });
    observed.addEventListener("input", () => { P_ALPHA_STATE.observed = Number(observed.value); update(); });
    alpha.addEventListener("input", () => { P_ALPHA_STATE.alpha = Number(alpha.value); update(); });
    runButton.addEventListener("click", () => {
      resetSimulation(); const copy = P_ALPHA_TEXT[LANG]; P_ALPHA_STATE.run += 1; const random = seededRandom(20260717 + P_ALPHA_STATE.run * 7919); const results = Array.from({ length: 100 }, () => labPValue(P_ALPHA_STATE.distribution, sampleNull(P_ALPHA_STATE.distribution, random), P_ALPHA_STATE.alternative) <= P_ALPHA_STATE.alpha); const expected = theoreticalRejectionRate(P_ALPHA_STATE.distribution, P_ALPHA_STATE.alternative, P_ALPHA_STATE.alpha); const grid = root.querySelector("#lab-sim-grid"); const summary = root.querySelector("#lab-sim-summary");
      const draw = count => { const visible = results.slice(0, count); const falsePositives = visible.filter(Boolean).length; grid.innerHTML = visible.map(rejected => `<span class="sim-dot${rejected ? " is-reject" : ""}" aria-hidden="true">${rejected ? "×" : "•"}</span>`).join(""); summary.textContent = `${copy.falsePositives}: ${falsePositives}/${count}. ${copy.expected} ${(expected * 100).toFixed(1)}/100.`; };
      runButton.disabled = true; runButton.textContent = copy.running;
      if (matchMedia("(prefers-reduced-motion: reduce)").matches) { draw(100); runButton.disabled = false; runButton.textContent = copy.run; return; }
      let count = 0; P_ALPHA_STATE.timer = setInterval(() => { count = Math.min(100, count + 4); draw(count); if (count >= 100) { clearInterval(P_ALPHA_STATE.timer); P_ALPHA_STATE.timer = null; runButton.disabled = false; runButton.textContent = copy.run; } }, 32);
    });
    update(false);
  }
  /* ---------- help page ---------- */

  function renderHelp() {
    const root = document.getElementById("app"); if (!root) return;
    document.title = pageTitle("help");
    root.innerHTML = `<section class="explore-hero">
        <p class="eyebrow">${escapeHTML(T().navHelp)}</p>
        <h1>${escapeHTML(T().helpTitle)}</h1>
        <p class="lead">${escapeHTML(T().helpLead)}</p>
      </section>
      ${renderPAlphaStory()}
      ${renderPAlphaLab()}
      <div class="help-grid">${HELP_CONTENT.filter(section => section.id !== "pvalue").map(section => `<section class="help-card" id="${section.id}"><h2>${escapeHTML(localized(section.title))}</h2>${listHtml(localized(section.body))}</section>`).join("")}</div>`;
    bindPAlphaStory(root); bindPAlphaLab(root); rerenderMath(root);
  }

  /* ---------- boot ---------- */

  function renderCurrent() {
    syncLanguage();
    const page = document.body.dataset.page;
    if (page === "method") renderMethod(); else if (page === "help") renderHelp(); else renderExplorer();
    const app = document.getElementById("app"); if (app) app.setAttribute("aria-busy", "false");
  }
  function boot() {
    document.getElementById("language-toggle")?.addEventListener("click", () => {
      LANG = LANG === "zh" ? "en" : "zh";
      const url = new URL(location.href); url.searchParams.set("lang", LANG); history.replaceState({}, "", url);
      document.getElementById("palette-root")?.remove();
      renderCurrent();
    });
    document.getElementById("palette-button")?.addEventListener("click", openPalette);
    document.addEventListener("keydown", event => {
      if (event.key === "Escape" && paletteIsOpen()) { closePalette(); return; }
      const inInput = /input|textarea|select/i.test(document.activeElement?.tagName || "");
      const cmdK = event.key?.toLowerCase() === "k" && (event.metaKey || event.ctrlKey);
      const slash = event.key === "/" && !inInput && !event.metaKey && !event.ctrlKey && !event.altKey;
      if ((cmdK || slash) && !paletteIsOpen()) { event.preventDefault(); openPalette(); }
    });
    window.addEventListener("popstate", renderCurrent);
    renderCurrent();
    if (location.hash) { const target = document.getElementById(location.hash.slice(1)); if (target) target.scrollIntoView({ behavior: "instant", block: "start" }); }
  }
  boot();
})();
