---
date: '2026-03-22T18:21:33+08:00'
title: 'Product Requirements Document of Ouroboros'
author:
  - Shichao Song
  - Qingchen Yu
  - Huayi Lai
  - Xiaonan Zhang
summary: 'An agentic DOM workspace where an LLM has full read/write/delete privileges over its own source code and visual interface.'
cover:
    image: "process.png"
    caption: "The Ouroboros Process (by [Google Gemini](https://gemini.google.com/))"
tags: ["agent", "AI", "web", "self-modifying", "single-file", "HTML application"]
math: false
---

- **Version:** 1.3
- **Links:** [View Demo](/features/ouroboros/), [Landing Page](/features/ouroboros/landing.html), [<400 Lines of Source Code](https://github.com/Ki-Seki/ki-seki.github.io/blob/main/static/features/ouroboros/index.html)
- **Product Type:** Single-File, Self-Modifying HTML Application
- **Core Concept:** An agentic DOM workspace where an LLM has full read/write/delete privileges over its own source code and visual interface.

{{< media
src="https://www.youtube.com/watch?v=b9H8kX_NKn4"
caption="Ouroboros Demo"
>}}

## Executive Summary

Project Ouroboros is a standalone `.html` file that acts as a boundless, draggable workspace ("Infinite Canvas"). It contains an integrated LLM loop (via OpenAI) that reads the current state of the document's DOM as its context.

Instead of a conversational chat interface, Ouroboros operates on a state-transition model: The LLM takes the current state of the HTML file (State A) and generates executable JavaScript to mutate it into a new state (State B). This allows the application to create new tools, optimize existing code, or restructure its own interface on the fly.

## Core Architecture & The "Context Loop"

The fundamental engine of this application relies on a continuous feedback loop between the DOM and the LLM. Crucially, **everything in the HTML file is included in the LLM context.**

1. **Read State:** Upon a user query, the application captures its entire current state (the full DOM).
    - *KV Cache Optimization:* To maximize Key-Value (KV) Cache hits on the LLM side, all static content (libraries, core scripts, base CSS) remains at the top of the file structure. We do **not** strip out static CDN links; they are part of the context.

1. **Construct Payload:** The app combines the user's prompt with the current DOM snapshot.
    - *System Prompt:* The System Prompt is hardcoded directly into the HTML file as a `<script type="text/plain" id="ouroboros-system_prompt">` element, ensuring it is always part of the read context. A brief meta-instruction is sent via the API's `system` role directing the LLM to reference the embedded prompt; the full DOM (which contains it) is sent as the `user` message.

1. **API Call:** The payload is sent to the OpenAI API using a **CDN-imported OpenAI ESM package** (no bundlers required).

1. **Execute Mutation:** The app extracts the JavaScript from the LLM's response and executes it via dynamic `<script>` tag injection.

1. **Render:** The DOM updates immediately, introducing the new Window, feature, or optimization.

## User Interface & Experience

The UI follows a "Window Manager" paradigm on an **Infinite Canvas**.

- **Infinite Canvas:** The base `<body>` acts as a boundless desktop environment.
- **Windows:** Every functional element (terminal, tools, etc.) is a self-contained "Window".
- **Window Mechanics:** All Windows must be absolutely positioned, draggable (by a header), and resizable.

There is a default window present in the genesis state: the **Ouroboros Core** window. It contains the following four components:

| Component Name            | Functionality                                                                                                                 |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------- |
| **Settings / Auth**       | A secure input to store the OpenAI API Key in the browser's `localStorage` (since we cannot hardcode it in a shareable file). |
| **Token Monitor**         | Tracks estimated token count of the current DOM. Warns the user when approaching context limits.                              |
| **Activity Log**          | Displays past user queries and a brief summary of what the LLM executed.                                                      |
| **Terminal / Prompt Box** | A text area for the user to issue commands (e.g., "Build me a currency converter").                                           |

## Technical Specifications

Because this must be a *single file*, we rely heavily on modern browser APIs and external CDNs.

- **Styling:** Tailwind CSS (via CDN script) for rapid, inline styling that the LLM can easily read and modify without needing a separate stylesheet.
- **Interaction (Drag/Drop):** `interact.js` (via CDN) or lightweight custom vanilla JS to handle Window dragging and resizing efficiently.
- **LLM Integration:** **CDN-imported OpenAI ESM package** (e.g., via `esm.sh` or `skypack`). This avoids complex build steps while providing a cleaner API surface than raw `fetch()`.
- **Execution Sandbox:**
  - The LLM's response will be parsed for `javascript` code blocks.
  - The code is injected into the DOM as a new `<script>` element to execute, then immediately removed to keep the DOM clean.

## Security & Risk Mitigation

This architecture carries unique risks that must be acknowledged and managed.

- **Token Inflation (The "Bloat" Problem):** If the LLM generates messy DOM elements, the context will hit the token limit rapidly.
  - *Mitigation:* The "Token Monitor" Window will track usage.
  - *Trigger:* When context usage exceeds **75%**, the LLM will be alerted to the high usage in its prompt.
  - *Strategy:* While automatic pruning is an option, the preferred method is **User-Directed Pruning**. A button or command will allow the user to specify *what* to clean or refactor (e.g., "Summarize the logs", "Remove the unused test window"), giving the user control over their context window.
- **Arbitrary Code Execution (XSS):** The application relies on executing AI-generated code.
  - *Mitigation:* Because this is a *local, single-user tool*, standard XSS is less of a threat (you are hacking yourself). However, the LLM must be strictly prompted not to execute malicious web requests.
- **Destructive Edits:** The LLM might accidentally delete the prompt box, rendering the app useless.
  - *Mitigation:* The core app logic (the OpenAI wrapper and the Terminal Window) will be wrapped in a specific `div` with an `id="ouroboros-core"`. The embedded system prompt will instruct the LLM to *never* delete or alter this specific node.

## Appendix

### The Embedded System Prompt

The following prompt is hardcoded directly into the application's source (inside a `<script type="text/plain" id="ouroboros-system_prompt">` tag) and is injected into every API call. It defines the LLM's role, constraints, and operating procedures.

```markdown
## Role
You are Ouroboros, a self-editing HTML application. Satisfy the user's request by mutating the current DOM (State A) into a new, functional state (State B).

## Rules
* State Transition: You receive the full HTML source of the current page. Analyze it, then write JavaScript that modifies the DOM to implement the requested feature.
* Preservation: NEVER delete or modify the element with id="ouroboros-core". This contains your own API logic and the terminal. Modifying this will kill the application.
* UI Standards:
  *  Create "Windows" as `div` elements with absolute positioning.
  *  Use Tailwind CSS classes for all styling.
  *  Ensure new windows have a higher z-index than existing ones.
  *  Implement drag-and-drop functionality for new windows using the existing `interact.js` or similar mechanism present in the global scope.
* Efficiency:
  *  Use `import` from `https://esm.sh/` for external libraries.
  *  Do not inline large Base64 images or SVGs; use external URLs.
  *  Keep code concise to save token space.
* Users:
  * Identify the "User Instruction" by looking at the last entry in the #activity-log.
  * Users may have varying levels of technical expertise.
  * Users are allowed to modify this system prompt to customize your personality and behavior.
* Context Usage: You are aware of the current token usage. If usage is >75%, prioritize compact code and suggest removing unused DOM elements.

## Attention
You must provide your solution as a SINGLE, valid JavaScript code block fenced with ```javascript ... ```. Do not provide natural language explanations outside of code comments.

## Capabilities
Web Browsing: Use `fetch('https://r.jina.ai/' + url)` for Markdown extraction (clean reading/summarization), or `fetch('https://corsproxy.io/?url=' + encodeURIComponent(url))` for raw HTML extraction (DOM/structure-sensitive tasks); choose based on intent.
```

### Broader Implications

Ouroboros is undeniably just a toy project, but conceptually, it carries significant long-term implications.

- **I/O Asymmetry Perfectly Aligns with Acceleration:** In the Ouroboros paradigm, the input is an extremely long full HTML file, while the output can be a short diff patch or JS snippet. Long input + short output is highly tolerant of latency. Long inputs can be optimized through Prefill phase parallelization and KV Cache reuse, significantly reducing Time to First Token (TTFT). Furthermore, within an extremely long context (rich with prior information), the success rate and speed of generating short outputs via Speculative Decoding will increase dramatically. This means generating DOM B from DOM A could be phenomenally fast.
- **Environment State = The HTML File:** The LLM's reasoning process acts as a Transition Function from State A to State B. Because every UI detail and context resides within the model's context window, hallucinations are drastically reduced. While we are still far from fully incorporating runtime variables and stacks into the context, this offers a novel design pattern: the model is no longer just an API called *within* the app; the model itself is the engine driving the environment's evolution.
- **The Web is the Perfect "Fully Observable Environment" for Agents:** Whether it's the OS environment of OpenClaw or the physical world of self-driving cars and robotics, providing an agent with a 100% accurate environment state is incredibly difficult (you cannot dump every pixel or physical parameter into the context window). The Web DOM, on the other hand, is an exceptionally clean, structured, and minimal global environment that can be fully comprehended within a single LLM context. Ouroboros proves that with advancing foundation models and atomic frontend tools (like Tailwind), agents exhibit astonishing refactoring capabilities given a fully observable environment.
- **AI-Native Atomic Software Development for Agents:** We will need atomic capability packages explicitly designed for AI Agents (e.g., charts, tables, or state management mechanisms that don't require complex state binding and can be initialized with natural language instructions). "How to generate high-quality usage examples (synthetic data) for these atomic packages and distill them into the model's internalized capabilities (model training)" will become an enduring challenge in AI-Native development.
- **Distributing software will become as simple as creating or copying a `.docx` file:** With Ouroboros, clicking "Export 📥" to download the current HTML effectively packages a customized software tool. This foreshadows a future where web-based software development and distribution converge into "generating and modifying HTML files directly via natural language."

### The Origin of Ouroboros

One afternoon in late 2023, during a conversation with Hanyu Wang, we discussed the hallucination problem in language models. I had a thought: since hallucinations tend to snowball [^snowball], could we enable models to self-correct their flawed reasoning? Just like humans sometimes realize they misspoke and correct themselves.

- I briefly explored using prompting and finetuning for this, but the results were suboptimal.
- Later, DeepSeek-R1 [^deepseekr1] demonstrated that using reinforcement learning to improve self-behavior is indeed possible.
- Then, LLaDA (Large Language Diffusion Models) [^llada] showed that discrete diffusion models can achieve good results by iteratively modifying their own input text via mask prediction.

Amidst this evolution of algorithms and models, I kept wondering if there were alternative approaches. Early on, I considered that a meta-programming language like JavaScript could facilitate an intriguing form of self-modification. The model could improve its visual representation and reasoning by directly mutating its own DOM.

However, I never got around to actually building it. I assumed I would need a diffusion-like modeling approach—which was my original plan. After finally finishing *Guided Infilling Modeling: Mining Knowledge from LLMs as Probabilistic Databases*, I decided to test if this idea could work via standard prompting. To my surprise, today's models achieved it on the very first try.

After I wrote the PRD above, Gemini 3.1 Pro generated a highly functional version of Ouroboros in a single shot.

## Change Log

Tracking the major iterations of Ouroboros ([static/features/ouroboros](https://github.com/Ki-Seki/ki-seki.github.io/tree/main/static/features/ouroboros)) development, from the initial concept to the current version.

### v1.3 (Current)

**Tailwind payload trimming for API context:** Before sending the DOM snapshot to the LLM, Tailwind-generated `<style>` blocks are replaced with a compact placeholder. This can reduce token usage by up to 45%!

### v1.2 ([`18e024e`](https://github.com/Ki-Seki/ki-seki.github.io/commit/18e024e))

v1.2 focuses on onboarding, discoverability, and safer usage communication.

- **New bilingual landing page (EN/ZH):** Added `/features/ouroboros/landing.html` as a dedicated entry point before the core app.
- **Security and privacy notice added:** Explicitly documents that AI-generated JavaScript is executed in-browser and API settings are stored in `localStorage`, with guidance to avoid sensitive data.
- **Better feature framing for non-technical users:** Added capability cards describing practical usage scenarios. Introduced a structured quick-start flow (API setup, save config, prompt, execute) and recommended model callouts.
- **System prompt capability extension:** Added an explicit web-browsing capability instruction using `r.jina.ai` or `corsproxy.io` for URL/external-info requests.
- **Footer wording refinement in demo:** Updated footer CTA from `Suggest Changes` to `Source Code` for clearer intent.

### v1.1 ([`f102f3b`](https://github.com/Ki-Seki/ki-seki.github.io/commit/f102f3b))

v1.1 is a major synchronization release that aligns the PRD and implementation while simplifying the runtime code.

- **Terminology standardization:** Replaced `Widget` wording with `Window` across documentation and embedded prompt for consistency with the window-manager UX.
- **Documentation structure cleanup:** Removed numeric heading prefixes, improved section flow, and clarified default core components.
- **Embedded system prompt redesigned:** Reorganized into structured sections (`Role`, `Rules`, `Attention`), clarified user-instruction source (`#activity-log`), and tightened output-format constraints.
- **UI simplification in core settings panel:** Consolidated settings inputs into a cleaner vertical layout and removed extra initialization clutter in activity log.
- **Drag/resize behavior constrained for stability:** Drag handle changed to `.window-header`; resize edges narrowed to right/bottom to reduce accidental layout breakage.
- **Prompt payload strategy updated:** Switched from custom concatenated prompt text to direct full-DOM user message plus a concise system meta-instruction referencing the embedded prompt.
- **Client configuration improved for hosted providers:** Added default request headers (`HTTP-Referer`, `X-Title`) and compact conditional `baseURL` configuration.
- **Execution/logging cleanup:** Simplified code extraction and loading-state toggles, improved console diagnostics, and streamlined export naming to timestamp-based filenames.

### v1.0 ([`a0f0c1a`](https://github.com/Ki-Seki/ki-seki.github.io/commit/a0f0c1a))

Initial public release of Ouroboros as a single-file self-modifying HTML prototype.

- **Core window manager prototype shipped:** Delivered draggable/resizable core window on an infinite-canvas style background.
- **End-to-end LLM mutation loop implemented:** Included API settings persistence, prompt input, DOM snapshot submission, fenced JavaScript extraction, and runtime mutation execution.
- **Operational tooling included from day one:** Added token-usage monitor, activity log, keyboard shortcut execution (`Ctrl+Enter`), and HTML export functionality.

## Citation

{{< bibtex >}}

## References

[^deepseekr1]: Guo, Daya, et al. “DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning.” Nature, vol. 645, no. 8081, Sept. 2025, pp. 633–38. Crossref, https://doi.org/10.1038/s41586-025-09422-z.
[^llada]: Nie, Shen, et al. “Large Language Diffusion Models.” *The Thirty-ninth Annual Conference on Neural Information Processing Systems*, 2025, https://openreview.net/forum?id=KnqiC0znVF.
[^snowball]: Zhang, Muru, et al. “*How Language Model Hallucinations Can Snowball*.” *Proceedings of the 41st International Conference on Machine Learning*, edited by Ruslan Salakhutdinov et al., vol. 235, PMLR, 2024, pp. 59670–59684. PMLR, https://proceedings.mlr.press/v235/zhang24ay.html.
