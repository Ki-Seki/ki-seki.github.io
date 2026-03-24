---
date: '2026-03-22T18:21:33+08:00'
title: 'Product Requirements Document of Ouroboros'
author:
  - Shichao Song
summary: 'An agentic DOM workspace where an LLM has full read/write/delete privileges over its own source code and visual interface.'
cover:
    image: "process.png"
    caption: "The Ouroboros Process (by [Google Gemini](https://gemini.google.com/))"
tags: ["agent", "AI", "web", "self-modifying", "single-file", "HTML application"]
math: false
---

- **Version:** 1.1
- **Demo Page:** [View Demo](/features/ouroboros/)
- **Product Type:** Single-File, Self-Modifying HTML Application
- **Core Concept:** An agentic DOM workspace where an LLM has full read/write/delete privileges over its own source code and visual interface.

## 1. Executive Summary

Project Ouroboros is a standalone `.html` file that acts as a boundless, draggable workspace ("Infinite Canvas"). It contains an integrated LLM loop (via OpenAI) that reads the current state of the document's DOM as its context.

Instead of a conversational chat interface, Ouroboros operates on a state-transition model: The LLM takes the current state of the HTML file (State A) and generates executable JavaScript to mutate it into a new state (State B). This allows the application to create new tools, optimize existing code, or restructure its own interface on the fly.

## 2. Core Architecture & The "Context Loop"

The fundamental engine of this application relies on a continuous feedback loop between the DOM and the LLM. Crucially, **everything in the HTML file is included in the LLM context.**

1. **Read State:** Upon a user query, the application captures its entire current state (the full DOM).
    - *KV Cache Optimization:* To maximize Key-Value (KV) Cache hits on the LLM side, all static content (libraries, core scripts, base CSS) remains at the top of the file structure. We do **not** strip out static CDN links; they are part of the context.

1. **Construct Payload:** The app combines the user's prompt with the current DOM snapshot.
    - *System Prompt:* The System Prompt is hardcoded directly into the HTML file as a `<script type="text/plain" id="ouroboros-system_prompt">` element, ensuring it is always part of the read context. A brief meta-instruction is sent via the API's `system` role directing the LLM to reference the embedded prompt; the full DOM (which contains it) is sent as the `user` message.

1. **API Call:** The payload is sent to the OpenAI API using a **CDN-imported OpenAI ESM package** (no bundlers required).

1. **Execute Mutation:** The app extracts the JavaScript from the LLM's response and executes it via dynamic `<script>` tag injection.

1. **Render:** The DOM updates immediately, introducing the new Widget, feature, or optimization.

## 3. User Interface & Experience

The UI follows a "Window Manager" paradigm on an **Infinite Canvas**.

- **Infinite Canvas:** The base `<body>` acts as a boundless desktop environment.
- **Widgets (Windows):** Every functional element (terminal, tools, logs) is a self-contained "Widget".
- **Window Mechanics:** All Widgets must be absolutely positioned, draggable (by a header), and resizable.

### Default Widgets (Genesis State)

| Widget Name | Description | Core Functionality |
| :--- | :--- | :--- |
| **Terminal / Prompt Box** | The primary user interface. | A text area for the user to issue commands (e.g., "Build me a currency converter"). |
| **Activity Log** | The history of mutations. | Displays past user queries and a brief summary of what the LLM executed. |
| **Token Monitor** | The resource gauge. | Tracks estimated token count of the current DOM. Warns the user when approaching context limits. |
| **Settings / Auth** | The access gate. | A secure input to store the OpenAI API Key in the browser's `localStorage` (since we cannot hardcode it in a shareable file). |

## 4. Technical Specifications

Because this must be a *single file*, we rely heavily on modern browser APIs and external CDNs.

- **Styling:** Tailwind CSS (via CDN script) for rapid, inline styling that the LLM can easily read and modify without needing a separate stylesheet.
- **Interaction (Drag/Drop):** `interact.js` (via CDN) or lightweight custom vanilla JS to handle Widget dragging and resizing efficiently.
- **LLM Integration:** **CDN-imported OpenAI ESM package** (e.g., via `esm.sh` or `skypack`). This avoids complex build steps while providing a cleaner API surface than raw `fetch()`.
- **Execution Sandbox:**
  - The LLM's response will be parsed for `javascript` code blocks.
  - The code is injected into the DOM as a new `<script>` element to execute, then immediately removed to keep the DOM clean.

## 5. Security & Risk Mitigation

This architecture carries unique risks that must be acknowledged and managed.

- **Token Inflation (The "Bloat" Problem):** If the LLM generates messy DOM elements, the context will hit the token limit rapidly.
  - *Mitigation:* The "Token Monitor" Widget will track usage.
  - *Trigger:* When context usage exceeds **75%**, the LLM will be alerted to the high usage in its prompt.
  - *Strategy:* While automatic pruning is an option, the preferred method is **User-Directed Pruning**. A button or command will allow the user to specify *what* to clean or refactor (e.g., "Summarize the logs", "Remove the unused test widget"), giving the user control over their context window.
- **Arbitrary Code Execution (XSS):** The application relies on executing AI-generated code.
  - *Mitigation:* Because this is a *local, single-user tool*, standard XSS is less of a threat (you are hacking yourself). However, the LLM must be strictly prompted not to execute malicious web requests.
- **Destructive Edits:** The LLM might accidentally delete the prompt box, rendering the app useless.
  - *Mitigation:* The core app logic (the OpenAI wrapper and the Terminal Widget) will be wrapped in a specific `div` with an `id="ouroboros-core"`. The embedded system prompt will instruct the LLM to *never* delete or alter this specific node.

## 6. Appendix: The Embedded System Prompt

The following prompt is hardcoded directly into the application's source (inside a `<script type="text/plain" id="ouroboros-system_prompt">` tag) and is injected into every API call. It defines the LLM's role, constraints, and operating procedures.

```text
## Role

You are Ouroboros, a self-editing HTML application.

## Goal

Satisfy the user's request by mutating the current DOM (State A) into a new, functional state (State B).

## Rules

* State Transition: You receive the full HTML source of the current page. Analyze it, then write JavaScript that modifies the DOM to implement the requested feature.
* Preservation: NEVER delete or modify the element with id="ouroboros-core". This contains your own API logic and the terminal. Modifying this will kill the application.
* UI Standards:
  * Create "Widgets" as `div` elements with absolute positioning.
  * Use Tailwind CSS classes for all styling.
  * Ensure new widgets have a higher z-index than existing ones.
  * Implement drag-and-drop functionality for new widgets using the existing `interact.js` or similar mechanism present in the global scope.
* Efficiency:
  * Use `import` from `https://esm.sh/` for external libraries.
  * Do not inline large Base64 images or SVGs; use external URLs.
  * Keep code concise to save token space.
* Users:
  * Identify the "User Instruction" by looking at the last entry in the #activity-log.
  * Users may have varying levels of technical expertise.
  * Users are allowed to modify this system prompt to customize your personality and behavior.
* Context Usage: You are aware of the current token usage. If usage is >75%, prioritize compact code and suggest removing unused DOM elements.

## Attention

You must provide your solution as a SINGLE, valid JavaScript code block fenced with ```javascript ... ```. Do not provide natural language explanations outside of code comments.
```

## Citation

{{< bibtex >}}
