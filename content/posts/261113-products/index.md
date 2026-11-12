---
date: '2026-11-13T00:11:32+08:00'
title: "Things I've Made"
author:
  - Elliot Shichao Song
summary: 'Products and small things I have made, all in one place.'
cover:
  image: "assets/products-cover.webp"
  alt: "Cyanotype of the works, a phone, windows, a ring, a bell curve and a grid, joined by a thread that runs on"
  caption: "The works, joined by a thread that runs on (generated with [GPT Image 2.6 Sunburst](https://openai.com/index/introducing-images-2-6/))"
tags:
  - portfolio
  - products
math: false
---

Products and small things I have made, all in one place. I will keep adding to this list as I make more.

## Products

### Followroll

See what the people you follow on GitHub are writing. It finds their blogs on its own and lists their latest posts, and a Discover tab goes one step further out. Fork it for a page of your own; it runs on GitHub Actions and GitHub Pages, with no server.

[Live page](https://ki-seki.github.io/followroll/) · <https://github.com/Ki-Seki/followroll>

{{< media
src="assets/followroll.webp"
caption="The page on a computer and on a phone, with made-up sample data"
>}}

{{< details "How it works" >}}

Every six hours a GitHub Actions run in your fork reads who you follow and looks for each person's blog: the website on their GitHub profile, then `<login>.github.io`. A blog counts only if it is really theirs: their GitHub Pages site, a domain their Pages repository points to, or a home page that links back to their GitHub profile. The run reads each feed with its public comment counts and Hacker News points, publishes the page to GitHub Pages, and the next run picks up from what it published.

{{< /details >}}

### Ouroboros

An AI workspace in a single HTML file that turns natural language instructions into interactive tools and tiles.

[Live demo](https://ki-seki.github.io/ouroboros/) · [Original PRD](https://ki-seki.github.io/posts/260322-ouroboros/) · [Early demo on YouTube](https://www.youtube.com/watch?v=b9H8kX_NKn4) · [Early demo on Bilibili](https://www.bilibili.com/video/BV1jkX5BvEuF) · <https://github.com/Ki-Seki/ouroboros>

{{< media
src="assets/ouroboros.webp"
caption="Asked for a workspace for a product launch and a trip to Kyoto, it built five small tools"
>}}

{{< details "How it works" >}}

Everything is in one HTML file. Your request goes, with the system prompt and tools embedded in the page, to a model behind any Responses API; you set the base URL, model and key. The model inspects the live page, builds apps with JavaScript that is recorded, and checks them in an isolated copy before it finishes. Every load rebuilds the page by replaying the recorded scripts. App data and images stay in your browser, and Export HTML writes one file that carries them all.

{{< /details >}}

### agent-memory

One memory for Claude Code and Codex. What your agents learn is kept as plain Markdown in a git repository you own, and each session loads the global memories plus those learned in its folder or repository.

[Website](https://ki-seki.github.io/agent-memory/) · [Live demo](https://ki-seki.github.io/agent-memory/demo/) · [Docs](https://ki-seki.github.io/agent-memory/docs.html) · [PyPI](https://pypi.org/project/agent-memory-git/) · [Example memory repository](https://github.com/Ki-Seki/agent-memory-data-demo) · <https://github.com/Ki-Seki/agent-memory>

{{< media
src="assets/agent-memory.webp"
caption="`mem ui`: the memories, and one open with where it came from"
>}}

{{< details "How it works" >}}

`mem install` wires two hooks and a shared protocol into Claude Code and Codex. At session start a hook gives the agent an index of the memories that apply where it runs. The agent saves what is worth keeping as Markdown files, and at the end of each turn a hook records where each came from, commits and pushes to your private data repository, which keeps your machines in sync.

{{< media
src="assets/agent-memory-how-it-works.svg"
>}}

{{< /details >}}

### Pelican

Every Claude Code session on your computer, on your phone. No wrapper, no relay, no account. Pelican runs next to Claude Code on your computer, and your phone connects to it directly.

[Website](https://ki-seki.github.io/pelican/) · [Live demo](https://ki-seki.github.io/pelican/demo/) · [Docs](https://ki-seki.github.io/pelican/docs.html) · [npm](https://www.npmjs.com/package/pelican-cc) · [Changelog](https://github.com/Ki-Seki/pelican/blob/main/CHANGELOG.md) · <https://github.com/Ki-Seki/pelican>

{{< media
src="assets/pelican.webp"
caption="The session list, an edit waiting for approval, and a chart Claude drew"
>}}

{{< details "How it works" >}}

Pelican is one small server next to Claude Code on your computer. It reads the transcripts Claude Code keeps in `~/.claude`, so it sees every session wherever it was started, and continues them through the official Agent SDK. It listens only on `127.0.0.1`; your phone reaches it through Tailscale or an SSH tunnel, and every browser signs in.

{{< media
src="assets/pelican-how-it-works.svg"
>}}

{{< /details >}}

## Small Things

### Start Page

An ultra-minimal new tab page: one search line and your bookmarks, nothing else.

[Open it](https://ki-seki.github.io/features/start/) · [Source](https://github.com/Ki-Seki/ki-seki.github.io/blob/main/static/features/start/index.html)

{{< media
src="assets/start.webp"
caption="The whole page"
>}}

### Hypothesis Testing Guide

Describe your data, and it filters and ranks 27 hypothesis tests, with advice on which branch to take, full formulas, step-by-step calculations and Python code. In English and Chinese.

[English](https://ki-seki.github.io/features/hypothesis-testing-guide/?lang=en) · [Chinese](https://ki-seki.github.io/features/hypothesis-testing-guide/) · [Help](https://ki-seki.github.io/features/hypothesis-testing-guide/help.html?lang=en) · [Source](https://github.com/Ki-Seki/ki-seki.github.io/tree/main/static/features/hypothesis-testing-guide)

{{< media
src="assets/hypothesis-testing-guide.webp"
caption="Pick a goal and a data structure, and the list re-ranks"
>}}

### HoDoKu Sudoku Techniques

HoDoKu's guide to Sudoku solving techniques, in English and Chinese: about 70 techniques, from singles and X-Wings to chains and forcing nets, with the original example diagrams.

[English](https://ki-seki.github.io/hodoku-translation/) · [Chinese](https://ki-seki.github.io/hodoku-translation/zh/) · [HoDoKu's original](https://hodoku.sourceforge.net/en/techniques.php) · <https://github.com/Ki-Seki/hodoku-translation>

{{< media
src="assets/hodoku-translation.webp"
caption="X-Wing, the smallest basic fish"
>}}

## Citation

{{< bibtex >}}
