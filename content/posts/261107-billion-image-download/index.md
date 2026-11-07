---
date: '2026-11-07T22:47:37+08:00'
title: 'Efficient Large-Scale Crawling with an AI Agent: 3 Days, $175, 3 Billion Original Images'
author:
  - Elliot Shichao Song
summary: 'Downloading the originals behind 2.98 billion image URLs in OmniCorpus-CW, I left almost all operations to an AI agent: patrols, rate tuning, creating and deleting machines, incident handling and billing. I only made the decisions. In 79 hours it saved 99.6% of the images, 744 TB, for about $175 in machines.'
cover:
  image: "assets/billion-image-download-cover.png"
  caption: "An AI-coordinated image hub (generated with [GPT Image 2.6 Sunburst](https://openai.com/index/introducing-images-2-6/))"
tags:
  - AI
  - large-scale
  - web-crawling
math: false
---

## Introduction

I still remember when I first [learned web crawling](https://ki-seki.github.io/posts/210817-scrapy/). Back then I had to learn many details: CSS selectors, XPath, request parsing, URL patterns. I never expected that five years later all of them would matter so little.

This post covers how I downloaded the original images in [OmniCorpus-CW](https://openxlab.org.cn/datasets/Li-Qingyun/OmniCorpus-CW) [^li_omnicorpus], a large-scale dataset of Chinese web images. It is an interleaved image-text corpus: each web page is stored as text paragraphs plus links to its images, mostly from WeChat, Xiaohongshu and Zhihu. Such images are valuable for training text-to-image models.

I know the basics of crawling but had never crawled at scale. With AI, I got it done cheaply, quickly and reliably: 95 Hetzner cloud servers, $175 in machine costs, and 2,973,430,108 images saved (99.62%) in 79 hours 25 minutes, 744 TB in all, averaging about 10,000 images and 2.6 GB per second.

Over those three days the AI agent did almost all the operations: patrols, rate tuning, creating and deleting machines, incident handling and billing. I only made the decisions that cost money, added resources or carried risk.

{{< media
src="assets/architecture.svg"
caption="Figure 1: The human gives the AI agent goals and permissions; the agent operates the Hetzner download fleet, which moves images from website CDNs into S3."
>}}

## Letting an AI Agent Run the Operations

I used Claude Code (CC) [^anthropic_claude_code] with Opus 5.5 to run the whole crawling system. CC lived on a dev machine and ran nonstop. The agent isn't cheap: at Opus 5.5 API list prices, its tokens are worth about $500, nearly 3 times the machine cost. I suspect a model this strong isn't needed; today's DeepSeek 4 might do, with a bit more human attention.

Hetzner's API tokens make it easy for a model to start and stop machines, so from the start I gave CC that permission and let it probe and tune download rates itself.

CC patrolled hourly by day and every 30 minutes at night, and event monitors woke it the moment something broke. Over three days its work log passed 2,600 lines. Some examples:

- **WeChat rate control:** After machines were added, all of them slowed down at once and in proportion. It concluded that WeChat caps the fleet's total rate and wrote a rate controller that kept the fleet right at the cap. A control machine in the US showed that another region didn't help; it was deleted 8 minutes later.
- **Xiaohongshu speed-up:** On one machine it raised the per-IP rate step by step. No challenge pages appeared even at 80 requests per second, so it settled on 70 for margin and rolled it out to the fleet in batches.
- **10,000+ small sites:** It re-split their roughly 15 million rows into tasks by site and tuned each site's rate separately. Along the way, on my decisions, it lifted the caps for a few large image hosts and dropped two sites that mostly returned 403.
- **Memory:** Finding machines low on memory, it restarted them one by one as planned and added swap files where swap was short.
- **Retries:** After the run it counted the failures and re-downloaded only the transient ones, recovering 98.3% of them for Xiaohongshu and 99.3% for WeChat.
- **Dev machine reclaimed:** CC's dev machine was reclaimed, stopping CC for 1 hour 42 minutes. The credentials then expired and every machine exited cleanly. A CC session on my own computer took over: it pushed credentials from another dev machine and restarted every lane, then handed back to the original CC once its machine returned.
- **Wrap-up:** It deleted all 95 machines, confirmed no resources were left in the accounts, read all of the roughly 120,000 completion markers and produced the final statistics.

## Prompt

Below is a fairly complete example prompt for handing a large-scale image download to an AI agent.

```text
Download about <N> image URLs from <dataset> (<location>) as originals, and store them with basic metadata in <S3 prefix>. You run everything from probing to wrap-up; I only make decisions.

Goals
- Store raw bytes; no transcoding.
- One result row per URL, with its dataset row number for joining back.
- Success: storage path and basic metadata (width, height, format, bytes, content hash). Failure: status code and reason.
- Resumable; reruns never re-download.

Permissions
- Hetzner API token at <path>, S3 credentials at <path>.
- You may create and delete machines and tune rates. Ask me before exceeding the budget, requesting more quota or deleting stored data.

Step 1: Probe to set budget and deadline
1. At most <K> machines at a time while probing.
2. Count URLs by domain; split into lanes by site.
3. Run each lane on one machine first: measure throughput, find the bottleneck (CPU, bandwidth or the site's limit), compare throughput per dollar across machine types.
4. Ramp each large site's rate step by step. Then add a few machines: if per-machine speed drops in proportion, the site caps the total, not each IP.
5. Give me an estimate: each site's limit and limit type, machine types and counts, total time, total cost. Scale up only after I set budget and deadline.

Step 2: Run
- Rate tuning never stops. Each lane adjusts automatically by error rate: small steps up after a stable stretch; on throttling, cut by a factor and record the ceiling, then step more carefully near it. Limits shift with time of day, often higher late at night, so keep probing.
- Slow down only on errors. Slow is not throttled: large images and evening network peaks are slow too.
- Three levels of refusal: throttling, slow down; persistent 403 from a site, stop that site; challenge pages needing human verification piling up, trip the whole lane and wait for me.
- Long-tail small sites can't each be probed: start low with few connections and auto-tune per site; pause unreachable sites and retry later.
- Rates and start/stop come from one central control file, no process restarts; each lane has its own stop switch.
- Validate responses: HTML, challenge pages and placeholders are failures, even with status 200.
- Separate retryable from final failures; retry transient ones in a final round.
- Update the estimate whenever a site's limit becomes clear.

Step 3: Wrap up
- Done only when every URL has a final result and transient failures have had one retry round.
- Input row count must equal result row count.
- Delete all machines; confirm nothing is left.
- Compute cost from machine hours, reconcile with the console, produce final statistics.

Throughout: patrols and monitoring
- Patrol hourly by day, every 30 minutes at night; event monitors wake you the moment something breaks.
- Each patrol: per-lane progress and speed, error mix, machine CPU/memory/disk, spend, ETA. Fix what you can and log it; contact me via <contact> only for decisions.
- Slow-machine rules must allow for the long tail: near the end, slow is normal; don't delete machines that are finishing.
- Build a live monitoring page showing the above.
- Persist state and logs so a new session can take over if your machine dies.

Throughout: live spot checks
- From the trial run on, keep sampling each lane at random, read images back from storage, and show them on a spot-check page of the monitor for the data's users (<who>) to review anytime.
- Users decide usability: placeholders, "image deleted" images, thumbnails, watermarks and crops can all make data unusable. If they reject a lane, stop it first, find the cause, then decide.
- Self-check too: find each site's most repeated hashes (usually placeholders); compare samples with what a browser gets for the same URL to confirm originals.
```

Interestingly, a safety-aligned AI may refuse outright to try things like solving "verify you are human" challenges; Opus 5.5 firmly refuses such requests. An unaligned AI might try to bypass them. So this post leaves that topic aside for now.

## Using Hetzner from China

Plenty has been written about signing up for Hetzner, so just one point: you don't need to make anything up. Fill in your details truthfully and you'll pass. A passport makes identity verification easy.

New accounts start with few vCPUs, which caps how many machines you can run at once. Open a ticket through the official channel; in my case, even with a brand-new account, the quota went from 20 to 100 vCPUs in 3 days.

Thanks to Hetzner's low prices, this large-scale download cost very little.

## Conclusion

An interesting large-scale image download, led by an AI agent.

## Citation

{{< bibtex >}}

## References

[^anthropic_claude_code]: **Anthropic.** *Claude Code*. 2025, https://claude.com/claude-code.

[^li_omnicorpus]: **Li, Qingyun, et al.** *OmniCorpus: A Unified Multimodal Corpus of 10 Billion-Level Images Interleaved with Text*. 2024. *arXiv*, https://arxiv.org/abs/2406.08418.
