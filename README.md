# Agentic Computer-Vision Video Editing

> A new product feature at an AI video-editing startup — an agentic AI that automates multi-camera switching from raw footage to a final cut. I owned the feature end to end as DRI, from spec to production to post-launch optimisation.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![OpenAI GPT-5](https://img.shields.io/badge/OpenAI-GPT--5-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-structured_output-1C3C3C?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-fallback_VLM-8E75B2?logo=googlegemini&logoColor=white)
![DINOv2](https://img.shields.io/badge/DINOv2-CLS_embeddings-FF6F00)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)

I worked at an **AI video-editing startup** as a server / AI engineer, where I **owned multi-camera switching and personalization-based editing end to end** — from interviewing customers, writing the spec, designing the algorithms and the LLM-driven agent on top, shipping, optimising cost and latency, and stabilising the feature in production.

This repo is a technical write-up of that work — architecture, engineering decisions, and outcomes at a conceptual level, with no production source code or proprietary tuning.

---

## Role & Impact

I planned, built, and shipped the multi-camera switching agent as DRI — a new product feature that **analyses raw multi-camera footage, finds the visually impactful moments, and cuts to the right camera at the right time**.

- **End-to-end ownership.** Customer interviews → spec → labelled data → algorithm → agent → release → post-launch optimisation → infra stability work. I was directly responsible from start to ship and beyond.
- **Customer impact.** Editors reported the feature cut their multi-cam switching time by **more than 90% versus the manual workflow.**
- **Business impact.** The feature became a step-change in the product's overall quality and **directly contributed to closing new B2B accounts.**

---

## How I Worked: From Customer Interviews to Production

I didn't start this from a spec — I started it from **conversations with the people who would actually use it.** Before writing a line of the switching algorithm, I ran repeated interview rounds with professional video editors, and I kept going back to them throughout the project.

**Locking down the essential spec with customers.** Many interview rounds let me extract the essential rules a multi-cam edit has to obey — the non-negotiable spec the system would later be measured against. One clear ordering came out of those conversations: **clean cuts come first (jump-cut avoidance), engagement (reactions, wide shots) comes second.** Everything in the algorithm was built around that priority.

**Turning requirements into the feature set.** Asking editors how they fix specific failure modes surfaced the concrete pipeline features the system had to support — reaction inserts, wide-shot overrides, jump-cut filtering. The places where they spent the most manual time also confirmed exactly where the automation should land.

**POC before production.** Every idea went through a **Streamlit prototype first**: I'd wire up a small CV model or an LLM scorer, run it on real footage, and review the output *with* a customer before committing. Only once a POC held up did I build the production module.

**Iterating on feedback.** The system didn't arrive fully formed. Customers reviewed drafts at every milestone and pointed at concrete gaps; each round became a specific change — a new requirement, a re-tuned rate, a new control. Deploy decisions were gated on reviewing a few stylistically different videos with customers first — not on my own judgment alone.

**Leading and collaborating across the team.** Multi-cam switching couldn't ship in isolation — it depends on upstream pieces like speaker-to-segment alignment, transcription, and per-frame embeddings. I scoped what I needed, **wrote the specs, and handed that work to teammates**, then integrated their modules into the pipeline. Driving a feature end-to-end while coordinating other engineers is where I really *felt* how collaboration works — not as a hand-off, but as a shared contract everyone builds against.

---

## Tech Stack

- **Vision-language scoring:** OpenAI GPT-5 family via LangChain structured output; Google Gemini as fallback
- **Agentic orchestration:** LangChain structured-output sub-agents + LangGraph for parallel stages
- **Visual embeddings:** DINOv2 `CLS` token, cosine similarity over a shared visual–text space
- **Vision / video:** OpenCV, PySceneDetect, MediaPipe, yt-dlp
- **Transcription:** ElevenLabs diarized STT, OpenAI Whisper
- **Runtime:** Python 3.12, fully async, Pydantic-typed boundaries

---

## The Problem

Editing multi-camera, long-form video (interviews, podcasts, reviews) is slow and repetitive. A human editor constantly answers two questions: *who do I cut to right now?* and *does this cut feel motivated?* Doing that for hours of footage across several camera tracks is exactly the kind of judgment-heavy, repetitive work worth automating.

The goal: take raw multi-camera footage plus its transcript and produce a **camera-switching timeline** that feels intentional — and, given a **reference video**, reproduce *that* editor's pacing and style.

---

## Problem Solving

The parts of this project I'm proudest of are the engineering problems underneath it — each one started as "the obvious approach doesn't work," and the fix is where the real thinking happened.

### 1. Defining the spec and the eval set from customer interviews

Before writing the agent, I ran enough customer interviews to lock down a tight feature spec — what the agent **must do**, what it **must never do**, and what good output looks like for each kind of input.

That spec turned into a **labelled dataset**: concrete inputs paired with the expected outputs editors would actually accept. Building the agent against those answer keys (rather than a vague prompt) gave me a clear target during development. The same dataset doubled as the **evaluation set**, so every later change — a new requirement, a swapped model, a re-tuned threshold — was measured rather than guessed at.

This was the single most important decision in the project. Everything later — the algorithm-first refactor, the cost-and-latency pass, the infra hardening — only became *measurable* because the answer key existed from day one.

### 2. Going agentic without losing the floor — algorithm-first, agent on top

Video editing is a domain with rich, layered context — visual content, dialogue, pacing, narrative arc. My first instinct was to let an LLM decide every cut from raw input. It didn't work: across roughly **150 trials**, the end-to-end LLM approach landed at an **average F1 ≈ 0.7, with a minimum of ≈ 0.22 and a wide standard deviation**. The lower bound wasn't reliable enough to ship — the model couldn't internalise enough of what *good editing* looks like from prompting alone, and the worst-case output was unacceptable.

The fix was to invert the design into **three layers: deterministic algorithm → agentic tuning → per-scene detail.** The agent stops trying to be the editor and starts being the editor's *assistant for tuning a system that already knows what good looks like.*

```text
user feedback
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│ Layer 2  Agentic tuning                                  │
│   ┌──────────────┐   ┌────────────────────┐              │
│   │  Mode agent  │   │ Intent classifier  │  (async)     │
│   └──────────────┘   └────────────────────┘              │
│           │                    │                         │
│           ▼                    ▼                         │
│       global mode      ┌───────────────┐ ┌─────────────┐ │
│                        │ Numeric agent │ │ Detail bucket│ │
│                        │ (knob config) │ │ (per-scene) │ │
│                        └───────────────┘ └─────────────┘ │
└────────────┬──────────────────────────┬──────────────────┘
             │                          │
             ▼                          ▼
┌─────────────────────────────┐    ┌───────────────────────┐
│ Layer 1  Deterministic core │    │ Layer 3  Detail agent │
│ (algorithm, scaled by knobs)│    │ (local overrides)     │
└────────────┬────────────────┘    └────────┬──────────────┘
             │ baseline timeline            │
             └──────────────┬───────────────┘
                            ▼
                  final cam-switching
```

**Layer 1 — a deterministic algorithm does the cutting.** Every essential rule from the interview-derived spec — jump-cut avoidance, minimum / maximum visible shot duration, reaction insert, wide-shot pacing, chapter alignment — lives in this layer as plain typed code. It's fast, reproducible, and unit-testable. The LLM never makes a per-cut decision; everything agentic is built *around* this layer, not in place of it. This is what guarantees the floor on output quality.

**Layer 2 — agents tune the algorithm, agentically.** Instead of letting a model cut, I let it **turn the algorithm's knobs**. The user's free-text prompt flows through three small structured-output sub-agents that prepare the tuning:

- A **mode agent** picks the top-level behaviour the user is asking for (e.g. snappier / calmer / more reactions) so the rest of the routing knows the baseline target.
- An **intent classifier** splits the request into *global-dial* work vs. *specific-scene* work — a single sentence often contains both ("make it snappier overall, but hold on the left guest at the end"), and each half has to reach the right layer.
- A **numeric agent** reads the global-dial portion and emits a small, typed config of multipliers that scale the algorithm's behaviour — so *"more reactions, fewer wides"* becomes a few bounded, debuggable numbers rather than a giant prompt steering every cut.

Each sub-agent returns a **typed Pydantic config** via LangChain structured output, so the model's output is impossible to mis-parse and each agent is independently testable.

**Layer 3 — a detail agent handles what the algorithm can't express.** Some instructions are inherently local — *"go wide during the chorus", "hold on the left guest at the end."* The algorithm has no knob for those, so they get routed into a **detail agent** that locates the exact moment in the timeline and overrides the baseline directly. Agentic precision is applied only where it's actually needed.

**How it runs.** The three layers run as one short **async** routine — the mode agent and classifier fire together (`asyncio.gather`), the numeric agent produces the knob config, the algorithm runs, and the detail agent overlays last. Every hop crosses a **typed boundary**, so a change to any one layer can be made in isolation behind contract tests without breaking the others.

This gave every user a **personalised agent** that still respected the non-negotiable rules editors had baked into the spec. Re-running the same 150-trial harness, the rebuilt system landed at **F1 ≥ 0.98 in 148 of 150 trials** — the floor was finally guaranteed.

The deeper lesson: when the domain is hard to learn from prompts alone, the question stops being *how agentic can I make this* and becomes *where should the line between deterministic code and LLM judgment fall.* Layering agentic decisions on a tested algorithmic core gave me the flexibility of an agent with the floor of plain code.

### 3. Cutting cost and latency — embedding search → LLM verification

After launch, two clear signals came back from production: editors found the workflow too slow, and the company found the per-job LLM cost too high. I proposed a follow-up optimisation initiative and **co-designed the new pipeline with another engineer** on the team.

The original approach swept frames at a fixed cadence and ran a vision-LLM call on every one to find visually impactful moments. We replaced that with **embedding-driven search**: every frame is embedded once into a shared visual–text space (**DINOv2** `CLS` token), candidate moments are surfaced by **cosine similarity** against the user's query embeddings, and the vision LLM is invoked only to **verify** the shortlist. We also pushed selected calls down to cheaper CV primitives where the task was structured enough — **HSV color-histogram correlation** for set / background change, **YOLO** segmentation to mask people out so similarity is driven by the scene rather than who's on screen, **PySceneDetect** for shot boundaries — keeping the LLM only for the genuinely semantic judgments nothing else could make.

Aggressive cost cuts can quietly trade away output quality, so the right compromise point isn't a purely technical decision. I picked it in conversation with **marketing, finance, engineering leadership, and customers** at the same time — each stakeholder had a different stake in the speed / cost / quality trade-off, and the right point was the one all four could live with.

End result of the optimisation pass:

- **Wall-clock per workflow:** ~15 min → ~3 min.
- **LLM cost per workflow:** about **1/20** of the original.
- **Quality:** F1 ≥ 0.98 in **136 of 150 trials** — a measured trade-off from the pre-optimisation 148 / 150, deliberately kept inside the bar editors had set.

### 4. A clustering metric that lied — centroid vs. top-5-mean

Inside the pipeline I needed to decide whether a candidate camera track is a genuine second camera (worth cutting to) or one-off B-roll. The natural design is to embed each cut with **DINOv2** (`CLS` token, with **YOLO** segmentation masking people out so similarity is driven by the set/background rather than who's on screen), cluster the cuts by cosine similarity, and summarize each cluster with a single representative score that gets thresholded. The obvious choice for that summary — the **centroid** (cosine similarity of the cluster's mean embedding to the target) — turned out to be wrong, and working out *why* drove the final design.

**The bug.** A cluster whose individual frames were clearly *more* similar to one another could score a *lower* centroid similarity than a looser cluster. Straight from the real data:

| Cluster | Individual member similarities | Centroid similarity |
|---------|--------------------------------|:-------------------:|
| Cluster 0 | **0.57 – 0.67** (tight, high) | 0.435 ⬇ |
| Cluster 2 | 0.14 – 0.47 (loose, low) | 0.512 ⬆ |

Cluster 0's members are individually the stronger matches, yet its centroid ranks *below* Cluster 2's. The ordering inverts — and a real multicam angle gets misclassified as B-roll.

**Root cause.** The centroid is the *mean* of the (L2-normalized) member embeddings, and the cosine of that mean rewards **directional coherence inside the cluster**, not similarity to the target. When members point in slightly different directions they partially cancel under averaging, shrinking the mean vector's magnitude and pulling its cosine down — even though every member is individually similar. So the centroid silently conflates *"how tight is this cluster"* with *"how relevant is this cluster,"* and unfairly penalizes a diffuse-but-strong cluster.

**The fix.** Score each cluster by the **mean of its top-5 most-similar members** (a "5-max-mean") instead. This judges a cluster on its *strongest evidence* rather than its average direction, so cancellation can never hide a genuine match — and unlike a single max, one lucky outlier can't carry it. I swept the summarizer choice on the labelled reference set and top-5-mean came out on top:

| Summarizer | Micro-F1 |
|------------|----------|
| Centroid | 0.9987 |
| Single max | 0.9984 |
| Top-2 mean | 0.9987 |
| Top-3 mean | 0.9989 |
| **Top-5 mean** | **0.9993** |
| Mean | 0.9987 |
| Two-criterion | 0.9978 |
| Adaptive | 0.9978 |

Per-video performance with the chosen summarizer on the held-out reference set:

| Reference video | Multicam-classification F1 | Cut-detection F1 |
|-----------------|:--------------------------:|:----------------:|
| A | 1.000 | 1.000 |
| B | 1.000 | 0.963 |
| C | 0.940 | 0.976 |
| D | 0.9995 | 0.958 |
| E | 0.9964 | 0.9952 |

The general lesson from this one: an "obvious" summary statistic can encode a different question than you think it does. The fix wasn't a more sophisticated model — it was choosing a statistic whose definition actually matched the decision being made.

### 5. Memory OOM under heavy image load — streaming refactor with infra

After the feature went live, batched-image workflows started triggering **out-of-memory** failures on the server. I traced the root cause with the infra team: the original logic pre-downloaded every image for a workflow into local `/tmp` and then read the frames it needed by timestamp — fine for a few hundred frames, ruinous at production scale.

We refactored the server logic to **stream frames from S3 on demand**, fetching only the timestamps the pipeline actually requested. To unblock infra to ship the rewrite quickly without worrying about silent regressions, I wrote **byte-for-byte equivalence tests** asserting that the new path produced exactly the same output as the old one on the same inputs. The infra team could swap the implementation with confidence; memory pressure dropped to a fraction of the original, and the deploy landed without a quality incident.

---

## Personalization — Matching a Reference Video's Style

A good default edit isn't enough — creators want their footage cut **in the style of a video they already like.** The personalization layer takes a reference video (e.g. a YouTube URL) and produces an edit that reproduces its pacing and structural decisions.

"Style" splits cleanly into two halves, and the system treats each one differently — same philosophy as Problem #2 above: numbers where a number suffices, an agent only where genuine judgment is needed.

**The measurable half — pacing, cut frequency, shot-duration distribution, reaction density, wide-shot rate.** A CV pipeline analyses the reference video end to end: **PySceneDetect** finds every shot boundary so pace is measured from real cuts rather than guessed; **MediaPipe** pose/person detection separates a solo close-up from a group wide shot cheaply; **DINOv2** `CLS` embeddings group shots from the same camera/framing; and a vision LLM (GPT-5 / Gemini) handles the calls that need actual semantic understanding (*"is this a reaction? an establishing shot?"*). The output is a structured profile of *how* the reference was edited, and the switching algorithm's parameters are then tuned to reproduce that profile.

The trick is that I tune the parameters *jointly*, not independently. Every parameter is expressed as a curve between two style anchors (one calm reference, one fast), and a single density value derived from the reference moves the whole parameter set together. So matching a reference collapses to estimating **one number**, and the parameters can never drift into an unnatural combination — a problem that would absolutely happen if each knob were free to be tuned in isolation.

**The structural half — intro / outro pacing, when to pull wide for context, how a review-style video opens vs. a podcast.** These aren't single numbers; they're patterns over time. An agent recognises and replicates them from the reference, used for intro / outro matching and review-style videos. The agent handles only the structural calls; everything quantifiable stays in the numerical pipeline.

---

## How to Evaluate an Agentic System

Evaluation was the part I underestimated most at the start. Multi-cam switching is a taste problem — there's no canonical oracle, no public benchmark, and two skilled human editors will produce two defensible versions of the same scene. The interview-derived spec + labelled dataset (§ 1 above) gave me an answer key per input, but on top of that I had to define **what "passing" actually meant** at three levels:

- **The deterministic core** — the algorithm-first layer is fully testable: given the same input, it has to produce the same output. Regressions are caught by exact-match contract tests on every commit.
- **The algorithm vs. the human edit** — for each labelled reference video, the algorithm's output is compared against the human edit on cut positions, category distributions, and structural invariants (minimum / maximum visible cut duration, no gaps, every camera id valid). Tolerances are set per layer of the output — strict where determinism allows it, banded where natural variation is healthy.
- **The agentic layer** — the agent's job is to translate free-text user feedback into pipeline instructions. I built a **labelled golden set** of feedback prompts paired with expected agent decisions, and iteration on prompts and structure is gated on **non-decreasing pass rate plus zero regressions on previously-passing prompts**. New real-world failures get added to the golden set, so the eval keeps getting harder in exactly the directions the system is weakest.

All three layers run as a CI gate — a PR that drops the agent's pass rate, breaks an algorithm regression case, or changes an upstream contract this module depends on can't land green. Evaluation is a *gate*, not a *dashboard*.

The result is that I could keep changing things — a new requirement, a swapped model, an engine replacement, the cost-and-latency pass above — without the nagging fear of a silent regression. Behaviour was locked at the *contract* level, not the *implementation* level.

---

## Testing — the seam between deterministic and agentic

Most of my thinking about this system came together while writing the tests. Early on, the editing logic and the LLM / video calls were tangled together, and every change felt risky — when something broke, I couldn't tell whether I'd broken the algorithm or just hit a flaky model call.

What I settled on was drawing a hard line between a **deterministic core** (the editing logic — pure functions over typed data) and a **nondeterministic boundary** (LLM calls, video decode, embedding search), and letting the unit / integration split land exactly on that line.

- **Unit tests** lock the deterministic core in milliseconds. Each piece is a small typed function with no hidden state — exactly the shape that's both easy to test and easy to evolve.
- **Integration tests** replay recorded fixtures (real transcripts, real production-shaped inputs) through the real modules — no live LLM calls, so CI stays fast and deterministic, but the seams between stages are actually exercised.
- **Signature-contract tests** asserted byte-for-byte identical I/O at every stage, which is what made it safe to swap engines underneath (the LLM → embedding swap in § 3, the local-tmp → S3-streaming swap in § 4) without a silent regression.

The thing I took from this: once contracts are pinned by tests, teammates can own different modules and trust the seams. The tests doubled as living documentation of what each stage promised, which saved a lot of "wait, what does this return again?" back-and-forth — and it meant the agentic layer on top was free to evolve aggressively, because the floor it stood on couldn't quietly shift.
