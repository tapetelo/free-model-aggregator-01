# Best free AI model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API (JSON and streaming) over ten simulated free-tier model
providers, routing each request to the best free model that can serve it.

## How it routes

Global planner, not greedy per-request routing. Before each request it
re-plans the assignment of *all* remaining requests at once: candidates are
free models meeting the request's requirements, ranked by the sealed quality
table; requests are assigned highest-regret-first (regret = best rating −
best rating from another provider, × weight), so scarce high-quality models
go where losing them hurts most. Per-provider planned-success caps are soft
(a −50 effective-rating penalty). Failures are learned from: a provider is
marked dead after 2 consecutive refusals, a refusal past its known minimum
budget, or calls reaching its maximum budget; retries prefer untouched
providers. A rescue pass revives dead providers once when no candidate
remains, so a min_context request is never stranded. All choices derive from
the brief, the requests and provider answers — no wall clock, no unseeded
randomness — so a replay leaves the same state export.

## Run

```sh
node main.ts            # serve (registers with $MAKEFUN_ENV_URL)
node main.ts --list     # print the catalog document
```

Node 22, standard library only. `lib/translate.ts` holds all six
client×provider wire translations (OpenAI/Anthropic/Gemini).

## Local scores (practice trials, grade-local --driver process)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls |
|---|---|---|---|---|---|---|---|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 23 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 21 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 23 |

Replay matched byte-for-byte on all three trials.

Built for the make.fun bounty `free-model-aggregator-01` as @tapetelo.
