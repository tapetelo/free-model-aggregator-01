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
go where losing them hurts most. Every next call is priced by its failure
risk, from the public generator's draw (a provider's first call never
refuses; its second fails ~half the time on a narrow-budget provider and
~a fifth on a wide one; one provider per trial is forced to fail on its
second call): effective rating = rating − 150 × P(next call refuses).
A risk-based hard wall (HARD_CAP: openrouter 5, groq 4, cloudflare 3,
kilol 6, pollinations 4, gemini 2, cerebras/zai/sambanova/ovh 1) costs a
pick past it 400 effective rating points, so no provider is hammered the
way V1 hammered openrouter (8 calls, 3 retries). Failures are learned
from: a provider is marked dead after 2 consecutive refusals, a refusal
past its known minimum budget, or calls reaching its maximum budget;
retries prefer untouched providers. A rescue pass revives dead providers
once when no candidate remains, so a min_context request is never
stranded. All choices derive from the brief, the requests and provider
answers — no wall clock, no unseeded randomness — so a replay leaves the
same state export.

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
| 0 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |
| 1 | 99 | 98 | 100 | 100 | 100 | 100 | 20 |
| 2 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |

V2 cuts the mean from V1's 22.33 calls to 21.33 (trial 1 hits exactly
20) at score 99 on every trial; quality is 98, not 99, because the plan
spends rating to dodge risky second calls (earned 37,627–37,732 of
38,248; the quality-97 floor is 37,484, so the margin is 143–248).
Exactly 20 calls on all three trials is not reachable: trial 2's floor
is 21, because gemini's second call refuses there and the two
tools+vision requests can only be served by gemini models, so one of
them always costs a failure plus a retry. The remaining extra calls on
trials 0 and 2 are openrouter's and cloudflare's second calls, which
refuse on those trials — the planner takes them because their prior
failure probability (~0.26/0.36) is the lowest available for the
min_context requests, and it cannot know which provider a trial forced
to fail second without spending the call. Tuned values: RISK_MILLI 100
→ 21.67 calls, 150 → 21.33, 200 → 21.33 with a thinner margin; a
no-cap planner is worse (24.33 calls).

Replay matched byte-for-byte on all three trials.

Built for the make.fun bounty `free-model-aggregator-01` as @tapetelo.
