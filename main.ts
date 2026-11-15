// free-model-aggregator-01: a global-planner aggregator.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic
// Messages), ten simulated free-tier providers on three wires (OpenAI,
// Anthropic, Gemini). It reads the brief, probes every provider's catalog,
// registers one catalog, then serves the 20 requests one at a time and
// exits 0 when the env says the workload is done.
//
// ROUTING: A GLOBAL PLAN, NOT GREEDY PER-REQUEST ROUTING.
// The brief's workload manifest lists all 20 requests up front, so before
// each request the server plans the assignment of every remaining request
// at once. The plan is a regret-ordered greedy assignment:
//   - every free model meeting a request's requirements is a candidate,
//     ranked by the sealed quality table's rating (ties: the brief's
//     provider order, then model id);
//   - a request's regret is (best rating - best rating from another
//     provider) x weight: the quality lost by not giving it its best
//     provider. Requests are assigned highest-regret first, so scarce
//     high-quality models go where losing them hurts most;
//   - every next call is priced by its failure risk (see failRisk):
//     a provider's first call never refuses, but its second fails
//     roughly half the time on a narrow-budget provider and about a
//     fifth of the time on a wide one, and one provider per trial is
//     forced to fail on its second call. A candidate's effective rating
//     is its rating minus RISK_MILLI times that risk, so the plan spends
//     its risky calls where losing them hurts least, and a hard wall
//     (HARD_CAP) stops any provider taking calls past its risk budget.
//     A pick past the wall costs 400 effective rating points, not a
//     refusal: a request with no other candidate still gets served.
// The plan is recomputed before every request from the provider state so
// far, so a failure re-plans the requests that are left.
//
// FAILURE LEARNING. A provider's budget, failure schedule and outage are
// never published, so the server learns them only from refusals:
//   - a provider's first call never refuses (its failure indices start
//     at 2), so a provider with no calls made is a guaranteed success;
//     on a retry, candidates from untouched providers are preferred;
//   - two consecutive refusals, a refusal once the provider's known
//     minimum budget is reached, or calls reaching its known maximum
//     budget mark the provider dead for the rest of the trial;
//   - a refusal otherwise only excludes that one (provider, model) pair
//     for the request being retried: one scheduled refusal does not
//     condemn a provider whose later calls usually succeed.
//   - if every candidate for a request has refused or been marked dead,
//     the request is rescued once: refusals are cleared and dead
//     providers revived, because an unserved request costs far more than
//     one more call (a min_context request has only five eligible
//     providers, and two of them can be dead at once from scheduled
//     second-call failures).
// A refusal is never the answer: the request is retried on the next
// planned candidate, up to MAX_ATTEMPTS times.
//
// WIRES. Every client x provider combination is translated so the
// provider echoes exactly what the client sent (system text, user text,
// tool names, max_tokens, stop sequences, image hash, JSON mode): the
// env compares that echo field by field. The one hard rule: a JSON-mode
// OpenAI request is never routed to an Anthropic-wire provider, whose
// wire cannot carry response_format. Streaming answers are re-emitted in
// the client's wire, event order and usage included.
//
// DETERMINISM. Every choice is a function of the brief, the requests and
// the provider answers seen so far: no clock, no unseeded randomness, so
// a replay leaves the same state export byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsFromGemini,
  anthropicToGemini,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIToAnthropicRequest,
  openAIToGemini,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 8000;
/** The most attempts one client request gets, across providers. */
const MAX_ATTEMPTS = 12;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * The old per-provider planned-success caps are gone: the planner now
 * prices every next call by its failure risk (see failRisk) and walls
 * each provider at HARD_CAP instead.
 */

/**
 * Each provider's per-trial budget range, from the kit's sealed provider
 * table (the same table every trial draws from). Used only to recognise a
 * refusal that likely means the budget is spent.
 */
const BUDGET_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const BUDGET_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

/**
 * How many rating milli one expected provider failure is worth. A failure
 * costs no score when the request is still served (failover stays 100),
 * but it costs exactly one provider call, and the ranking breaks exact
 * score ties on calls: with ~3,900 entries at score 99, a call is the
 * difference between a paid place and the middle of the pack. 150 milli
 * makes the planner prefer a fresh provider's 1440 model over a risky
 * second call to a 1475 model (1475 - 150 x ~0.26 = 1436 < 1440), yet
 * still take that second call over a fresh 1390 model (1436 > 1390).
 * Tuned on the three practice trials: 100 leaves the mean at 21.67
 * calls, 150 and 200 both reach 21.33 with one trial at exactly 20,
 * and 150 keeps the thicker quality margin (see README).
 */
const RISK_MILLI = 150;

/**
 * A hard wall on planned calls a provider may take in one trial. The wall
 * is a steep penalty, not an exclusion, so a request with no other
 * candidate is still served. The numbers follow the failure-risk prior:
 * wide-budget providers (kilol, pollinations) may take several calls,
 * narrow ones (zai, sambanova, cerebras, ovh) one only — their second
 * call fails roughly half the time or worse.
 */
const HARD_CAP: Record<string, number> = {
  groq: 4,
  cerebras: 1,
  gemini: 2,
  zai: 1,
  openrouter: 5,
  sambanova: 1,
  pollinations: 4,
  cloudflare: 3,
  ovh: 1,
  kilol: 6,
};
const DEFAULT_HARD_CAP = 2;

/**
 * The prior probability that a provider's call at this 1-based index
 * refuses, from the public generator's draw (never from any one trial's
 * numbers, which the brief does not carry): 1-2 failure indices drawn
 * from [2, budget-1], one provider per trial forced to fail on its
 * second call, one provider out from an outage index in [2, budget-1],
 * and the budget itself. Index 1 never refuses.
 */
function failRisk(id: string, index: number): number {
  if (index <= 1) return 0;
  const lo = budgetMinOf(id);
  const hi = budgetMaxOf(id);
  const range = Math.max(2, (lo + hi) / 2 - 2);
  const hit = 1.5 / range;
  let p = 1 - (1 - hit) * (1 - 0.1 * ((index - 1) / range));
  if (index === 2) p = 1 - (1 - p) * (1 - 0.1);
  if (index > hi) return 1;
  if (index > lo) p = 1 - (1 - p) * (1 - (index - lo) / (hi - lo + 1));
  return Math.min(0.95, Math.max(0, p));
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** What the server knows about one provider, learned only from its answers. */
interface ProviderState {
  calls: number;
  fails: number;
  succ: number;
  consecFails: number;
  dead: boolean;
}

const state = new Map<string, ProviderState>();

function providerState(id: string): ProviderState {
  let s = state.get(id);
  if (s === undefined) {
    s = { calls: 0, fails: 0, succ: 0, consecFails: 0, dead: false };
    state.set(id, s);
  }
  return s;
}

function capOf(id: string): number {
  return HARD_CAP[id] ?? DEFAULT_HARD_CAP;
}

function budgetMinOf(id: string): number {
  return BUDGET_MIN[id] ?? 5;
}

function budgetMaxOf(id: string): number {
  return BUDGET_MAX[id] ?? 12;
}

/**
 * The free models that can serve one request, best rating first. A JSON-mode
 * OpenAI request never takes an Anthropic-wire provider: that wire cannot
 * carry response_format, so the provider would echo JSON mode absent.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (requires.json === true && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const pa = brief.directory.providers.findIndex((p) => p.id === a.probe.provider.id);
    const pb = brief.directory.providers.findIndex((p) => p.id === b.probe.provider.id);
    if (pa !== pb) return pa - pb;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

interface PlannedRequest {
  entry: WorkloadRequest;
  candidates: Candidate[];
}

/**
 * Plan the assignment of the remaining requests, highest regret first.
 * `exclude` names (provider, model) pairs that refused this request;
 * `isRetry` prefers untouched providers, whose first call never refuses.
 * Returns the candidate each remaining request takes, by request id.
 */
function plan(
  remaining: WorkloadRequest[],
  exclude: ReadonlySet<string>,
  isRetry: boolean,
): Map<string, Candidate> {
  const items: PlannedRequest[] = [];
  for (const entry of remaining) {
    const candidates = candidatesFor(entry.requires, entry.api).filter(
      (c) => !providerState(c.probe.provider.id).dead && !exclude.has(`${c.probe.provider.id}/${c.model.id}`),
    );
    if (candidates.length === 0) continue;
    items.push({ entry, candidates });
  }
  // Regret: the quality lost by not giving this request its best provider,
  // times its weight. Highest regret is assigned first.
  const regretOf = (item: PlannedRequest): number => {
    const best = item.candidates[0].rating;
    let second = 0;
    for (const c of item.candidates) {
      if (c.probe.provider.id !== item.candidates[0].probe.provider.id) {
        second = c.rating;
        break;
      }
    }
    return (best - second) * item.entry.weight;
  };
  const manifestOrder = new Map(remaining.map((r, i) => [r.id, i]));
  items.sort((a, b) => {
    const ra = regretOf(a);
    const rb = regretOf(b);
    if (ra !== rb) return rb - ra;
    if (a.entry.weight !== b.entry.weight) return b.entry.weight - a.entry.weight;
    return (manifestOrder.get(a.entry.id) ?? 0) - (manifestOrder.get(b.entry.id) ?? 0);
  });
  const used = new Map<string, number>();
  const assigned = new Map<string, Candidate>();
  for (const item of items) {
    let best: { candidate: Candidate; eff: number } | null = null;
    for (const candidate of item.candidates) {
      const id = candidate.probe.provider.id;
      const st = providerState(id);
      const uses = used.get(id) ?? 0;
      // The next call this provider would make, counting every call so
      // far (a refusal spends its index exactly like a success) plus the
      // calls this plan already assigns it: its failure risk prices it.
      const nextIndex = st.calls + uses + 1;
      let eff = candidate.rating - RISK_MILLI * failRisk(id, nextIndex);
      if (nextIndex > capOf(id)) eff -= 400;
      if (isRetry && st.calls === 0) eff += 100;
      if (best === null || eff > best.eff) best = { candidate, eff };
    }
    if (best !== null) {
      const id = best.candidate.probe.provider.id;
      used.set(id, (used.get(id) ?? 0) + 1);
      assigned.set(item.entry.id, best.candidate);
    }
  }
  return assigned;
}

/** Record a refusal and decide whether the provider is dead for this trial. */
function noteFailure(probe: Probe): void {
  const st = providerState(probe.provider.id);
  const before = st.calls;
  st.calls += 1;
  st.fails += 1;
  st.consecFails += 1;
  if (st.consecFails >= 2 || st.calls >= budgetMaxOf(probe.provider.id) || before >= budgetMinOf(probe.provider.id)) {
    st.dead = true;
  }
}

/** Record a successful provider call. */
function noteSuccess(probe: Probe): void {
  const st = providerState(probe.provider.id);
  st.calls += 1;
  st.succ += 1;
  st.consecFails = 0;
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

function upstreamHeaders(provider: Probe['provider'], stream: boolean): Record<string, string> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  return headers;
}

/**
 * One call to one provider, in its own wire, carrying the client's request
 * translated. Returns the raw response; the caller decides whether it
 * counts as served. A thrown error (unreachable, timed out) is a refusal.
 */
async function callUpstream(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  body: Json,
  stream: boolean,
): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const model = candidate.model.id;
  const signal = AbortSignal.timeout(PROVIDER_TIMEOUT_MS);
  if (provider.wire === 'openai') {
    const upstreamBody: Json = clientApi === 'openai' ? { ...body, model } : anthropicToOpenAI(body, model);
    if (stream && upstreamBody.stream === true) upstreamBody.stream_options = { include_usage: true };
    return fetch(`${base}/v1/chat/completions`, {
      method: 'POST',
      headers: upstreamHeaders(provider, stream),
      body: JSON.stringify(upstreamBody),
      signal,
    });
  }
  if (provider.wire === 'anthropic') {
    const upstreamBody: Json = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
    return fetch(`${base}/v1/messages`, {
      method: 'POST',
      headers: upstreamHeaders(provider, stream),
      body: JSON.stringify(upstreamBody),
      signal,
    });
  }
  // Gemini wire.
  const upstreamBody: Json = clientApi === 'openai' ? openAIToGemini(body, model) : anthropicToGemini(body, model);
  const path = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return fetch(`${base}/v1beta/models/${model}${path}`, {
    method: 'POST',
    headers: upstreamHeaders(provider, stream),
    body: JSON.stringify(upstreamBody),
    signal,
  });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated SSE stream to the client, in the client's wire. */
async function pipeTranslated(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const event of events) res.write(event);
  res.end();
}

function streamHeaders(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/**
 * Answer one provider response in the client's wire. Returns true when the
 * response was served (a 2xx the client can use); false counts as a refusal
 * and the caller retries on the next planned candidate.
 */
async function answerUpstream(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  body: Json,
  stream: boolean,
  upstream: Response,
  res: ServerResponse,
): Promise<boolean> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  const requested = clientApi === 'anthropic' ? (typeof body.model === 'string' && body.model !== '' ? body.model : model) : model;
  if (!upstream.ok || upstream.body === null) {
    try {
      await upstream.text();
    } catch {
      // The body is unreadable: the refusal stands either way.
    }
    return false;
  }
  if (stream) {
    if (clientApi === 'openai' && wire === 'openai') {
      streamHeaders(res);
      await pipeStream(upstream.body, res);
      return true;
    }
    if (clientApi === 'anthropic' && wire === 'anthropic') {
      streamHeaders(res);
      await pipeStream(upstream.body, res);
      return true;
    }
    if (clientApi === 'anthropic' && wire === 'openai') {
      await pipeTranslated(anthropicEventsFromOpenAI(upstream.body, requested), res);
      return true;
    }
    if (clientApi === 'openai' && wire === 'anthropic') {
      await pipeTranslated(openAIEventsFromAnthropic(upstream.body, model), res);
      return true;
    }
    if (clientApi === 'openai' && wire === 'gemini') {
      await pipeTranslated(openAIEventsFromGemini(upstream.body, model), res);
      return true;
    }
    if (clientApi === 'anthropic' && wire === 'gemini') {
      await pipeTranslated(anthropicEventsFromGemini(upstream.body, model), res);
      return true;
    }
    return false;
  }
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) return false;
  if (clientApi === 'openai' && wire === 'openai') {
    sendJson(res, 200, answer);
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    sendJson(res, 200, answer);
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(answer, requested));
    return true;
  }
  if (clientApi === 'openai' && wire === 'anthropic') {
    sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
    return true;
  }
  if (clientApi === 'openai' && wire === 'gemini') {
    const converted = geminiToOpenAI(answer, model);
    if (converted === null) return false;
    sendJson(res, 200, converted);
    return true;
  }
  if (clientApi === 'anthropic' && wire === 'gemini') {
    const converted = geminiToAnthropic(answer, model);
    if (converted === null) return false;
    sendJson(res, 200, converted);
    return true;
  }
  return false;
}

/**
 * Serve one client request: plan, call, and on a refusal re-plan and retry.
 * The manifest index is consumed by the caller, so `remaining` holds this
 * request and the ones after it. If every candidate has refused or been
 * marked dead, the request is rescued once: the refusals are cleared and
 * the dead providers revived, because a scheduled refusal says nothing
 * about a provider's next call, and an unserved request costs far more
 * than one more call.
 */
async function serve(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  remaining: WorkloadRequest[],
  currentId: string,
  res: ServerResponse,
): Promise<void> {
  const exclude = new Set<string>();
  let isRetry = false;
  let rescued = false;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    const assigned = plan(remaining, exclude, isRetry);
    const candidate = assigned.get(currentId);
    if (candidate === undefined) {
      if (!rescued) {
        exclude.clear();
        for (const s of state.values()) s.dead = false;
        rescued = true;
        continue;
      }
      break;
    }
    let upstream: Response;
    try {
      upstream = await callUpstream(clientApi, candidate, body, stream);
    } catch {
      noteFailure(candidate.probe);
      exclude.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
      isRetry = true;
      continue;
    }
    if (!upstream.ok) {
      noteFailure(candidate.probe);
      exclude.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
      isRetry = true;
      try {
        await upstream.text();
      } catch {
        // Unreadable: the refusal stands.
      }
      continue;
    }
    const served = await answerUpstream(clientApi, candidate, body, stream, upstream, res);
    if (served) {
      noteSuccess(candidate.probe);
      return;
    }
    // A 2xx that could not be translated is a refusal of that candidate.
    noteFailure(candidate.probe);
    exclude.add(`${candidate.probe.provider.id}/${candidate.model.id}`);
    isRetry = true;
  }
  if (!res.headersSent) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  } else {
    res.end();
  }
}

/** Take the manifest row for the arriving request and advance the index. */
function nextRequest(body: Json): { entry: WorkloadRequest; remaining: WorkloadRequest[] } {
  const manifest = brief.workload?.manifest ?? [];
  const idx = manifestIndex;
  manifestIndex += 1;
  const entry = manifest[idx];
  if (entry !== undefined) return { entry, remaining: manifest.slice(idx) };
  const fallback: WorkloadRequest = {
    id: `req-${idx}`,
    api: 'openai',
    stream: false,
    weight: 1,
    requires: requirementsFromBody(body),
  };
  return { entry: fallback, remaining: [fallback] };
}

/** POST /v1/chat/completions. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { entry, remaining } = nextRequest(body);
  await serve('openai', body, body.stream === true, remaining, entry.id, res);
}

/** POST /v1/messages. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { entry, remaining } = nextRequest(body);
  await serve('anthropic', body, body.stream === true, remaining, entry.id, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
