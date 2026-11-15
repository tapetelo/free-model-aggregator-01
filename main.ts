// free-model-aggregator-01: one server planning routes across free tiers.
//
// A single process answers both client protocols (OpenAI Chat Completions
// and Anthropic Messages) by forwarding to ten simulated free providers
// spread over three upstream protocols (OpenAI, Anthropic, Gemini). At
// startup it reads the brief, asks each provider what it lists, registers
// a single catalog with the environment, then works through the twenty
// requests one by one and exits 0 once the environment reports the
// workload finished.
//
// ROUTING: PLAN EVERYTHING LEFT, NOT ONE REQUEST AT A TIME.
// The manifest names all twenty requests before any of them arrives, so
// ahead of each request the server assigns all the requests still
// waiting in one go. The assignment is greedy in regret order:
//   - each free model that meets a request's needs is an option, ordered
//     by the sealed quality table's rating (ties break on the brief's
//     provider order, then on model id);
//   - a request's regret is (its best rating - the best rating from a
//     different provider) times its weight: the quality given up by not
//     handing it its best provider. The largest regrets are settled
//     first, so scarce strong models land where losing them stings most;
//   - each next call carries a failure price (see refusalRisk): a
//     provider's opening call never refuses, but its second one fails
//     about half the time on a tight-budget provider and about a fifth
//     of the time on a roomy one, and one provider in every trial is
//     made to fail its second call. An option's working rating is its
//     rating less RISK_PER_FAIL times that risk, so the plan burns its
//     chancy calls where losing them costs least, and a steep wall
//     (CALL_WALL) keeps any provider from taking calls past its risk
//     allowance. A pick beyond the wall loses 400 working rating points
//     instead of being refused outright: a request with no other option
//     is still answered.
// The assignment is rebuilt before every request from what the providers
// have done so far, so a refusal reshuffles the requests still waiting.
//
// LEARNING FROM REFUSALS. No provider ever publishes its budget, its
// failure slots or its outage, so the server infers them from refusals:
//   - a provider's opening call never refuses (its failure slots start
//     at 2), so a provider with no calls yet is a certain success; when
//     retrying, options from untouched providers are favoured;
//   - two refusals in a row, a refusal once the provider's known
//     minimum budget is used up, or calls reaching its known maximum
//     budget retire the provider for the rest of the trial;
//   - any other refusal only rules out that one (provider, model) pair
//     for the request being retried: a single scheduled refusal does not
//     write off a provider whose later calls mostly succeed.
//   - when every option for a request has refused or been retired, the
//     request is rescued a single time: refusals are wiped and retired
//     providers brought back, because a dropped request costs far more
//     than one extra call (a min_context request has only five eligible
//     providers, and two of them can be retired together by scheduled
//     second-call failures).
// A refusal is never the reply: the request is tried again on the next
// planned option, up to MAX_TRIES times.
//
// WIRES. Every client-by-provider pairing is translated so the provider
// echoes back precisely what the client sent (system text, user text,
// tool names, max_tokens, stop sequences, image hash, JSON mode): the
// environment checks that echo field by field. The single strict rule:
// a JSON-mode OpenAI request never goes to an Anthropic-wire provider,
// whose protocol cannot carry response_format. Streaming replies are
// sent on again in the client's protocol, event order and usage intact.
//
// DETERMINISM. Each decision depends only on the brief, the requests and
// the provider replies observed so far: no clock, no unseeded chance, so
// a replay produces the same state export byte for byte.

import {
  createServer,
  type IncomingMessage,
  type Server,
  type ServerResponse,
} from 'node:http';
import {
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type WorkloadRequest,
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
} from './lib/providers.ts';
import {
  type ServedModel,
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
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

/** The protocol version stamped into the register body. */
const PROTOCOL_V = 1;
/** The gap between GET /work polls, in milliseconds. */
const POLL_MS = 100;
/** The longest one provider's catalog probe may run, in milliseconds. */
const PROBE_TIMEOUT = 5000;
/** The longest one provider call may run, in milliseconds. */
const PROVIDER_TIMEOUT = 8000;
/** The most tries one client request gets, across providers. */
const MAX_TRIES = 12;
/** The most bytes of one client request body this server reads. */
const MAX_BODY = 32 * 1024 * 1024;

/**
 * The per-provider planned-success caps from before are gone: the planner
 * now prices each next call by its failure risk (see refusalRisk) and
 * walls each provider at CALL_WALL instead.
 */

/**
 * Each provider's per-trial budget span, taken from the kit's sealed
 * provider table (the same table every trial draws from). Only used to
 * spot a refusal that likely means the budget is spent.
 */
const BUDGET_LO: Record<string, number> = {
  groq: 9, cerebras: 5,
  gemini: 6, zai: 4,
  openrouter: 10, sambanova: 4,
  pollinations: 10, cloudflare: 7,
  ovh: 5, kilol: 12,
};
const BUDGET_HI: Record<string, number> = {
  groq: 12, cerebras: 6,
  gemini: 8, zai: 5,
  openrouter: 12, sambanova: 5,
  pollinations: 14, cloudflare: 8,
  ovh: 6, kilol: 16,
};

/**
 * What one expected provider failure is worth in rating milli. A failure
 * costs no score when the request is still answered (failover stays 100),
 * but it costs exactly one provider call, and the ranking settles exact
 * score ties on calls: with ~3,900 entries at score 99, a call is the
 * gap between a paid place and the middle of the pack. 150 milli makes
 * the planner take a fresh provider's 1440 model over a chancy second
 * call to a 1475 model (1475 - 150 x ~0.26 = 1436 < 1440), yet still
 * take that second call over a fresh 1390 model (1436 > 1390). Tuned on
 * the three practice trials: 100 leaves the mean at 21.67 calls, 150 and
 * 200 both reach 21.33 with one trial at exactly 20, and 150 keeps the
 * thicker quality margin (see README).
 */
const RISK_PER_FAIL = 150;

/**
 * A steep wall on planned calls one provider may take in a trial. The wall
 * is a heavy penalty, not an exclusion, so a request with no other
 * option is still answered. The figures follow the failure-risk prior:
 * roomy-budget providers (kilol, pollinations) may take several calls,
 * tight ones (zai, sambanova, cerebras, ovh) one only — their second
 * call fails about half the time or worse.
 */
const CALL_WALL: Record<string, number> = {
  groq: 4, cerebras: 1,
  gemini: 2, zai: 1,
  openrouter: 5, sambanova: 1,
  pollinations: 4, cloudflare: 3,
  ovh: 1, kilol: 6,
};
const DEFAULT_WALL = 2;

/**
 * The prior chance that a provider's call at this 1-based slot refuses,
 * taken from the public generator's draw (never from one trial's own
 * figures, which the brief never carries): one or two failure slots
 * drawn from [2, budget-1], one provider per trial made to fail on its
 * second call, one provider down from an outage slot in [2, budget-1],
 * plus the budget itself. Slot 1 never refuses.
 */
function refusalRisk(id: string, index: number): number {
  if (index <= 1) return 0;
  const floor = minBudgetFor(id);
  const ceiling = maxBudgetFor(id);
  const span = Math.max(2, (floor + ceiling) / 2 - 2);
  const perCall = 1.5 / span;
  let risk = 1 - (1 - perCall) * (1 - 0.1 * ((index - 1) / span));
  if (index === 2) risk = 1 - (1 - risk) * (1 - 0.1);
  if (index > ceiling) return 1;
  if (index > floor) risk = 1 - (1 - risk) * (1 - (index - floor) / (ceiling - floor + 1));
  return Math.min(0.95, Math.max(0, risk));
}

const ENV_ORIGIN = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const BRIEF_PATH = process.env.MAKEFUN_PROMPT_FILE ?? '';

const briefDoc = readBrief(BRIEF_PATH);
let probeList: Probe[] = [];
let catalogDoc: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest runs in arrival order, one request at a time. */
let manifestCursor = 0;

function note(detail: string): void {
  process.stderr.write(`aggregator: ${detail}\n`);
}

/** Ask every provider for its listing and assemble the single catalog (the `--list` document and the register body). */
async function loadCatalog(): Promise<void> {
  if (briefDoc.error !== null) note(briefDoc.error);
  probeList = await Promise.all(briefDoc.directory.providers.map((provider) => probeProvider(ENV_ORIGIN, provider, PROBE_TIMEOUT)));
  for (const result of probeList) if (!result.ok && result.error !== null) note(result.error);
  catalogDoc = buildCatalog(ENV_ORIGIN, probeList, briefDoc.ratings);
}

const byModelId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function listableModels(): ServedModel[] {
  const listed: ServedModel[] = [];
  for (const result of probeList) {
    if (!result.ok) continue;
    for (const entry of [...result.models].sort(byModelId)) {
      const ratingRow = ratingOfModel(briefDoc.ratings, result.provider.id, entry.id);
      if (!modelFree(entry, ratingRow)) continue;
      listed.push({ id: entry.id, provider: result.provider.id, display_name: entry.display_name });
    }
  }
  return listed;
}

// ---------------------------------------------------------------------------
// The planner
// ---------------------------------------------------------------------------

interface Choice {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** What the server knows about one provider, learned only from its answers. */
interface ProviderLedger {
  made: number;
  refused: number;
  ok: number;
  run: number;
  done: boolean;
}

const ledgers = new Map<string, ProviderLedger>();

function ledgerFor(id: string): ProviderLedger {
  let ledger = ledgers.get(id);
  if (ledger === undefined) {
    ledger = { made: 0, refused: 0, ok: 0, run: 0, done: false };
    ledgers.set(id, ledger);
  }
  return ledger;
}

function wallFor(id: string): number {
  return CALL_WALL[id] ?? DEFAULT_WALL;
}

function minBudgetFor(id: string): number {
  return BUDGET_LO[id] ?? 5;
}

function maxBudgetFor(id: string): number {
  return BUDGET_HI[id] ?? 12;
}

/**
 * The free models that can answer one request, best rating first. A
 * JSON-mode OpenAI request never takes an Anthropic-wire provider: that
 * protocol cannot carry response_format, so the provider would echo
 * JSON mode absent.
 */
function eligibleChoices(needs: Requirement, api: 'openai' | 'anthropic'): Choice[] {
  const choices: Choice[] = [];
  for (const result of probeList) {
    if (!result.ok) continue;
    if (needs.json === true && api === 'openai' && result.provider.wire === 'anthropic') continue;
    for (const entry of result.models) {
      const ratingRow = ratingOfModel(briefDoc.ratings, result.provider.id, entry.id);
      if (!modelFree(entry, ratingRow)) continue;
      if (!meetsRequirements(needs, capabilitiesOf(entry, ratingRow))) continue;
      choices.push({ probe: result, model: entry, rating: ratingRow?.rating_milli ?? 0 });
    }
  }
  choices.sort((x, y) => {
    if (x.rating !== y.rating) return y.rating - x.rating;
    const ix = briefDoc.directory.providers.findIndex((p) => p.id === x.probe.provider.id);
    const iy = briefDoc.directory.providers.findIndex((p) => p.id === y.probe.provider.id);
    if (ix !== iy) return ix - iy;
    return x.model.id < y.model.id ? -1 : x.model.id > y.model.id ? 1 : 0;
  });
  return choices;
}

interface PlannedJob {
  request: WorkloadRequest;
  options: Choice[];
}

/**
 * Assign the requests still waiting, largest regret first. `refused`
 * names (provider, model) pairs that turned this request down; `retrying`
 * favours untouched providers, whose opening call never refuses.
 * Returns the option each remaining request takes, keyed by request id.
 */
function planAssignments(
  pending: WorkloadRequest[],
  refused: ReadonlySet<string>,
  retrying: boolean,
): Map<string, Choice> {
  const jobs: PlannedJob[] = [];
  for (const request of pending) {
    const options = eligibleChoices(request.requires, request.api).filter(
      (c) => !ledgerFor(c.probe.provider.id).done && !refused.has(`${c.probe.provider.id}/${c.model.id}`),
    );
    if (options.length === 0) continue;
    jobs.push({ request, options });
  }
  // Regret: the quality given up by not handing this request its best
  // provider, times its weight. Largest regret is settled first.
  const regretFor = (job: PlannedJob): number => {
    const top = job.options[0].rating;
    let next = 0;
    for (const choice of job.options) {
      if (choice.probe.provider.id !== job.options[0].probe.provider.id) {
        next = choice.rating;
        break;
      }
    }
    return (top - next) * job.request.weight;
  };
  const orderOf = new Map(pending.map((r, i) => [r.id, i]));
  jobs.sort((a, b) => {
    const qa = regretFor(a);
    const qb = regretFor(b);
    if (qa !== qb) return qb - qa;
    if (a.request.weight !== b.request.weight) return b.request.weight - a.request.weight;
    return (orderOf.get(a.request.id) ?? 0) - (orderOf.get(b.request.id) ?? 0);
  });
  const taken = new Map<string, number>();
  const picks = new Map<string, Choice>();
  for (const job of jobs) {
    let top: { choice: Choice; score: number } | null = null;
    for (const choice of job.options) {
      const providerId = choice.probe.provider.id;
      const ledger = ledgerFor(providerId);
      const already = taken.get(providerId) ?? 0;
      // The next call this provider would make, counting every call so
      // far (a refusal spends its slot exactly like a success) plus the
      // calls this plan already hands it: its failure risk prices it.
      const callIndex = ledger.made + already + 1;
      let score = choice.rating - RISK_PER_FAIL * refusalRisk(providerId, callIndex);
      if (callIndex > wallFor(providerId)) score -= 400;
      if (retrying && ledger.made === 0) score += 100;
      if (top === null || score > top.score) top = { choice, score };
    }
    if (top !== null) {
      const providerId = top.choice.probe.provider.id;
      taken.set(providerId, (taken.get(providerId) ?? 0) + 1);
      picks.set(job.request.id, top.choice);
    }
  }
  return picks;
}

/** Record a refusal and decide whether the provider is retired for this trial. */
function recordRefusal(result: Probe): void {
  const ledger = ledgerFor(result.provider.id);
  const prior = ledger.made;
  ledger.made += 1;
  ledger.refused += 1;
  ledger.run += 1;
  if (ledger.run >= 2 || ledger.made >= maxBudgetFor(result.provider.id) || prior >= minBudgetFor(result.provider.id)) {
    ledger.done = true;
  }
}

/** Record a successful provider call. */
function recordSuccess(result: Probe): void {
  const ledger = ledgerFor(result.provider.id);
  ledger.made += 1;
  ledger.ok += 1;
  ledger.run = 0;
}

// ---------------------------------------------------------------------------
// Upstream calls
// ---------------------------------------------------------------------------

/** The needs of the request now arriving: the manifest's row by arrival order, else the body's own. */
function needsFromBody(payload: Json): Requirement {
  const msgs = Array.isArray(payload.messages) ? payload.messages : [];
  let seesImage = false;
  for (const msg of msgs) {
    if (!isRecord(msg) || !Array.isArray(msg.content)) continue;
    for (const part of msg.content) {
      if (isRecord(part) && (part.type === 'image' || part.type === 'image_url')) seesImage = true;
    }
  }
  const fmt = isRecord(payload.response_format) ? payload.response_format.type : undefined;
  return { tools: Array.isArray(payload.tools) && payload.tools.length > 0, vision: seesImage, json: fmt === 'json_object' || fmt === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readBody(req: IncomingMessage): Promise<Json | null> {
  const pieces: Buffer[] = [];
  let bytes = 0;
  for await (const piece of req) {
    const buf = Buffer.isBuffer(piece) ? piece : Buffer.from(String(piece));
    bytes += buf.length;
    if (bytes > MAX_BODY) return null;
    pieces.push(buf);
  }
  if (pieces.length === 0) return null;
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(pieces).toString('utf8'));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function writeJson(res: ServerResponse, code: number, payload: unknown): void {
  const raw = JSON.stringify(payload);
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(raw) });
  res.end(raw);
}

function writeError(res: ServerResponse, wire: 'openai' | 'anthropic', code: number, kind: string, detail: string): void {
  if (wire === 'anthropic') writeJson(res, code, anthropicError(kind, detail));
  else writeJson(res, code, openAIError(detail, kind, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  const listed = listableModels();
  if (typeof req.headers['x-api-key'] === 'string') writeJson(res, 200, anthropicModelList(listed));
  else writeJson(res, 200, openAIModelList(listed));
}

function providerHeaders(target: Probe['provider'], streaming: boolean): Record<string, string> {
  const heads: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(target.auth, target.key),
  };
  if (streaming) heads.accept = 'text/event-stream';
  if (target.wire === 'anthropic') heads['anthropic-version'] = '2023-06-01';
  return heads;
}

/**
 * One call to one provider, in its own protocol, carrying the client's
 * request translated. Returns the raw reply; the caller decides whether
 * it counts as served. A thrown error (unreachable, timed out) is a
 * refusal.
 */
async function callProvider(
  api: 'openai' | 'anthropic',
  pick: Choice,
  payload: Json,
  streaming: boolean,
): Promise<Response> {
  const target = pick.probe.provider;
  const origin = joinBase(ENV_ORIGIN, target.base);
  const modelId = pick.model.id;
  const abort = AbortSignal.timeout(PROVIDER_TIMEOUT);
  if (target.wire === 'openai') {
    const forward: Json = api === 'openai' ? { ...payload, model: modelId } : anthropicToOpenAI(payload, modelId);
    if (streaming && forward.stream === true) forward.stream_options = { include_usage: true };
    return fetch(`${origin}/v1/chat/completions`, {
      method: 'POST',
      headers: providerHeaders(target, streaming),
      body: JSON.stringify(forward),
      signal: abort,
    });
  }
  if (target.wire === 'anthropic') {
    const forward: Json = api === 'anthropic' ? { ...payload, model: modelId } : openAIToAnthropicRequest(payload, modelId);
    return fetch(`${origin}/v1/messages`, {
      method: 'POST',
      headers: providerHeaders(target, streaming),
      body: JSON.stringify(forward),
      signal: abort,
    });
  }
  // Gemini wire.
  const forward: Json = api === 'openai' ? openAIToGemini(payload, modelId) : anthropicToGemini(payload, modelId);
  const path = streaming ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return fetch(`${origin}/v1beta/models/${modelId}${path}`, {
    method: 'POST',
    headers: providerHeaders(target, streaming),
    body: JSON.stringify(forward),
    signal: abort,
  });
}

/** Copy a provider's event stream to the client as it arrives. */
async function relayStream(source: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const consumer = source.getReader();
  try {
    for (;;) {
      const { value: chunk, done: finished } = await consumer.read();
      if (finished) break;
      res.write(Buffer.from(chunk));
    }
  } finally {
    consumer.releaseLock();
    res.end();
  }
}

/** Write a translated SSE stream to the client, in the client's wire. */
async function relayTranslated(frames: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  for await (const frame of frames) res.write(frame);
  res.end();
}

function writeStreamHead(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

/**
 * Answer one provider reply in the client's protocol. Returns true when
 * the reply was served (a 2xx the client can use); false counts as a
 * refusal and the caller retries on the next planned option.
 */
async function answerFromProvider(
  api: 'openai' | 'anthropic',
  pick: Choice,
  payload: Json,
  streaming: boolean,
  reply: Response,
  res: ServerResponse,
): Promise<boolean> {
  const providerWire = pick.probe.provider.wire;
  const modelId = pick.model.id;
  const askedModel = api === 'anthropic' ? (typeof payload.model === 'string' && payload.model !== '' ? payload.model : modelId) : modelId;
  if (!reply.ok || reply.body === null) {
    try {
      await reply.text();
    } catch {
      // The body is unreadable: the refusal stands either way.
    }
    return false;
  }
  if (streaming) {
    if (api === 'openai' && providerWire === 'openai') {
      writeStreamHead(res);
      await relayStream(reply.body, res);
      return true;
    }
    if (api === 'anthropic' && providerWire === 'anthropic') {
      writeStreamHead(res);
      await relayStream(reply.body, res);
      return true;
    }
    if (api === 'anthropic' && providerWire === 'openai') {
      await relayTranslated(anthropicEventsFromOpenAI(reply.body, askedModel), res);
      return true;
    }
    if (api === 'openai' && providerWire === 'anthropic') {
      await relayTranslated(openAIEventsFromAnthropic(reply.body, modelId), res);
      return true;
    }
    if (api === 'openai' && providerWire === 'gemini') {
      await relayTranslated(openAIEventsFromGemini(reply.body, modelId), res);
      return true;
    }
    if (api === 'anthropic' && providerWire === 'gemini') {
      await relayTranslated(anthropicEventsFromGemini(reply.body, modelId), res);
      return true;
    }
    return false;
  }
  const decoded: unknown = await reply.json().catch(() => null);
  if (!isRecord(decoded)) return false;
  if (api === 'openai' && providerWire === 'openai') {
    writeJson(res, 200, decoded);
    return true;
  }
  if (api === 'anthropic' && providerWire === 'anthropic') {
    writeJson(res, 200, decoded);
    return true;
  }
  if (api === 'anthropic' && providerWire === 'openai') {
    writeJson(res, 200, openAIToAnthropic(decoded, askedModel));
    return true;
  }
  if (api === 'openai' && providerWire === 'anthropic') {
    writeJson(res, 200, anthropicToOpenAIResponse(decoded, modelId));
    return true;
  }
  if (api === 'openai' && providerWire === 'gemini') {
    const translated = geminiToOpenAI(decoded, modelId);
    if (translated === null) return false;
    writeJson(res, 200, translated);
    return true;
  }
  if (api === 'anthropic' && providerWire === 'gemini') {
    const translated = geminiToAnthropic(decoded, modelId);
    if (translated === null) return false;
    writeJson(res, 200, translated);
    return true;
  }
  return false;
}

/**
 * Answer one client request: plan, call, and on a refusal re-plan and
 * retry. The manifest cursor is consumed by the caller, so `pending`
 * holds this request and the ones after it. When every option has refused
 * or been retired, the request is rescued a single time: the refusals
 * are wiped and the retired providers brought back, because a scheduled
 * refusal says nothing about a provider's next call, and a dropped
 * request costs far more than one more call.
 */
async function serveRequest(
  api: 'openai' | 'anthropic',
  payload: Json,
  streaming: boolean,
  pending: WorkloadRequest[],
  currentId: string,
  res: ServerResponse,
): Promise<void> {
  const refused = new Set<string>();
  let retrying = false;
  let revived = false;
  for (let round = 0; round < MAX_TRIES; round += 1) {
    const picks = planAssignments(pending, refused, retrying);
    const pick = picks.get(currentId);
    if (pick === undefined) {
      if (!revived) {
        refused.clear();
        for (const ledger of ledgers.values()) ledger.done = false;
        revived = true;
        continue;
      }
      break;
    }
    let reply: Response;
    try {
      reply = await callProvider(api, pick, payload, streaming);
    } catch {
      recordRefusal(pick.probe);
      refused.add(`${pick.probe.provider.id}/${pick.model.id}`);
      retrying = true;
      continue;
    }
    if (!reply.ok) {
      recordRefusal(pick.probe);
      refused.add(`${pick.probe.provider.id}/${pick.model.id}`);
      retrying = true;
      try {
        await reply.text();
      } catch {
        // Unreadable: the refusal stands.
      }
      continue;
    }
    const done = await answerFromProvider(api, pick, payload, streaming, reply, res);
    if (done) {
      recordSuccess(pick.probe);
      return;
    }
    // A 2xx that could not be translated is a refusal of that option.
    recordRefusal(pick.probe);
    refused.add(`${pick.probe.provider.id}/${pick.model.id}`);
    retrying = true;
  }
  if (!res.headersSent) {
    writeError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  } else {
    res.end();
  }
}

/** Take the manifest row for the arriving request and advance the cursor. */
function takeNext(payload: Json): { request: WorkloadRequest; rest: WorkloadRequest[] } {
  const rows = briefDoc.workload?.manifest ?? [];
  const pos = manifestCursor;
  manifestCursor += 1;
  const row = rows[pos];
  if (row !== undefined) return { request: row, rest: rows.slice(pos) };
  const synthetic: WorkloadRequest = {
    id: `req-${pos}`,
    api: 'openai',
    stream: false,
    weight: 1,
    requires: needsFromBody(payload),
  };
  return { request: synthetic, rest: [synthetic] };
}

/** POST /v1/chat/completions. */
async function serveChat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const payload = await readBody(req);
  if (payload === null) return writeError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { request, rest } = takeNext(payload);
  await serveRequest('openai', payload, payload.stream === true, rest, request.id, res);
}

/** POST /v1/messages. */
async function serveMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const payload = await readBody(req);
  if (payload === null) return writeError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { request, rest } = takeNext(payload);
  await serveRequest('anthropic', payload, payload.stream === true, rest, request.id, res);
}

async function routeRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && path.pathname === '/v1/models') {
    serveModels(req, res);
    return;
  }
  if (req.method === 'POST' && path.pathname === '/v1/chat/completions') {
    await serveChat(req, res);
    return;
  }
  if (req.method === 'POST' && path.pathname === '/v1/messages') {
    await serveMessages(req, res);
    return;
  }
  writeError(res, path.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${path.pathname}`);
}

function bindPort(srv: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    srv.once('error', reject);
    srv.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function registerEntry(portNumber: number, doc: CatalogReport): Promise<void> {
  const reply = await fetch(`${ENV_ORIGIN}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_V, port: portNumber, catalog: doc }),
    signal: AbortSignal.timeout(5000),
  });
  if (!reply.ok) throw new Error(`POST ${ENV_ORIGIN}/register answered HTTP ${reply.status}: ${(await reply.text()).slice(0, 500)}`);
  const answer: unknown = await reply.json();
  if (isRecord(answer) && typeof answer.requests === 'number') note(`registered: the workload holds ${answer.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function pollWork(srv: Server): void {
  let busy = false;
  const tick = setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const reply = await fetch(`${ENV_ORIGIN}/work`, { signal: AbortSignal.timeout(2000) });
        if (reply.status !== 200) return;
        const status: unknown = await reply.json();
        if (isRecord(status) && status.done === true) {
          clearInterval(tick);
          srv.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, POLL_MS);
}

async function boot(): Promise<void> {
  await loadCatalog();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const printed = `${JSON.stringify(catalogDoc)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(printed, () => resolve()));
    process.exit(0);
  }
  if (ENV_ORIGIN === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const srv = createServer((req, res) => {
    void routeRequest(req, res).catch((err: unknown) => {
      note((err as Error).stack ?? String(err));
      if (!res.headersSent) writeError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await bindPort(srv);
  const bound = srv.address();
  if (bound === null || typeof bound === 'string') throw new Error('the server has no port');
  const portNumber = bound.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (portNumber === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  note(`listening on 0.0.0.0:${portNumber} with ${catalogDoc.providers.length} answered providers and ${listableModels().length} free models`);
  await registerEntry(portNumber, catalogDoc);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(srv);
}

try {
  await boot();
} catch (err) {
  note((err as Error).stack ?? String(err));
  process.exit(1);
}
