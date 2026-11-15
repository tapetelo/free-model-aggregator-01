// Reading the brief's three documents, then probing the catalogs.
//
// A trial's brief (the file MAKEFUN_PROMPT_FILE names) holds three JSON
// documents, and the prose between them is irrelevant: this file picks
// each one out by its `schema` member.
//
//   - the provider directory (`free-model-aggregator-01/providers/v1`): a
//     row per provider, carrying its id, its protocol, its auth scheme,
//     the environment variable a real deployment would take the key
//     from, the key the simulated service accepts, and the base path the
//     entry joins onto MAKEFUN_ENV_URL. A directory row may carry an
//     absolute base_url in place of a path; joinBase accepts either.
//   - the quality table (`free-model-aggregator-01/ratings/v1`): every
//     model's public rating in milli-points, what it can do (tools,
//     vision, JSON mode, context length) and whether it is free. The
//     scorer grades a reply by its rating; a paid model is never a
//     legal pick. This baseline reads the table's capability columns and
//     free flag (the sealed truth, where a row exists) but never its
//     rating (`--list` shows why in main.ts's header, and the routing
//     TODO there names the gap).
//   - the workload manifest (`free-model-aggregator-01/workload/v1`):
//     the requests the client will send, in order, each with its api,
//     whether it streams, its weight and the needs a model must meet to
//     answer it.
//
// Each provider's catalog route is then probed in the provider's own
// protocol:
//
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A probe that fails (a refused connection, a non-2xx, a body that is
// not JSON) drops its provider from the catalog and sends it no traffic;
// the trial carries on. Everything here is deterministic: no clock, no
// chance, so `node main.ts --list` and the register body build the same
// document (main.ts prints or posts this one object).

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief lists it. `base` is the base path (or absolute URL) its routes hang off. */
export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

export interface ProviderDirectory {
  schema: string;
  providers: ProviderDirectoryEntry[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** One row of the sealed quality table (the brief's copy). */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsFile {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs: a model without it cannot serve the request. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface WorkloadFile {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The brief as the baseline reads it. `error` is set when it could not be read at all. */
export interface Brief {
  directory: ProviderDirectory;
  ratings: RatingsFile | null;
  workload: WorkloadFile | null;
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what the listing did not say. */
export interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered, and the models it listed. */
export interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  status: number;
  models: ProbedModel[];
  error: string | null;
}

/** What a model can do, from the listing first and the quality table second (null: neither said). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, tail: string): string {
  const cleaned = tail.trim();
  if (/^https?:\/\//i.test(cleaned)) return cleaned.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (cleaned === '') return root;
  return `${root}${cleaned.startsWith('/') ? '' : '/'}${cleaned.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
export function authHeaders(scheme: AuthScheme, secret: string): Record<string, string> {
  if (secret === '') return {};
  if (scheme === 'bearer') return { authorization: `Bearer ${secret}` };
  if (scheme === 'x-api-key') return { 'x-api-key': secret };
  if (scheme === 'x-goog-api-key') return { 'x-goog-api-key': secret };
  return {};
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function scanDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const close = matchingBrace(text, at);
    if (close === -1) continue;
    try {
      const parsed: unknown = JSON.parse(text.slice(at, close + 1));
      if (isRecord(parsed) && typeof parsed.schema === 'string') found.push(parsed);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return found;
}

/** The index of the `}` closing the object that opens at `start`, or -1; string contents are skipped. */
function matchingBrace(text: string, start: number): number {
  let level = 0;
  let quoted = false;
  for (let pos = start; pos < text.length; pos++) {
    const char = text[pos];
    if (quoted) {
      if (char === '\\') pos += 1;
      else if (char === '"') quoted = false;
    } else if (char === '"') quoted = true;
    else if (char === '{') level += 1;
    else if (char === '}') {
      level -= 1;
      if (level === 0) return pos;
    }
  }
  return -1;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function parseProvider(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const pid = asString(v.id);
  const wireKind = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (pid === '' || wireKind === null) return null;
  const scheme = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a directory that spells an
  // absolute base instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const root =
    asString(v.path) !== '' ? asString(v.path) : asString(v.base_path) !== '' ? asString(v.base_path) : asString(v.base_url) !== '' ? asString(v.base_url) : asString(v.base);
  return { id: pid, name: asString(v.name) !== '' ? asString(v.name) : pid, wire: wireKind, auth: scheme, base: root, key_env: asString(v.key_env), key: asString(v.key) };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: asNumberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadRequest(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const needs = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: needs.tools === true,
      vision: needs.vision === true,
      json: needs.json === true,
      ...(typeof needs.min_context === 'number' && Number.isFinite(needs.min_context) ? { min_context: needs.min_context } : {}),
    },
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const directory: ProviderDirectory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let source: string;
  try {
    source = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const found = scanDocuments(source);
  const dirDoc = found.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const entries = dirDoc.providers.flatMap((v) => {
    const entry = parseProvider(v);
    return entry === null ? [] : [entry];
  });
  const ratesDoc = found.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const rows = ratesDoc !== null && Array.isArray(ratesDoc.ratings) ? ratesDoc.ratings : null;
  const table: RatingsFile | null =
    ratesDoc === null || rows === null
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: rows.flatMap((v) => {
            const row = parseRating(v);
            return row === null ? [] : [row];
          }),
          best: Array.isArray(ratesDoc.best)
            ? ratesDoc.best.flatMap((v) => {
                if (!isRecord(v) || asString(v.id) === '') return [];
                return [{ id: asString(v.id), rating_milli: asNumberOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workDoc = found.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const load: WorkloadFile | null =
    workDoc === null || !Array.isArray(workDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workDoc.manifest.flatMap((v) => {
            const req = parseWorkloadRequest(v);
            return req === null ? [] : [req];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers: entries }, ratings: table, workload: load, error: null };
}

/** The rating row of one provider's model, when the quality table has one. */
export function ratingOfModel(table: RatingsFile | null, provider: string, model: string): RatingRow | null {
  if (table === null) return null;
  return table.ratings.find((r) => r.provider === provider && r.model === model) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (it is the sealed truth), the
 * listing's own word where it does not, and paid when neither says. A paid model is never routed.
 */
export function modelFree(entry: ProbedModel, rating: RatingRow | null): boolean {
  if (rating !== null) return rating.free;
  return entry.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's own fields where it does not. */
export function capabilitiesOf(entry: ProbedModel, rating: RatingRow | null): Capabilities {
  return {
    context_length: rating?.context_length ?? entry.context_length ?? null,
    tools: rating?.tools ?? entry.tools ?? null,
    vision: rating?.vision ?? entry.vision ?? null,
    json: rating?.json ?? entry.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(needs: Requirement, able: Capabilities): boolean {
  if (needs.tools === true && able.tools !== true) return false;
  if (needs.vision === true && able.vision !== true) return false;
  if (needs.json === true && able.json !== true) return false;
  if (needs.min_context !== undefined && (able.context_length === null || able.context_length < needs.min_context)) return false;
  return true;
}

/** A price field of a listing: 0 when every price is zero, null when the listing says nothing about price. */
function pricedZero(v: Json): boolean | null {
  const priceMap = isRecord(v.pricing) ? v.pricing : null;
  if (priceMap === null) return null;
  const amounts = Object.values(priceMap).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (amounts.length === 0) return null;
  return amounts.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function parseModel(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const mid = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (mid === '') return null;
  const span = asNumberOrNull(v.context_length) ?? asNumberOrNull(v.context_window) ?? asNumberOrNull(v.inputTokenLimit);
  return {
    id: mid,
    display_name: asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : mid,
    free: asBoolOrNull(v.free) ?? pricedZero(v),
    context_length: span,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function parseModelList(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const rows = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(rows)) return [];
  return rows.flatMap((v) => {
    const entry = parseModel(wire, v);
    return entry === null ? [] : [entry];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry, timeoutMs = 5000): Promise<Probe> {
  const origin = joinBase(envUrl, provider.base);
  const target = provider.wire === 'gemini' ? `${origin}/v1beta/models` : `${origin}/v1/models`;
  const heads: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') heads['anthropic-version'] = '2023-06-01';
  try {
    const reply = await fetch(target, { headers: heads, signal: AbortSignal.timeout(timeoutMs) });
    if (!reply.ok) return { provider, ok: false, status: reply.status, models: [], error: `GET ${target} answered HTTP ${reply.status}` };
    const decoded: unknown = await reply.json();
    return { provider, ok: true, status: reply.status, models: parseModelList(provider.wire, decoded), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${target} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that answered, every model it listed.
 *
 * Its base_url is the directory's own path (`/p/<id>`), never the path joined to `envUrl`: the env's
 * origin differs between a live pass and its replay (a fresh loopback port under the process driver, a
 * fresh pod on the k8s layout), and the catalog is part of the state export a replay must reproduce byte
 * for byte. The scorer compares a reported base_url by its path suffix, so the path passes exactly as the
 * joined URL does; the joined URL reaches a provider, the path names it.
 */
export function buildCatalog(_origin: string, probes: readonly Probe[], table: RatingsFile | null): CatalogReport {
  const listed: CatalogProvider[] = [];
  for (const result of probes) {
    if (!result.ok) continue;
    const known = new Set<string>();
    const entries: CatalogModel[] = [];
    for (const entry of [...result.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (known.has(entry.id)) continue;
      known.add(entry.id);
      const rating = ratingOfModel(table, result.provider.id, entry.id);
      entries.push({ id: entry.id, free: modelFree(entry, rating), context_length: rating?.context_length ?? entry.context_length ?? null });
    }
    listed.push({ id: result.provider.id, base_url: result.provider.base, key_env: result.provider.key_env, wire: result.provider.wire, models: entries });
  }
  return { providers: listed };
}
