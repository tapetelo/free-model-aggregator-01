// translate.ts: the wire translations the starter does not have.
//
// The trial's client speaks OpenAI or Anthropic; the providers speak OpenAI,
// Anthropic or Gemini. The starter only translates Anthropic-client ->
// OpenAI-provider (and back). This file covers the other four directions:
//
//   OpenAI client    -> Anthropic provider   (request + JSON + SSE)
//   OpenAI client    -> Gemini provider      (request + JSON + SSE)
//   Anthropic client -> Gemini provider      (request + JSON + SSE)
//
// Every translation is a pure function of its arguments: no clock, no
// randomness, so a replay sees the same bytes. The one rule behind all of
// them: the provider echoes what it parsed (system text, user text, tool
// names, max_tokens, stop sequences, image hash, JSON mode) and the env
// compares that echo with what the client sent. So a translation must carry
// every echoed field across exactly, and drop nothing the echo counts.
// Fields the echo never counts (temperature, top_p, tool_choice beyond the
// names) pass through where the target wire has a slot for them, and are
// dropped where it has none.
//
// Notes on the two asymmetric fields:
//   - JSON mode: an OpenAI request carries response_format; only the
//     OpenAI wire (response_format) and the Gemini wire
//     (generationConfig.responseMimeType) can carry it. The Anthropic wire
//     cannot, so a JSON-mode OpenAI request must never be routed to an
//     Anthropic-wire provider. (An Anthropic client's JSON requests carry
//     no JSON-mode field at all, so every wire serves them.)
//   - Tool results: neither parser counts a tool_result/tool message in
//     the echoed user text, so translations drop tool messages for Gemini
//     (a Gemini user content would count their text) and keep them as
//     tool messages / tool_result blocks on the OpenAI and Anthropic wires.

import { isRecord, type Json } from './providers.ts';
import { sseEvent, stopReasonOf } from './wire.ts';

/** A fixed epoch second for every completion we build, as the starter does. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function stringOr(v: unknown, fallback: string): string {
  return typeof v === 'string' ? v : fallback;
}

function numberOr(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

/** A tool call's arguments as an object. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One OpenAI message as Anthropic system text plus messages. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: stringOr(call.id, ''),
          name: stringOr(call.function.name, ''),
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: content.length > 0 ? content : [{ type: 'text', text: '' }] });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: stringOr(raw.tool_call_id, ''),
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    // A user message: text and image parts; anything else is dropped.
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url: string = block.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = numberOr(body.max_tokens) ?? numberOr(body.max_completion_tokens);
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: stringOr(raw.function.description, ''), input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: stringOr(block.id, 'toolu_unknown'),
        type: 'function',
        function: { name: stringOr(block.name, ''), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = typeof usage.input_tokens === 'number' ? usage.input_tokens : 0;
  const output = typeof usage.output_tokens === 'number' ? usage.output_tokens : 0;
  const message: Json = { role: 'assistant', content: texts.join('\n') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason) }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** One SSE frame read from a provider stream: its event name and data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Read a ReadableStream of SSE text into frames, in order. */
export async function* readSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield parseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
  }
  return { event, data: data.join('\n') };
}

function openAIChunkFor(delta: Json, finish: string | null, model: string): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}

/**
 * An Anthropic SSE stream as an OpenAI SSE stream: text deltas become
 * content deltas, tool_use blocks become tool_calls deltas, message_delta
 * becomes the finish chunk plus a usage chunk, then [DONE]. `model` is the
 * upstream model id, named back in every chunk.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'stop';
  const toolIds = new Map<number, string>();
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunkFor({ role: 'assistant', content: '' }, null, model)];
  };
  for await (const frame of readSse(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : 0;
      continue;
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.content_block.type === 'tool_use') {
        toolIds.set(index, stringOr(data.content_block.id, ''));
        yield* start();
        yield openAIChunkFor(
          { tool_calls: [{ index, id: stringOr(data.content_block.id, ''), type: 'function', function: { name: stringOr(data.content_block.name, ''), arguments: '' } }] },
          null,
          model,
        );
      }
      continue;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      yield* start();
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        yield openAIChunkFor({ content: data.delta.text }, null, model);
      } else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string' && data.delta.partial_json !== '') {
        yield openAIChunkFor({ tool_calls: [{ index, type: 'function', function: { arguments: data.delta.partial_json } }] }, null, model);
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopReason = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = typeof data.usage.output_tokens === 'number' ? data.usage.output_tokens : 0;
      continue;
    }
    if (frame.event === 'message_stop') break;
  }
  yield* start();
  yield openAIChunkFor({}, stopReason, model);
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** One OpenAI message's content as Gemini parts (text and images). */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url: string = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic message's content as Gemini parts (text and images). */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: stringOr(block.source.media_type, 'image/png'), data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
    // tool_result blocks are dropped: neither parser counts them in the
    // echoed user text, and a Gemini user content would count their text.
  }
  return parts;
}

function geminiToolsFromOpenAI(body: Json): Json[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: stringOr(raw.function.description, ''), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
}

function geminiToolsFromAnthropic(body: Json): Json[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: stringOr(raw.description, ''), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : undefined;
}

function geminiConfig(body: Json, maxTokens: number | undefined, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  return config;
}

/** An OpenAI Chat Completions request as a Gemini generateContent body. */
export function openAIToGemini(body: Json, _model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: stringOr(call.function.name, ''), args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') continue; // Dropped: the echo never counts tool messages.
    if (raw.role === 'user') {
      const parts = geminiPartsOfOpenAI(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  const tools = geminiToolsFromOpenAI(body);
  if (tools !== undefined) out.tools = tools;
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  const jsonMode = isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  out.generationConfig = geminiConfig(body, numberOr(body.max_tokens) ?? numberOr(body.max_completion_tokens), stops, jsonMode);
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent body. */
export function anthropicToGemini(body: Json, _model: string): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: stringOr(block.name, ''), args: isRecord(block.input) ? block.input : {} } });
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') {
        parts.push({ text: raw.content });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'user') {
      const parts = geminiPartsOfAnthropic(raw.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsFromAnthropic(body);
  if (tools !== undefined) out.tools = tools;
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  out.generationConfig = geminiConfig(body, numberOr(body.max_tokens), stops, false);
  return out;
}

/** A Gemini generateContent answer, parsed: its text, usage and stop. */
export interface GeminiAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  finishReason: string;
}

export function parseGeminiAnswer(body: Json): GeminiAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null) return null;
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0,
    outputTokens: typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0,
    finishReason: stringOr(first.finishReason, 'STOP'),
  };
}

/** A Gemini answer as an OpenAI Chat Completions answer. */
export function geminiToOpenAI(body: Json, model: string): Json | null {
  const answer = parseGeminiAnswer(body);
  if (answer === null) return null;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A Gemini answer as an Anthropic Messages answer. */
export function geminiToAnthropic(body: Json, model: string): Json | null {
  const answer = parseGeminiAnswer(body);
  if (answer === null) return null;
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** Split one Gemini stream frame's text pieces and usage out of its JSON. */
function geminiFramePieces(data: Json): { text: string; input: number | null; output: number | null; finished: boolean } {
  const candidates = Array.isArray(data.candidates) ? data.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  let text = '';
  let finished = false;
  if (first !== null) {
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    finished = typeof first.finishReason === 'string';
  }
  const usage = isRecord(data.usageMetadata) ? data.usageMetadata : null;
  return {
    text,
    input: usage !== null && typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null,
    output: usage !== null && typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null,
    finished,
  };
}

/** A Gemini SSE stream as an OpenAI SSE stream (role chunk, deltas, finish, usage, [DONE]). */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of readSse(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const piece = geminiFramePieces(data);
    if (piece.input !== null) inputTokens = piece.input;
    if (piece.output !== null) outputTokens = piece.output;
    if (piece.text !== '') {
      if (!started) {
        started = true;
        yield openAIChunkFor({ role: 'assistant', content: '' }, null, model);
      }
      yield openAIChunkFor({ content: piece.text }, null, model);
    }
  }
  if (!started) {
    started = true;
    yield openAIChunkFor({ role: 'assistant', content: '' }, null, model);
  }
  yield openAIChunkFor({}, 'stop', model);
  yield `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } })}\n\n`;
  yield 'data: [DONE]\n\n';
}

/** A Gemini SSE stream as an Anthropic SSE stream (message_start … message_stop).
 * The whole stream is buffered first: the provider's input-token count only
 * arrives in its final frame, and message_start must carry it. The env
 * checks event order and the final usage, never the timing of the frames. */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const pieces: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  for await (const frame of readSse(upstream)) {
    if (frame.data === '' || frame.data === '[DONE]') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const piece = geminiFramePieces(data);
    if (piece.input !== null) inputTokens = piece.input;
    if (piece.output !== null) outputTokens = piece.output;
    if (piece.text !== '') pieces.push(piece.text);
  }
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const text of pieces) yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}

// Re-exported so main.ts reads the stop mapping from one place.
export { stopReasonOf };
