// The client protocols and the upstream protocol this starter speaks.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a
// JSON body or with server-sent events. This starter forwards both to a
// provider on the openai protocol and translates back:
//
//   - the OpenAI path hands the client's body (with the routed model's
//     id) to the provider and streams the provider's JSON or SSE straight
//     back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request into Chat Completions
//     (system prompt, image blocks, tool definitions, tool results and
//     tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start /
//     content_block_delta / content_block_stop / message_delta /
//     message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta
//     events.
//
// The gemini and anthropic upstream protocols are not covered in this
// starter (main.ts answers 501 for a provider on either): a real entry
// needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no
// chance, so the same body and the same provider answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const OPENAI_CREATED_AT = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: OPENAI_CREATED_AT, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function toPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = extractText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function parseArgs(value: unknown): Json {
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

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const turns: Json[] = [];
  const sysText = extractText(body.system);
  if (sysText !== '') turns.push({ role: 'system', content: sysText });
  const rows = Array.isArray(body.messages) ? body.messages : [];
  for (const item of rows) {
    if (!isRecord(item)) continue;
    const fromAssistant = item.role === 'assistant';
    const body_content = item.content;
    if (typeof body_content === 'string') {
      turns.push({ role: fromAssistant ? 'assistant' : 'user', content: body_content });
      continue;
    }
    if (!Array.isArray(body_content)) continue;
    if (fromAssistant) {
      const calls: Json[] = [];
      for (const part of body_content) {
        if (!isRecord(part) || part.type !== 'tool_use') continue;
        calls.push({
          id: typeof part.id === 'string' ? part.id : '',
          type: 'function',
          function: {
            name: typeof part.name === 'string' ? part.name : '',
            arguments: JSON.stringify(isRecord(part.input) ? part.input : {}),
          },
        });
      }
      const plain = extractText(body_content);
      turns.push({
        role: 'assistant',
        content: plain !== '' ? plain : null,
        ...(calls.length > 0 ? { tool_calls: calls } : {}),
      });
      continue;
    }
    const pieces: Json[] = [];
    for (const part of body_content) {
      if (!isRecord(part)) continue;
      if (part.type === 'tool_result') {
        turns.push({
          role: 'tool',
          tool_call_id: typeof part.tool_use_id === 'string' ? part.tool_use_id : '',
          content: resultText(part.content),
        });
        continue;
      }
      const piece = toPart(part);
      if (piece !== null) pieces.push(piece);
    }
    if (pieces.length > 0) turns.push({ role: 'user', content: pieces });
  }

  const converted: Json = { model, messages: turns };
  if (typeof body.max_tokens === 'number') converted.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') converted.temperature = body.temperature;
  if (typeof body.top_p === 'number') converted.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) converted.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const defs = body.tools.flatMap((item) => {
      if (!isRecord(item) || typeof item.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: item.name,
            description: typeof item.description === 'string' ? item.description : '',
            parameters: isRecord(item.input_schema) ? item.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (defs.length > 0) converted.tools = defs;
  }
  const pick = body.tool_choice;
  if (isRecord(pick)) {
    if (pick.type === 'auto') converted.tool_choice = 'auto';
    else if (pick.type === 'any') converted.tool_choice = 'required';
    else if (pick.type === 'tool' && typeof pick.name === 'string') converted.tool_choice = { type: 'function', function: { name: pick.name } };
  }
  if (body.stream === true) {
    converted.stream = true;
    converted.stream_options = { include_usage: true };
  }
  return converted;
}

function asNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const options = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(options[0]) ? options[0] : {};
  const msg = isRecord(first.message) ? first.message : {};
  const blocks: Json[] = [];
  const plain = typeof msg.content === 'string' ? msg.content : extractText(msg.content);
  if (plain !== '') blocks.push({ type: 'text', text: plain });
  const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
  for (const item of toolCalls) {
    if (!isRecord(item) || !isRecord(item.function)) continue;
    blocks.push({
      type: 'tool_use',
      id: typeof item.id === 'string' ? item.id : 'toolu_unknown',
      name: typeof item.function.name === 'string' ? item.function.name : '',
      input: parseArgs(item.function.arguments),
    });
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
  const counts = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: blocks,
    stop_reason: stopReasonOf(first.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: asNumber(counts.prompt_tokens), output_tokens: asNumber(counts.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const consumer = upstream.getReader();
  const utf8 = new TextDecoder();
  let pending = '';
  let begun = false;
  let closed = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let heldStop: string | null = null;
  let textSlot = -1;
  let nextSlot = 0;
  const toolSlots = new Map<number, number>();
  let inTokens = 0;
  let outTokens = 0;

  const openEvents = (): string[] => {
    if (begun) return [];
    begun = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const shutEvents = (): string[] => {
    const out: string[] = [];
    if (textSlot !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textSlot }));
      textSlot = -1;
    }
    for (const slot of [...toolSlots.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: slot }));
    toolSlots.clear();
    return out;
  };
  const endEvents = (stopReason: string): string[] => {
    if (closed) return [];
    closed = true;
    return [
      ...shutEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const frameEvents = (frame: string): string[] => {
    const out: string[] = [];
    for (const row of frame.split('\n')) {
      const clean = row.trim();
      if (!clean.startsWith('data:')) continue;
      const raw = clean.slice(5).trim();
      if (raw === '[DONE]') {
        out.push(...openEvents(), ...endEvents(heldStop ?? 'end_turn'));
        continue;
      }
      let piece: unknown;
      try {
        piece = JSON.parse(raw);
      } catch {
        continue;
      }
      if (!isRecord(piece)) continue;
      if (isRecord(piece.usage)) {
        inTokens = asNumber(piece.usage.prompt_tokens);
        outTokens = asNumber(piece.usage.completion_tokens);
      }
      const options = Array.isArray(piece.choices) ? piece.choices : [];
      const first = isRecord(options[0]) ? options[0] : null;
      if (first === null) continue;
      const diff = isRecord(first.delta) ? first.delta : {};
      out.push(...openEvents());
      if (typeof diff.content === 'string' && diff.content !== '') {
        if (textSlot === -1) {
          textSlot = nextSlot;
          nextSlot += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textSlot, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textSlot, delta: { type: 'text_delta', text: diff.content } }));
      }
      const toolCalls = Array.isArray(diff.tool_calls) ? diff.tool_calls : [];
      for (const item of toolCalls) {
        if (!isRecord(item)) continue;
        const sourceIndex = typeof item.index === 'number' ? item.index : toolSlots.size;
        let slot = toolSlots.get(sourceIndex);
        if (slot === undefined) {
          if (textSlot !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textSlot }));
            textSlot = -1;
          }
          slot = nextSlot;
          nextSlot += 1;
          toolSlots.set(sourceIndex, slot);
          const func = isRecord(item.function) ? item.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index: slot,
              content_block: {
                type: 'tool_use',
                id: typeof item.id === 'string' && item.id !== '' ? item.id : `toolu_${slot}`,
                name: typeof func.name === 'string' ? func.name : '',
                input: {},
              },
            }),
          );
        }
        const func = isRecord(item.function) ? item.function : {};
        if (typeof func.arguments === 'string' && func.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: slot, delta: { type: 'input_json_delta', partial_json: func.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (first.finish_reason !== null && first.finish_reason !== undefined) heldStop = stopReasonOf(first.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await consumer.read();
      if (done) break;
      pending += utf8.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let cut = pending.indexOf('\n\n');
      while (cut !== -1) {
        const frame = pending.slice(0, cut);
        pending = pending.slice(cut + 2);
        for (const event of frameEvents(frame)) yield event;
        cut = pending.indexOf('\n\n');
      }
    }
    pending += utf8.decode();
    if (pending.trim() !== '') for (const event of frameEvents(pending)) yield event;
  } finally {
    consumer.releaseLock();
  }
  if (!closed) {
    for (const event of openEvents()) yield event;
    for (const event of endEvents(heldStop ?? 'end_turn')) yield event;
  }
}
