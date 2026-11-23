import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { z } from 'zod';

test('the default Anthropic CLI model uses native structured output', async (t) => {
  t.mock.method(globalThis, 'fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    assert.equal(request.url, 'https://api.anthropic.com/v1/messages');
    const payload = await request.json();
    assert.equal(payload.model, 'claude-sonnet-5-5');
    assert.equal(payload.output_config.format.type, 'json_schema');
    assert.equal(payload.tool_choice, undefined);

    return Response.json({
      id: 'msg_test',
      type: 'message',
      role: 'assistant',
      model: payload.model,
      content: [
        { type: 'thinking', thinking: 'Mock model reasoning', signature: 'fixture-signature' },
        { type: 'text', text: '{"matched":true}' },
      ],
      stop_reason: 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 12, output_tokens: 10 },
    });
  });

  const tempDir = mkdtempSync(join(tmpdir(), 'resume-matcher-model-'));
  const originalDotenvPath = process.env.DOTENV_CONFIG_PATH;
  const originalModel = process.env.ANTHROPIC_MODEL;
  const originalKey = process.env.ANTHROPIC_API_KEY;
  process.env.DOTENV_CONFIG_PATH = join(tempDir, '.env');
  delete process.env.ANTHROPIC_MODEL;
  process.env.ANTHROPIC_API_KEY = 'test-fixture-key';

  try {
    const { getModel } = await import('./ai.js');
    const { generateObject } = await import('./usage.js');
    const result = await generateObject({
      model: getModel('anthropic'),
      maxRetries: 0,
      schema: z.object({ matched: z.boolean() }),
      prompt: 'Return whether this fixture matches.',
    });
    assert.deepEqual(result.object, { matched: true });
  } finally {
    if (originalDotenvPath === undefined) delete process.env.DOTENV_CONFIG_PATH;
    else process.env.DOTENV_CONFIG_PATH = originalDotenvPath;
    rmSync(tempDir, { recursive: true, force: true });
    if (originalModel === undefined) delete process.env.ANTHROPIC_MODEL;
    else process.env.ANTHROPIC_MODEL = originalModel;
    if (originalKey === undefined) delete process.env.ANTHROPIC_API_KEY;
    else process.env.ANTHROPIC_API_KEY = originalKey;
  }
});
