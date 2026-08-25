import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import type { LanguageModel } from 'ai';
import { MODELS, type ApiMode } from './config.js';

function requireKey(name: string, value: string | undefined): string {
  if (!value) {
    console.error(`Missing ${name}. Add it to .env (see .env-example).`);
    process.exit(1);
  }
  return value;
}

export function getModel(mode: ApiMode): LanguageModel {
  switch (mode) {
    case 'anthropic': {
      const anthropic = createAnthropic({
        apiKey: requireKey(
          'ANTHROPIC_API_KEY (or CLAUDE_API_KEY)',
          process.env.ANTHROPIC_API_KEY ?? process.env.CLAUDE_API_KEY,
        ),
      });
      return anthropic(MODELS.anthropic);
    }
    case 'openai': {
      const openai = createOpenAI({
        apiKey: requireKey('OPENAI_API_KEY', process.env.OPENAI_API_KEY),
      });
      return openai(MODELS.openai);
    }
    case 'openrouter': {
      const openrouter = createOpenRouter({
        apiKey: requireKey('OPENROUTER_API_KEY', process.env.OPENROUTER_API_KEY),
      });
      // openrouter/auto routes each request to a capable model; the
      // response-healing plugin repairs malformed JSON from weaker targets.
      return openrouter(MODELS.openrouter, { plugins: [{ id: 'response-healing' }] });
    }
  }
}
