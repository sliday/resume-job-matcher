import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import type { LanguageModel } from 'ai';
import type { EmbeddingModel } from 'ai';
import { EMBEDDING_MODELS, MODELS, type ApiMode } from './config.js';

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

/**
 * Embedding model for --prefilter. Anthropic has no embeddings API, so that mode
 * fails here with a usable instruction rather than at request time with a 404.
 */
export function getEmbeddingModel(mode: ApiMode): { model: EmbeddingModel; id: string } {
  const id = EMBEDDING_MODELS[mode];
  if (!id) {
    console.error(
      `--prefilter needs embeddings, and --api ${mode} does not provide them.\n` +
        `Re-run with --api openai or --api openrouter, or drop --prefilter.`,
    );
    process.exit(1);
  }
  if (mode === 'openrouter') {
    const openrouter = createOpenRouter({
      apiKey: requireKey('OPENROUTER_API_KEY', process.env.OPENROUTER_API_KEY),
    });
    return { model: openrouter.textEmbeddingModel(id), id };
  }
  const openai = createOpenAI({
    apiKey: requireKey('OPENAI_API_KEY', process.env.OPENAI_API_KEY),
  });
  return { model: openai.textEmbeddingModel(id), id };
}
