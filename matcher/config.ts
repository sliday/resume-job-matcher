import 'dotenv/config';
import { parseArgs } from 'node:util';
import path from 'node:path';

export type ApiMode = 'anthropic' | 'openai' | 'openrouter';

/**
 * Filesystem-safe slug for a job description, so screening the same candidates
 * against a second job cannot overwrite the first job's emails. Outputs keyed on
 * the candidate filename alone silently collide across jobs.
 */
export function jobSlug(jobDescFile: string): string {
  const slug = path
    .parse(jobDescFile)
    .name.toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60);
  return slug || 'job';
}

export interface Config {
  mode: ApiMode;
  jobDescFile: string;
  /** out/<job-slug>; per-job so runs against different jobs never collide */
  outDir: string;
  pdfFolder: string;
  concurrency: number;
  prefilter: number;
  inviteThreshold: number;
  writeEmails: boolean;
  analyzeJd: boolean;
  unify: boolean;
  overallAnalysis: boolean;
}

// Anthropic has no embeddings API, so --prefilter is unavailable in that mode.
export const EMBEDDING_MODELS: Partial<Record<ApiMode, string>> = {
  openai: process.env.OPENAI_EMBEDDING_MODEL ?? 'text-embedding-3-small',
  openrouter: process.env.OPENROUTER_EMBEDDING_MODEL ?? 'openai/text-embedding-3-small',
};

export const MODELS: Record<ApiMode, string> = {
  anthropic: process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-5',
  openai: process.env.OPENAI_MODEL ?? 'gpt-5.6-luna',
  openrouter: process.env.OPENROUTER_MODEL ?? 'openrouter/auto',
};

export function loadConfig(argv: string[]): Config {
  const { values, positionals } = parseArgs({
    args: argv,
    options: {
      api: { type: 'string', default: 'openrouter' },
      concurrency: { type: 'string', default: '4' },
      prefilter: { type: 'string', default: '0' },
      threshold: { type: 'string', default: '90' },
      'no-email': { type: 'boolean', default: false },
      'analyze-jd': { type: 'boolean', default: false },
      unify: { type: 'boolean', default: false },
      'no-analysis': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
    allowPositionals: true,
  });

  if (values.help) {
    console.log(`Usage: npm run match -- [job_desc_file] [pdf_folder] [options]

Positionals:
  job_desc_file   Path to job description text file (default: job_description.txt)
  pdf_folder      Folder with resume PDFs (default: src)

Options:
  --api <mode>          anthropic | openai | openrouter (default: openrouter -> openrouter/auto)
  --concurrency <n>     Parallel resume evaluations (default: 4)
  --prefilter <n>       Embed resumes and keep only the n closest to the job before
                        scoring (default: 0 = off). Cheap first cut for large pools.
                        Needs --api openai or openrouter; Anthropic has no embeddings.
  --threshold <n>       Invite threshold for email generation (default: 90)
  --no-email            Skip candidate email generation
  --analyze-jd          Rank the job description and write job_description_enhanced.txt
  --unify               Standardize each resume to Markdown (out/<name>_unified.md) and score that
  --no-analysis         Skip the overall candidate-pool analysis
  -h, --help            Show this help

Env: ANTHROPIC_API_KEY (or CLAUDE_API_KEY), OPENAI_API_KEY, OPENROUTER_API_KEY,
     ANTHROPIC_MODEL, OPENAI_MODEL, OPENROUTER_MODEL`);
    process.exit(0);
  }

  const mode = values.api as ApiMode;
  if (!['anthropic', 'openai', 'openrouter'].includes(mode)) {
    console.error(`Unknown --api mode: ${mode}. Use anthropic | openai | openrouter.`);
    process.exit(1);
  }

  const jobDescFile = positionals[0] ?? 'job_description.txt';

  return {
    mode,
    jobDescFile,
    outDir: path.join('out', jobSlug(jobDescFile)),
    pdfFolder: positionals[1] ?? 'src',
    concurrency: Math.max(1, Number(values.concurrency) || 4),
    prefilter: Math.max(0, Number(values.prefilter) || 0),
    inviteThreshold: Math.min(100, Math.max(0, Number(values.threshold) || 90)),
    writeEmails: !values['no-email'],
    analyzeJd: Boolean(values['analyze-jd']),
    unify: Boolean(values.unify),
    overallAnalysis: !values['no-analysis'],
  };
}
