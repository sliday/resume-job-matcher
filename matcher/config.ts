import 'dotenv/config';
import { parseArgs } from 'node:util';

export type ApiMode = 'anthropic' | 'openai' | 'openrouter';

export interface Config {
  mode: ApiMode;
  jobDescFile: string;
  pdfFolder: string;
  concurrency: number;
  inviteThreshold: number;
  writeEmails: boolean;
}

export const MODELS: Record<ApiMode, string> = {
  anthropic: process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-5',
  openai: process.env.OPENAI_MODEL ?? 'gpt-5',
  openrouter: process.env.OPENROUTER_MODEL ?? 'openrouter/auto',
};

export function loadConfig(argv: string[]): Config {
  const { values, positionals } = parseArgs({
    args: argv,
    options: {
      api: { type: 'string', default: 'openrouter' },
      concurrency: { type: 'string', default: '4' },
      threshold: { type: 'string', default: '90' },
      'no-email': { type: 'boolean', default: false },
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
  --threshold <n>       Invite threshold for email generation (default: 90)
  --no-email            Skip candidate email generation
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

  return {
    mode,
    jobDescFile: positionals[0] ?? 'job_description.txt',
    pdfFolder: positionals[1] ?? 'src',
    concurrency: Math.max(1, Number(values.concurrency) || 4),
    inviteThreshold: Math.min(100, Math.max(0, Number(values.threshold) || 90)),
    writeEmails: !values['no-email'],
  };
}
