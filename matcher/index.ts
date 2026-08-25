import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import pc from 'picocolors';
import { loadConfig, MODELS } from './config.js';
import { getModel } from './ai.js';
import { extractPdfText } from './pdf.js';
import {
  analyzeOverallMatches,
  estimateCostUsd,
  extractJobRequirements,
  generateCandidateEmail,
  getScoreDetails,
  getTokenUsage,
  improveJobDescription,
  matchResume,
  rankJobDescription,
  unifyResume,
  type MatchResult,
  type RedFlags,
} from './match.js';

type Colorize = (s: string) => string;
const COLORS: Record<string, Colorize> = {
  red: pc.red, green: pc.green, yellow: pc.yellow, blue: pc.blue,
  magenta: pc.magenta, cyan: pc.cyan, gray: pc.gray,
};
const paint = (color: string, s: string) => (COLORS[color] ?? ((x: string) => x))(s);

interface CandidateRow {
  filename: string;
  ok: boolean;
  score: number;
  matchReasons: string;
  website: string;
  redFlags: RedFlags | null;
  error?: string;
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index], index);
    }
  });
  await Promise.all(workers);
  return results;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

async function main() {
  const config = loadConfig(process.argv.slice(2));
  const model = getModel(config.mode);
  console.log(pc.green(`API mode: ${config.mode} (${MODELS[config.mode]})`));

  if (!existsSync(config.jobDescFile)) {
    console.error(pc.red(`Job description file not found: ${config.jobDescFile}`));
    process.exit(1);
  }
  const jobDesc = (await readFile(config.jobDescFile, 'utf-8')).trim();

  const pdfFiles = (await readdir(config.pdfFolder).catch(() => [] as string[]))
    .filter((f) => f.toLowerCase().endsWith('.pdf'))
    .map((f) => path.join(config.pdfFolder, f));
  if (pdfFiles.length === 0) {
    console.error(pc.red(`No PDF files found in ${config.pdfFolder}`));
    process.exit(1);
  }

  console.log('Extracting job requirements (once, shared by all candidates)...');
  const jobRequirements = await extractJobRequirements(model, jobDesc);

  if (config.analyzeJd) {
    console.log('Ranking job description...');
    const ranking = await rankJobDescription(model, jobDesc, jobRequirements);
    console.log(pc.bold('\nJob Description Ranking'));
    for (const [key, score] of Object.entries(ranking.scores)) {
      console.log(paint('cyan', `${key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}: ${score}%`));
    }
    console.log(paint('yellow', `\nOverall: ${ranking.overallScore}%`));
    console.log(pc.bold('\nImprovement Tips'));
    for (const tip of ranking.improvementTips) console.log(paint('green', `• ${tip}`));
    const improved = await improveJobDescription(model, jobDesc, ranking);
    if (improved) {
      await writeFile('job_description_enhanced.txt', improved);
      console.log(paint('green', '\nEnhanced job description saved to job_description_enhanced.txt\n'));
    } else {
      console.log(paint('red', '\nFailed to enhance job description\n'));
    }
  }

  console.log(`Matching ${pdfFiles.length} resumes (concurrency ${config.concurrency})...`);
  let done = 0;
  const rows = await mapWithConcurrency(pdfFiles, config.concurrency, async (file): Promise<CandidateRow> => {
    const filename = path.basename(file);
    try {
      let resumeText = await extractPdfText(file);
      if (resumeText.length < 100) {
        throw new Error('PDF text extraction too short (scanned image? OCR not supported in TS build)');
      }
      if (config.unify) {
        resumeText = await unifyResume(model, resumeText);
        await mkdir('out', { recursive: true });
        await writeFile(path.join('out', `${path.parse(filename).name}_unified.md`), resumeText);
      }
      const result: MatchResult = await matchResume(model, resumeText, jobRequirements);

      if (config.writeEmails) {
        const email = await generateCandidateEmail(model, resumeText, result.score, config.inviteThreshold);
        await mkdir('out', { recursive: true });
        const outFile = path.join('out', `${path.parse(filename).name}_response.txt`);
        await writeFile(outFile, `Subject: ${email.subject}\n\n${email.body}`);
      }

      process.stdout.write(`\r${++done}/${pdfFiles.length} done`);
      return { filename, ok: true, score: result.score, matchReasons: result.matchReasons, website: result.website, redFlags: result.redFlags };
    } catch (error) {
      process.stdout.write(`\r${++done}/${pdfFiles.length} done`);
      return { filename, ok: false, score: 0, matchReasons: '', website: '', redFlags: null, error: (error as Error).message };
    }
  });
  process.stdout.write('\n\n');

  const sorted = [...rows].sort((a, b) => b.score - a.score);
  const width = Math.max(...sorted.map((r) => r.filename.length));

  for (const row of sorted) {
    if (!row.ok) {
      console.log(paint('red', `🔴 ${row.filename.padEnd(width)}: Error: ${row.error}`));
      continue;
    }
    const { emoji, color, label } = getScoreDetails(row.score);
    const site = row.website ? ` - ${row.website}` : '';
    console.log(paint(color, `${emoji} ${row.filename.padEnd(width)}${site}: ${row.score}% - ${label}`));
    if (row.redFlags) {
      for (const [flag, names] of Object.entries(row.redFlags)) {
        if (names.length) console.log(paint('red', `  ${flag} ${names.join(', ')}`));
      }
    }
    if (row.score > 80 && row.matchReasons) {
      console.log(paint('cyan', `→ ${row.matchReasons}`));
    }
  }

  const okRows = sorted.filter((r) => r.ok);
  const errors = sorted.length - okRows.length;
  if (okRows.length > 0) {
    const scores = okRows.map((r) => r.score);
    console.log(pc.bold('\nSummary'));
    console.log(paint('yellow', `Top Score: ${Math.max(...scores)}%`));
    console.log(paint('cyan', `Average: ${(scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2)}%`));
    console.log(paint('green', `Median: ${median(scores).toFixed(2)}%`));
    console.log(paint('magenta', `Standard Deviation: ${stdev(scores).toFixed(2)}`));
    const above90 = scores.filter((s) => s >= 90).length;
    const above80 = scores.filter((s) => s >= 80).length;
    if (above90) console.log(paint('blue', `Resumes ≥ 90%: ${above90}`));
    if (above80) console.log(paint('cyan', `Resumes ≥ 80%: ${above80}`));
    console.log(paint('magenta', `Lowest Score: ${Math.min(...scores)}%`));
    console.log(paint('green', `Processed: ${okRows.length}`));
  }
  if (config.overallAnalysis && okRows.length > 0) {
    console.log(pc.bold('\nOverall Match Analysis'));
    try {
      const { analysis, suggestions } = await analyzeOverallMatches(
        model,
        jobDesc,
        okRows.map((r) => ({ filename: r.filename, score: r.score, matchReasons: r.matchReasons })),
      );
      console.log(analysis);
      for (const suggestion of suggestions) console.log(paint('green', `• ${suggestion}`));
    } catch (error) {
      console.log(paint('red', `Analysis failed: ${(error as Error).message}`));
    }
  }

  if (errors > 0) console.log(paint('red', `Errors: ${errors}`));

  const tokens = getTokenUsage();
  const cost = estimateCostUsd(MODELS[config.mode], tokens);
  const costText = cost === null ? '' : ` ≈ $${cost.toFixed(4)}`;
  console.log(
    paint('gray', `Tokens: ${tokens.input} in + ${tokens.output} out over ${tokens.calls} LLM calls${costText}`),
  );

  console.log(paint('yellow', '\nMatching Complete'));
}

main().catch((error) => {
  console.error(pc.red(`Fatal: ${error?.message ?? error}`));
  process.exit(1);
});
