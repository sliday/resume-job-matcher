import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import pc from 'picocolors';
import { loadConfig, MODELS } from './config.js';
import { getEmbeddingModel, getModel } from './ai.js';
import { extractPdfText } from './pdf.js';
import { BLOCKING_SEVERITY, deriveGateQuestions } from './gate.js';
import { cosineSimilarity, embedWithCache, selectTopK } from './prefilter.js';
import {
  analyzeOverallMatches,
  estimateCostUsd,
  extractJobRequirements,
  generateCandidateEmail,
  getScoreDetails,
  getTokenUsage,
  improveJobDescription,
  rankJobDescription,
  screenCandidate,
  unifyResume,
  type Disqualifier,
  type RedFlags,
  type ScreenDecision,
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
  decision: ScreenDecision | null;
  disqualifiers: Disqualifier[];
  score: number | null;
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

  console.log('Deriving gate questions (once, shared by all candidates)...');
  const gateQuestions = await deriveGateQuestions(model, jobDesc);
  for (const q of gateQuestions) {
    const kind = q.severity >= BLOCKING_SEVERITY ? pc.red('blocking') : pc.gray('advisory');
    console.log(paint('gray', `  [${q.severity}] ${q.question} `) + kind);
  }

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

  // Phase A: pull text out of every PDF first. Local, no LLM, so it is cheap enough
  // to do for the whole pool even when most of the pool is about to be filtered out.
  const extracted = await mapWithConcurrency(pdfFiles, config.concurrency, async (file) => {
    const filename = path.basename(file);
    try {
      const text = await extractPdfText(file);
      if (text.length < 100) {
        throw new Error('PDF text extraction too short (scanned image? OCR not supported in TS build)');
      }
      return { file, filename, text, error: undefined as string | undefined };
    } catch (error) {
      return { file, filename, text: '', error: (error as Error).message };
    }
  });

  // Phase B: optional embedding pre-filter. The point is to avoid the job x candidate
  // cross product: embedding is ~100x cheaper per candidate than gating and scoring,
  // and the cache is keyed on content, so a candidate is embedded once across all jobs.
  const readable = extracted.filter((e) => !e.error);
  let shortlist = extracted;
  let filteredOut: string[] = [];
  if (config.prefilter > 0 && readable.length > config.prefilter) {
    const { model: embedModel, id: embedId } = getEmbeddingModel(config.mode);
    const { vectors, embedded, reused } = await embedWithCache(embedModel, embedId, [
      jobDesc,
      ...readable.map((e) => e.text),
    ]);
    const [jobVector, ...resumeVectors] = vectors;
    const ranked = selectTopK(
      readable.map((e, i) => ({ key: e.filename, score: cosineSimilarity(jobVector, resumeVectors[i]) })),
      config.prefilter,
    );
    const keep = new Set(ranked.map((r) => r.key));
    filteredOut = readable.filter((e) => !keep.has(e.filename)).map((e) => e.filename);
    shortlist = extracted.filter((e) => e.error || keep.has(e.filename));
    console.log(
      paint(
        'gray',
        `Pre-filter (${embedId}): embedded ${embedded}, reused ${reused} from cache; ` +
          `kept top ${keep.size} of ${readable.length}, skipped ${filteredOut.length}`,
      ),
    );
  }

  // Phase C: the expensive pass, survivors only.
  console.log(`Matching ${shortlist.length} resumes (concurrency ${config.concurrency})...`);
  let done = 0;
  const rows = await mapWithConcurrency(shortlist, config.concurrency, async (entry): Promise<CandidateRow> => {
    const { filename } = entry;
    try {
      if (entry.error) throw new Error(entry.error);
      let resumeText = entry.text;
      if (config.unify) {
        resumeText = await unifyResume(model, resumeText);
        await mkdir('out', { recursive: true });
        await writeFile(path.join('out', `${path.parse(filename).name}_unified.md`), resumeText);
      }
      const result = await screenCandidate(model, resumeText, gateQuestions, jobRequirements);

      if (config.writeEmails) {
        // Gated-out candidates get an email too: a rejection is exactly what they are owed,
        // and score 0 drives the prompt to write one.
        const email = await generateCandidateEmail(
          model,
          resumeText,
          result.score ?? 0,
          config.inviteThreshold,
        );
        await mkdir('out', { recursive: true });
        const outFile = path.join('out', `${path.parse(filename).name}_response.txt`);
        await writeFile(outFile, `Subject: ${email.subject}\n\n${email.body}`);
      }

      process.stdout.write(`\r${++done}/${shortlist.length} done`);
      return {
        filename,
        ok: true,
        decision: result.decision,
        disqualifiers: result.gate.disqualifiers,
        score: result.score,
        matchReasons: result.matchReasons,
        website: result.website,
        redFlags: result.redFlags,
      };
    } catch (error) {
      process.stdout.write(`\r${++done}/${shortlist.length} done`);
      return {
        filename,
        ok: false,
        decision: null,
        disqualifiers: [],
        score: null,
        matchReasons: '',
        website: '',
        redFlags: null,
        error: (error as Error).message,
      };
    }
  });
  process.stdout.write('\n\n');

  const sorted = [...rows].sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  const width = Math.max(...sorted.map((r) => r.filename.length));

  for (const row of sorted) {
    if (!row.ok) {
      console.log(paint('red', `🔴 ${row.filename.padEnd(width)}: Error: ${row.error}`));
      continue;
    }
    if (row.decision === 'NO') {
      const reasons = row.disqualifiers.map((d) => d.question).join('; ');
      console.log(paint('red', `⛔ ${row.filename.padEnd(width)}: NO - ${reasons}`));
      for (const d of row.disqualifiers) {
        console.log(paint('gray', `   evidence: ${d.evidence}`));
      }
      continue;
    }
    const score = row.score ?? 0;
    const { emoji, color, label } = getScoreDetails(score);
    const site = row.website ? ` - ${row.website}` : '';
    console.log(paint(color, `${emoji} ${row.filename.padEnd(width)}${site}: ${score}% - ${label}`));
    if (row.redFlags) {
      for (const [flag, names] of Object.entries(row.redFlags)) {
        if (names.length) console.log(paint('red', `  ${flag} ${names.join(', ')}`));
      }
    }
    if (score > 80 && row.matchReasons) {
      console.log(paint('cyan', `→ ${row.matchReasons}`));
    }
  }

  const scoredRows = sorted.filter((r) => r.ok && r.score !== null);
  const gatedOut = sorted.filter((r) => r.ok && r.decision === 'NO').length;
  const errors = sorted.length - sorted.filter((r) => r.ok).length;
  if (scoredRows.length > 0) {
    const scores = scoredRows.map((r) => r.score as number);
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
    console.log(paint('green', `Scored: ${scoredRows.length}`));
  }
  if (gatedOut) console.log(paint('red', `Gated out: ${gatedOut}`));
  if (filteredOut.length) {
    // Named, not just counted: a silent drop is indistinguishable from a bug.
    const shown = filteredOut.slice(0, 5).join(', ');
    const rest = filteredOut.length > 5 ? `, +${filteredOut.length - 5} more` : '';
    console.log(paint('gray', `Pre-filtered (never scored): ${filteredOut.length} - ${shown}${rest}`));
  }
  if (config.overallAnalysis && scoredRows.length > 0) {
    console.log(pc.bold('\nOverall Match Analysis'));
    try {
      const { analysis, suggestions } = await analyzeOverallMatches(
        model,
        jobDesc,
        scoredRows.map((r) => ({
          filename: r.filename,
          score: r.score as number,
          matchReasons: r.matchReasons,
        })),
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
