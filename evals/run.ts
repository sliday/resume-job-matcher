import { readFile, writeFile, mkdir } from 'node:fs/promises';
import pc from 'picocolors';
import { getModel } from '../matcher/ai.js';
import {
  extractJobRequirements,
  generateCandidateEmail,
  matchResume,
} from '../matcher/match.js';
import type { JobRequirements } from '../matcher/schemas.js';
import { FIXTURES, GERMAN_BACKEND_JD, type Fixture } from './fixtures.js';

interface FixtureResult {
  name: string;
  pass: boolean;
  score: number;
  failures: string[];
}

async function runFixture(
  model: ReturnType<typeof getModel>,
  fixture: Fixture,
  requirements: JobRequirements,
): Promise<FixtureResult> {
  const failures: string[] = [];
  const result = await matchResume(model, fixture.resume, requirements);
  const e = fixture.expect;

  if (e.scoreMin !== undefined && result.score < e.scoreMin) {
    failures.push(`score ${result.score} < min ${e.scoreMin}`);
  }
  if (e.scoreMax !== undefined && result.score > e.scoreMax) {
    failures.push(`score ${result.score} > max ${e.scoreMax}`);
  }
  if (e.redFlagContains) {
    const flagged = [...result.redFlags['🚩'], ...result.redFlags['📍']];
    if (!flagged.some((n) => n.includes(e.redFlagContains!))) {
      failures.push(`expected red flag containing "${e.redFlagContains}", got ${JSON.stringify(result.redFlags)}`);
    }
  }
  if (e.websiteIncludes && !result.website.includes(e.websiteIncludes)) {
    failures.push(`website "${result.website}" missing "${e.websiteIncludes}"`);
  }
  if (e.websiteEmpty && result.website !== '') {
    failures.push(`expected empty website, got "${result.website}"`);
  }
  if (e.emailNameIncludes) {
    const email = await generateCandidateEmail(model, fixture.resume, result.score, 90);
    const text = `${email.subject}\n${email.body}`;
    if (!text.includes(e.emailNameIncludes)) {
      failures.push(`email does not mention "${e.emailNameIncludes}": ${email.body.slice(0, 120)}`);
    }
  }

  return { name: fixture.name, pass: failures.length === 0, score: result.score, failures };
}

async function main() {
  const model = getModel((process.env.EVAL_API as 'openai' | 'anthropic' | 'openrouter') ?? 'openai');
  const frontendJd = (await readFile('EXAMPLE job_description.txt', 'utf-8')).trim();
  const jds: Record<Fixture['jd'], string> = { frontend: frontendJd, 'german-backend': GERMAN_BACKEND_JD };

  console.log('Extracting requirements per JD...');
  const requirements = {} as Record<Fixture['jd'], JobRequirements>;
  for (const key of Object.keys(jds) as Fixture['jd'][]) {
    requirements[key] = await extractJobRequirements(model, jds[key]);
  }

  const results: FixtureResult[] = [];
  for (const fixture of FIXTURES) {
    process.stdout.write(`${fixture.name}... `);
    try {
      const result = await runFixture(model, fixture, requirements[fixture.jd]);
      results.push(result);
      console.log(result.pass ? pc.green(`PASS (${result.score}%)`) : pc.red(`FAIL (${result.score}%)`));
      for (const failure of result.failures) console.log(pc.red(`    ${failure}`));
    } catch (error) {
      results.push({ name: fixture.name, pass: false, score: -1, failures: [(error as Error).message] });
      console.log(pc.red(`ERROR: ${(error as Error).message}`));
    }
  }

  const passed = results.filter((r) => r.pass).length;
  await mkdir('evals', { recursive: true });
  await writeFile(
    'evals/results.json',
    JSON.stringify({ date: new Date().toISOString(), passed, total: results.length, results }, null, 2),
  );
  console.log(`\n${passed}/${results.length} fixtures passed — evals/results.json written`);
  process.exit(passed === results.length ? 0 : 1);
}

main().catch((error) => {
  console.error(pc.red(`Eval harness fatal: ${error?.message ?? error}`));
  process.exit(1);
});
