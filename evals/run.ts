import { readFile, writeFile, mkdir } from 'node:fs/promises';
import pc from 'picocolors';
import { getModel } from '../matcher/ai.js';
import { deriveGateQuestions, type GateQuestion } from '../matcher/gate.js';
import {
  extractJobRequirements,
  generateCandidateEmail,
  screenCandidate,
} from '../matcher/match.js';
import type { JobRequirements } from '../matcher/schemas.js';
import { FIXTURES, GERMAN_BACKEND_JD, type Fixture } from './fixtures.js';

interface FixtureResult {
  name: string;
  pass: boolean;
  score: number | null;
  failures: string[];
}

async function runFixture(
  model: ReturnType<typeof getModel>,
  fixture: Fixture,
  requirements: JobRequirements,
  gateQuestions: GateQuestion[],
): Promise<FixtureResult> {
  const failures: string[] = [];
  const result = await screenCandidate(model, fixture.resume, gateQuestions, requirements);
  const score = result.score;
  const e = fixture.expect;

  if (e.decision && result.decision !== e.decision) {
    failures.push(
      `decision ${result.decision} !== expected ${e.decision} (gate: ${JSON.stringify(result.gate.disqualifiers)})`,
    );
  }
  if (e.disqualifierContains) {
    const haystack = result.gate.disqualifiers
      .map((d) => `${d.question} ${d.evidence}`)
      .join(' | ')
      .toLowerCase();
    if (!haystack.includes(e.disqualifierContains.toLowerCase())) {
      failures.push(
        `expected disqualifier containing "${e.disqualifierContains}", got ${JSON.stringify(result.gate.disqualifiers)}`,
      );
    }
  }
  // A gated-out candidate has no score. scoreMin still fails, because we expected a
  // keeper and the gate rejected them: that is the over-aggressive-gate alarm.
  // scoreMax is satisfied by a NO, because NO is strictly stronger than "below X".
  if (e.scoreMin !== undefined) {
    if (score === null) {
      failures.push(
        `expected score >= ${e.scoreMin} but candidate was gated out: ${JSON.stringify(result.gate.disqualifiers)}`,
      );
    } else if (score < e.scoreMin) {
      failures.push(`score ${score} < min ${e.scoreMin}`);
    }
  }
  if (e.scoreMax !== undefined && score !== null && score > e.scoreMax) {
    failures.push(`score ${score} > max ${e.scoreMax}`);
  }
  if (e.redFlagContains) {
    const flagged = result.redFlags ? [...result.redFlags['🚩'], ...result.redFlags['📍']] : [];
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
    const email = await generateCandidateEmail(model, fixture.resume, score ?? 0, 90);
    const text = `${email.subject}\n${email.body}`;
    if (!text.includes(e.emailNameIncludes)) {
      failures.push(`email does not mention "${e.emailNameIncludes}": ${email.body.slice(0, 120)}`);
    }
  }

  return { name: fixture.name, pass: failures.length === 0, score, failures };
}

async function main() {
  const model = getModel((process.env.EVAL_API as 'openai' | 'anthropic' | 'openrouter') ?? 'openai');
  const frontendJd = (await readFile('EXAMPLE job_description.txt', 'utf-8')).trim();
  const jds: Record<Fixture['jd'], string> = { frontend: frontendJd, 'german-backend': GERMAN_BACKEND_JD };

  console.log('Extracting requirements and gate questions per JD...');
  const requirements = {} as Record<Fixture['jd'], JobRequirements>;
  const gates = {} as Record<Fixture['jd'], GateQuestion[]>;
  for (const key of Object.keys(jds) as Fixture['jd'][]) {
    requirements[key] = await extractJobRequirements(model, jds[key]);
    gates[key] = await deriveGateQuestions(model, jds[key]);
    for (const q of gates[key]) {
      console.log(pc.gray(`  ${key} [${q.severity}] ${q.question}`));
    }
  }

  const results: FixtureResult[] = [];
  for (const fixture of FIXTURES) {
    process.stdout.write(`${fixture.name}... `);
    try {
      const result = await runFixture(model, fixture, requirements[fixture.jd], gates[fixture.jd]);
      results.push(result);
      const shown = result.score === null ? 'NO' : `${result.score}%`;
      console.log(result.pass ? pc.green(`PASS (${shown})`) : pc.red(`FAIL (${shown})`));
      for (const failure of result.failures) console.log(pc.red(`    ${failure}`));
    } catch (error) {
      results.push({ name: fixture.name, pass: false, score: null, failures: [(error as Error).message] });
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
