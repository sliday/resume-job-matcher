/**
 * FMEA severity at or above which a failed gate question blocks a candidate.
 * Below this, a failure is advisory: it is recorded but does not stop scoring.
 */
export const BLOCKING_SEVERITY = 7;

export interface GateQuestion {
  /** kebab-case slug, stable across a run */
  id: string;
  /** Binary question, phrased so that YES means the candidate is acceptable */
  question: string;
  /** FMEA severity 1-10. 10 = impossible to hire. */
  severity: number;
  /** The job-description span that justifies this gate */
  why: string;
}

export type GateVerdict = 'PASS' | 'FAIL' | 'UNCERTAIN';

export interface GateAnswer {
  id: string;
  verdict: GateVerdict;
  /** Exact resume quote, or "not stated" */
  evidence: string;
}

export interface Disqualifier {
  id: string;
  question: string;
  severity: number;
  evidence: string;
}

export interface GateResult {
  passed: boolean;
  answers: GateAnswer[];
  disqualifiers: Disqualifier[];
}

/**
 * Pure decision rule, no LLM.
 *
 * Only a FAIL on a question at or above BLOCKING_SEVERITY stops a candidate.
 * UNCERTAIN and missing answers pass forward: at the screen stage a false reject
 * costs months of lost value while a false accept costs one interview slot, so
 * the criterion is deliberately set for recall.
 */
export function evaluateGate(questions: GateQuestion[], answers: GateAnswer[]): GateResult {
  const byId = new Map(questions.map((q) => [q.id, q]));
  const disqualifiers: Disqualifier[] = [];

  for (const answer of answers) {
    if (answer.verdict !== 'FAIL') continue;
    const question = byId.get(answer.id);
    if (!question) continue;
    if (question.severity < BLOCKING_SEVERITY) continue;
    disqualifiers.push({
      id: question.id,
      question: question.question,
      severity: question.severity,
      evidence: answer.evidence,
    });
  }

  return { passed: disqualifiers.length === 0, answers, disqualifiers };
}
