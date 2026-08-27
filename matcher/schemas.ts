import { z } from 'zod';

const int0to100 = z.number().int().min(0).max(100);

export const EmphasisSchema = z.object({
  technical_skills_weight: z.number().int(),
  soft_skills_weight: z.number().int(),
  experience_weight: z.number().int(),
  education_weight: z.number().int(),
  language_proficiency_weight: z.number().int(),
  certifications_weight: z.number().int(),
  location_weight: z.number().int(),
});
export type Emphasis = z.infer<typeof EmphasisSchema>;

export const DEFAULT_EMPHASIS: Emphasis = {
  technical_skills_weight: 50,
  soft_skills_weight: 20,
  experience_weight: 20,
  education_weight: 10,
  language_proficiency_weight: 5,
  certifications_weight: 5,
  location_weight: 50,
};

export const JobRequirementsSchema = z.object({
  required_experience_years: z.number().int(),
  required_education_level: z.string(),
  required_skills: z.array(z.string()),
  optional_skills: z.array(z.string()),
  certifications_preferred: z.array(z.string()),
  soft_skills: z.array(z.string()),
  keywords_to_match: z.array(z.string()),
  location: z.object({ country: z.string(), city: z.string() }),
  emphasis: EmphasisSchema,
});
export type JobRequirements = z.infer<typeof JobRequirementsSchema>;

export const MatchEvaluationSchema = z.object({
  scores: z.object({
    language_proficiency: int0to100,
    education_level: int0to100,
    experience_years: int0to100,
    technical_skills: int0to100,
    certifications: int0to100,
    soft_skills: int0to100,
    location: int0to100,
  }),
  match_reasons: z
    .array(z.string())
    .describe('3-4 key match reasons, telegraphic English, max 10 words each'),
  website: z
    .string()
    .describe("Candidate's personal website URL from the resume, or empty string"),
});
export type MatchEvaluation = z.infer<typeof MatchEvaluationSchema>;

export const EmailSchema = z.object({
  subject: z.string(),
  body: z.string(),
});
export type Email = z.infer<typeof EmailSchema>;

export const JDRankingSchema = z.object({
  scores: z.object({
    language_proficiency: int0to100,
    education_level: int0to100,
    experience_years: int0to100,
    technical_skills: int0to100,
    certifications: int0to100,
    soft_skills: int0to100,
    location: int0to100,
  }),
  improvement_tips: z
    .array(z.string())
    .describe('3-5 concrete tips to improve the job description'),
});
export type JDRanking = z.infer<typeof JDRankingSchema>;

export const OverallAnalysisSchema = z.object({
  analysis: z.string().describe('Brief overall analysis of the candidate pool vs the job'),
  suggestions: z
    .array(z.string())
    .describe('3-5 actionable suggestions to attract better-matching candidates'),
});
export type OverallAnalysis = z.infer<typeof OverallAnalysisSchema>;

export const GateQuestionsSchema = z.object({
  questions: z
    .array(
      z.object({
        id: z.string().describe('kebab-case slug, e.g. work-authorization'),
        question: z
          .string()
          .describe('Binary question, phrased so that YES means the candidate is acceptable'),
        severity: z
          .number()
          .int()
          .min(1)
          .max(10)
          .describe(
            'FMEA severity. 10 = impossible to hire. 7-9 = stated hard must. 4-6 = strong preference. 1-3 = nice to have.',
          ),
        why: z.string().describe('Short quote from the job description that justifies this gate'),
      }),
    )
    .describe('2-5 hard-constraint questions. Never graded skills.'),
});
export type GateQuestions = z.infer<typeof GateQuestionsSchema>;

export const GateAnswersSchema = z.object({
  answers: z.array(
    z.object({
      id: z.string().describe('Must match one of the supplied question ids exactly'),
      verdict: z.enum(['PASS', 'FAIL', 'UNCERTAIN']),
      evidence: z.string().describe('Exact quote from the resume, or "not stated"'),
    }),
  ),
});
export type GateAnswers = z.infer<typeof GateAnswersSchema>;
