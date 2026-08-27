import type { Emphasis, MatchEvaluation } from './schemas.js';

export interface CriterionWeight {
  key: keyof MatchEvaluation['scores'];
  name: string;
  weightKey: keyof Emphasis;
}

export const CRITERIA: CriterionWeight[] = [
  { key: 'language_proficiency', name: 'Language Proficiency', weightKey: 'language_proficiency_weight' },
  { key: 'education_level', name: 'Education Level', weightKey: 'education_weight' },
  { key: 'experience_years', name: 'Years of Experience', weightKey: 'experience_weight' },
  { key: 'technical_skills', name: 'Technical Skills', weightKey: 'technical_skills_weight' },
  { key: 'certifications', name: 'Certifications', weightKey: 'certifications_weight' },
  { key: 'soft_skills', name: 'Soft Skills', weightKey: 'soft_skills_weight' },
  { key: 'location', name: 'Location', weightKey: 'location_weight' },
];
