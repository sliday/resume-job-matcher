export interface ScoreRange {
  min: number;
  max: number;
  label: string;
  color: string;
  emoji: string;
}

export const SCORE_RANGES: ScoreRange[] = [
  { min: 100, max: 101, label: 'Legendary Unicorn', color: 'magenta', emoji: '🦄' },
  { min: 99, max: 100, label: 'Dream Candidate', color: 'yellow', emoji: '🏆' },
  { min: 98, max: 99, label: 'Exceptional Fit', color: 'magenta', emoji: '🥇' },
  { min: 97, max: 98, label: 'Outstanding Candidate', color: 'magenta', emoji: '🥈' },
  { min: 96, max: 97, label: 'Superb Applicant', color: 'magenta', emoji: '🥉' },
  { min: 95, max: 96, label: 'Excellent Choice', color: 'magenta', emoji: '🌟' },
  { min: 94, max: 95, label: 'Top Prospect', color: 'blue', emoji: '💫' },
  { min: 93, max: 94, label: 'Strong Contender', color: 'blue', emoji: '🌠' },
  { min: 92, max: 93, label: 'Impressive Talent', color: 'blue', emoji: '✨' },
  { min: 91, max: 92, label: 'Highly Qualified', color: 'cyan', emoji: '🌊' },
  { min: 90, max: 91, label: 'Great Potential', color: 'cyan', emoji: '💎' },
  { min: 88, max: 90, label: 'Very Promising', color: 'cyan', emoji: '💎' },
  { min: 86, max: 88, label: 'Solid Candidate', color: 'green', emoji: '🍀' },
  { min: 84, max: 86, label: 'Good Fit', color: 'green', emoji: '🌿' },
  { min: 82, max: 84, label: 'Suitable Match', color: 'green', emoji: '🌴' },
  { min: 80, max: 82, label: 'Potential Hire', color: 'green', emoji: '🌱' },
  { min: 78, max: 80, label: 'Possible Fit', color: 'green', emoji: '🥑' },
  { min: 76, max: 78, label: 'Fair Prospect', color: 'green', emoji: '🥝' },
  { min: 74, max: 76, label: 'Moderate Match', color: 'green', emoji: '🥦' },
  { min: 72, max: 74, label: 'Average Candidate', color: 'yellow', emoji: '🌻' },
  { min: 70, max: 72, label: 'Partial Fit', color: 'yellow', emoji: '🌼' },
  { min: 68, max: 70, label: 'Limited Potential', color: 'yellow', emoji: '🌟' },
  { min: 66, max: 68, label: 'Weak Match', color: 'yellow', emoji: '🍋' },
  { min: 64, max: 66, label: 'Minimal Alignment', color: 'yellow', emoji: '🍌' },
  { min: 62, max: 64, label: 'Low Compatibility', color: 'yellow', emoji: '🧀' },
  { min: 60, max: 62, label: 'Needs Improvement', color: 'yellow', emoji: '🌽' },
  { min: 58, max: 60, label: 'Considerable Gap', color: 'yellow', emoji: '🍯' },
  { min: 56, max: 58, label: 'Poor Fit', color: 'yellow', emoji: '🍍' },
  { min: 54, max: 56, label: 'Significant Mismatch', color: 'yellow', emoji: '🍈' },
  { min: 52, max: 54, label: 'Major Differences', color: 'yellow', emoji: '🍏' },
  { min: 50, max: 52, label: 'Substantial Gap', color: 'yellow', emoji: '🐤' },
  { min: 45, max: 50, label: 'Unqualified Candidate', color: 'yellow', emoji: '🍊' },
  { min: 40, max: 45, label: 'Mismatched Skills', color: 'yellow', emoji: '🥕' },
  { min: 35, max: 40, label: 'Inadequate Fit', color: 'yellow', emoji: '🦊' },
  { min: 30, max: 35, label: 'Unsuitable Applicant', color: 'red', emoji: '🍎' },
  { min: 25, max: 30, label: 'Incompatible Match', color: 'red', emoji: '🍓' },
  { min: 20, max: 25, label: 'Irrelevant Background', color: 'red', emoji: '🍒' },
  { min: 15, max: 20, label: 'Completely Misaligned', color: 'red', emoji: '🍅' },
  { min: 10, max: 15, label: 'Wrong Field', color: 'red', emoji: '🌶️' },
  { min: 5, max: 10, label: 'Possibly Unsuitable', color: 'gray', emoji: '🎱' },
  { min: 0, max: 5, label: 'No Match', color: 'gray', emoji: '🕷️' },
];

export function getScoreDetails(score: number): { emoji: string; color: string; label: string } {
  for (const range of SCORE_RANGES) {
    if (score >= range.min && score < range.max) {
      return { emoji: range.emoji, color: range.color, label: range.label };
    }
  }
  return { emoji: '💀', color: 'red', label: 'Unable to score' };
}

