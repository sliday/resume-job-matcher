# Resume Job Matcher

## Overview

**Resume Job Matcher** screens a folder of resume PDFs against a job description and tells you who to interview. It runs as a TypeScript CLI (`matcher/`) on Claude, GPT, or OpenRouter. A legacy Python implementation is still included for OCR and PDF regeneration.

Screening is a funnel, cheapest stage first:

1. **Pre-filter** (optional, `--prefilter`): embed the job and every resume, keep the closest by cosine similarity. No LLM call per candidate, and the cache is keyed on resume content, so a candidate is embedded once across every job.
2. **Gate**: hard constraints from the job description (work authorization, on-site requirement, licences) become pass/fail questions with a severity. Fail a blocking one and you are rejected with the resume span that disqualified you, and never reach the scoring call. Silence is never treated as failure.
3. **Score**: survivors are rated per criterion on an anchored 0-4 scale, and every rating quotes the evidence behind it.

The point of the split: a candidate who cannot legally take the job is not "a 62% match", and a score you cannot trace back to a line of the resume is not worth having.

![Area](https://github.com/user-attachments/assets/1fee4382-7462-4463-9cb1-61704eea218b)

## Quick start

Requires Node 20 or newer (tested on 22). No Python needed.

```bash
npm install
cp .env-example .env              # add ONE key, for the mode you plan to use
mkdir -p resumes                  # drop your candidate PDFs in here
npm run match -- "EXAMPLE job_description.txt" resumes --api openrouter
```

Both positionals are optional, but their defaults (`job_description.txt` and `src/`) do not exist in a fresh clone, so pass them explicitly the first time.

Modes (`--api`):

- `openrouter` (default): routes through [openrouter/auto](https://openrouter.ai/openrouter/auto), needs `OPENROUTER_API_KEY`
- `anthropic`: Claude, needs `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`
- `openai`: GPT, needs `OPENAI_API_KEY`

One key is enough. Override the model per mode with `ANTHROPIC_MODEL`, `OPENAI_MODEL`, or `OPENROUTER_MODEL`.

## Options

| Flag | Effect |
|---|---|
| `--prefilter <n>` | Keep only the `n` resumes closest to the job before scoring. Default 0 (off). See below. |
| `--concurrency <n>` | Parallel evaluations. Default 4. |
| `--threshold <n>` | Score at or above which the generated email invites rather than rejects. Default 90. |
| `--no-email` | Skip email generation. Faster and cheaper. |
| `--analyze-jd` | Grade the job description itself and write an improved version to `job_description_enhanced.txt`. |
| `--unify` | Normalize each resume to Markdown before scoring, and save it to `out/`. |
| `--no-analysis` | Skip the pool-level summary. |

## How scoring works

Seven criteria are rated: language proficiency, education, years of experience, technical skills, certifications, soft skills, and location.

Each is rated on an anchored scale rather than a bare 0-100, and the model must quote the deciding resume span **before** it rates:

| Level | Meaning | Score |
|---|---|---|
| 4 | Direct, specific, verifiable evidence that meets or exceeds the requirement | 100 |
| 3 | Clear evidence of a close match, minor gap | 75 |
| 2 | Partial or adjacent evidence, claimed but not substantiated | 50 |
| 1 | Weak or tangential evidence only | 25 |
| 0 | Absent from the resume, or contradicted by it | 0 |

The final score is the weighted mean of those seven, with weights extracted from the job description.

A criterion at level 0 raises a red flag sized by that criterion's weight: 🚩 for weight 30 or more, 📍 for 20 or more, ⛳ below that.

## Large candidate pools (`--prefilter`)

Scoring every resume with an LLM is linear in pool size, and screening many jobs against many candidates is a cross product no per-call tuning survives. `--prefilter <n>` adds a cheap first cut: embed the job and every resume, keep the `n` closest by cosine similarity, and run the expensive gate-and-score pass only on those.

```bash
npm run match -- job.txt resumes --api openai --prefilter 50
```

Embeddings are cached in `out/.embedding-cache.json`, keyed on **resume content rather than filename**, so a candidate is embedded once no matter how many jobs you screen them against. That is the difference between `jobs × candidates` embedding calls and just `candidates`.

Needs `--api openai` or `--api openrouter`. Anthropic has no embeddings API.

A real run over 5 resumes with `--prefilter 3`: 2 skipped before any LLM call, 2 rejected by the gate on work location, 1 scored. Six LLM calls, $0.0025 total.

## Output

Results print ranked, with gated-out and pre-filtered candidates listed separately so nothing disappears quietly.

Written to `out/`:

- `<name>_response.txt`: generated candidate email (invite or rejection, per `--threshold`)
- `<name>_unified.md`: normalized resume, when `--unify` is set
- `.embedding-cache.json`: embedding cache, when `--prefilter` is used

Every run ends with a token and cost line. `--analyze-jd` also writes `job_description_enhanced.txt`.

![CleanShot 2024-10-09 at 17 08 09@2x](https://github.com/user-attachments/assets/e47b57e1-521a-4b21-aeb3-975af1e0f2ed)

## Known limitations

Worth knowing before you trust a number:

- **Scores are compensatory.** Strength on one criterion offsets weakness on another. Hard constraints are the gate's job precisely because the score cannot express them.
- **Cross-model score variance is significant.** The same resume against the same job has been measured 20+ points apart across models. Treat scores as bands, not as measurements, and treat the gate decision as the reliable signal.
- **The pre-filter is topical only.** It ranks by subject-matter closeness and knows nothing about constraints, so it can rank a disqualified candidate first. It can also drop a strong unconventional candidate whose vocabulary differs from the job description. Set `n` generously.
- **No OCR.** The TypeScript build reads embedded PDF text only. Scanned resumes fail with a short-text error. Use the Python implementation for those.

Ongoing work against these is tracked in `LOOP.md`.

## Troubleshooting

- **No PDF files found**: pass the resume folder explicitly. The default is `src/`, which does not exist in a fresh clone.
- **Job description file not found**: pass the path explicitly. The default is `job_description.txt`.
- **Missing key error**: the CLI names the exact variable it wants for your `--api` mode. Only that one is needed.
- **PDF text extraction too short**: the PDF is a scan with no embedded text. See OCR above.
- **`--prefilter` refuses to run**: you are on `--api anthropic`, which has no embeddings. Use `openai` or `openrouter`.

## Development

```bash
npm run typecheck     # tsc --noEmit
npm test              # unit tests (node:test)
npm run eval          # fixture evals against a live model, costs cents
```

`npm run eval` is the definition of green: five fixtures covering match band, red-flag firing, prompt-injection resistance, website extraction, and email personalization.

## Legacy Python implementation

`resume_matcher.py` predates the TypeScript rewrite. It is kept for two things the TS build does not do: OCR of scanned PDFs, and regenerating unified resumes as styled PDFs. It is scheduled for retirement once OCR lands in TypeScript.

```bash
pip install PyPDF2 anthropic openai tqdm termcolor json5 requests beautifulsoup4 pydantic
python resume_matcher.py [--sans-serif|--serif|--mono] [--pdf] [job_desc_file] [pdf_folder]
```

It has a different scoring model from the TypeScript path, so its numbers are not comparable.

## Contributing

Fork, branch, open a pull request. Please run `npm run typecheck` and `npm test` first; `npm run eval` too if you touched matching behaviour.

## Data handling

Resumes are personal data. Candidate text is sent to whichever model provider you configure. Handle it in line with the data protection rules that apply to you, and check your provider's retention policy before screening real applicants.
