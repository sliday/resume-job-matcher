# Cognitive Matcher - redesign of the matching mechanism

Date: 2026-08-27
Status: proposal, awaiting approval
Scope: `matcher/match.ts`, `matcher/schemas.ts`, `evals/`

---

## 1. What exists now

One sentence: **a compensatory weighted-linear model over 7 HR categories, with LLM-hallucinated
weights, anchorless 0-100 magnitude estimates, and decorative red flags.**

```
score = Σ (criterion_score × weight) / Σ weight
```

Seven criteria: language, education, experience_years, technical_skills, certifications,
soft_skills, location. Weights come from `extractJobRequirements` - the model invents them per job.
Red flags fire when a criterion scores < 10, but they only push a string into a list. The weighted
sum still runs. Nothing gates.

### The proof that this is broken

Look at the eval fixture `location-hard-miss`. Candidate in Brazil. No German. Will not relocate.
Job says *"Must be located in Germany. On-site in Berlin 4 days/week."*

The fixture asserts `scoreMax: 80`.

Eighty. For a candidate who cannot take the job. The fixture author could not write
`decision: NO`, because the model has no way to say it. A hard constraint got laundered into a
soft penalty. That is the whole defect in one line.

### Five defects, named

| # | Defect | Consequence |
|---|---|---|
| D1 | Compensatory aggregation | Disqualifiers become discounts. Brazil candidate scores 80. |
| D2 | Hallucinated fine-grained weights | `location_weight` default 50 = same as technical skills. Nobody chose that. |
| D3 | Anchorless magnitude estimation | "Score 0-100" with no reference. Source of the 53-vs-63 jitter in LOOP.md Known-flaky. |
| D4 | No comparison between candidates | Hiring is picking the best *available*. System scores each in a vacuum. |
| D5 | 41 output bands on a ±10 signal | `SCORE_RANGES` reports 4x finer than the instrument resolves. Instrumentation error. |

D3 is worth dwelling on. The current fix for jitter was to widen the band, 60 → 68. That treats
variance as noise to tolerate. Variance is signal about the *method*, and it is also a resource
(see §2.7).

---

## 2. The question: how does an expert brain actually do this?

The naive answer - "be more holistic, more intuitive" - is the wrong answer, and the literature
is unusually blunt about it.

Meehl (1954) and the Grove & Meehl (1996) meta-analysis of 136 studies: **mechanical combination
beats clinical/holistic judgment**, or ties it, in almost every domain tested. Human experts lose
to a simple formula. So: do not copy the expert's aggregation.

But that same literature says something more useful. Humans are **good at observing and coding
individual cues** and **bad at integrating them**. The expert's value is in noticing, not in summing.

So the design rule is: **copy the brain's decomposition and its attention economy. Replace its
arithmetic.**

Eight mechanisms, and what each one says to do.

### 2.1 Structured decomposition, delayed gestalt - Kahneman

Kahneman's own hiring procedure (Israeli army, retold in *Thinking, Fast and Slow*): pick ~6
independent traits. Score each on factual questions. **Forbid yourself from forming a global
impression until all six are scored.** Then sum. It beat expert intuition badly.

Then the twist he added: *after* the six scores, allow the holistic judgment. It carries real
information - but only once it cannot contaminate the evidence coding.

→ **Score dimensions blind. Then take one gestalt. Keep both. Never let the gestalt run first.**

### 2.2 Recognition-primed decision - Klein

Naturalistic decision making. Fire chiefs, nurses, chess masters do not compare options. They
pattern-match the situation to a **prototype**, retrieve the typical action, and **mentally
simulate** it.

An expert recruiter does not think "technical_skills: 72". They think *"platform-infra person who
drifted into ML tooling"* - an archetype label. Then they simulate: *would this person survive our
on-call rotation? would they be bored in six months?*

→ **Emit an archetype label. Run a 90-day simulation. Run a pre-mortem** (Klein's own technique:
assume it failed, explain why).

### 2.3 Predictive processing - Friston, Clark

The brain is a prediction machine. It does not score inputs. It generates predictions and encodes
**prediction error**. Attention flows to surprise. A signal matching the model carries near-zero
information.

A resume that is exactly what you would expect for the role tells you almost nothing. **The
information is in the deviations.** A staff engineer with no public artifacts: surprising. Five
jobs in three years: surprising. A PhD applying junior: surprising. Both good and bad surprise
outweigh any confirmed expectation.

→ **Build an expected-candidate model from the JD once. Diff each resume against it. Score the
surprise, not the conformity.** This is the highest-leverage addition and nothing like it exists
today. It is also the mechanism that rescues the great unconventional candidate a linear model
ranks 62.

### 2.4 Somatic markers - Damasio

Damasio's vmPFC patients keep full reasoning and lose the ability to decide. They enumerate
forever. Emotion is not noise on reason; it is the pruning function. The "gut feeling" is a cached
summary of past outcomes.

The current system has no feedback path. Nothing learns from hires that worked. It is a
vmPFC-lesioned system: it will score forever and never improve.

→ **A calibration set of labeled past decisions is the somatic marker.** Even 10-20 rows
(`interviewed / hired / flamed out / passed and regretted it`) converts static scoring into
learning. Few-shot anchors first; fitted weights only if the set grows past ~50.

### 2.5 Fast-and-frugal heuristics - Gigerenzer; and improper linear models - Dawes

Against "heuristics = bias": simple heuristics often beat regression **out of sample**, because
they do not overfit. *Take-the-best*: order cues by validity, use the first that discriminates,
stop. *Fast-and-frugal trees*: 3-5 binary questions, each with an exit.

Dawes (1979), "The robust beauty of improper linear models": **unit weights beat expert judgment,
and frequently beat optimally-fitted regression weights out of sample**, in small-sample noisy
domains. Resume screening is exactly that domain.

This is a direct hit on D2. The system spends an LLM call inventing precise weights. Precise
weights are *worse* than equal weights when you have no training data to fit them on.

→ **Kill per-job weight hallucination. Use unit weights, or three coarse levels
(critical / important / nice-to-have) a human sets once.**

### 2.6 Signal detection theory

Screening is a decision under uncertainty with asymmetric costs, not a scoring exercise.

- False accept: one wasted interview slot. ~2-6 person-hours.
- False reject: a good hire lost, silently, forever. Cost measured in months of value.

At the screen stage, base rate of qualified is low and the asymmetry favours **recall**. The
current design sets d′ and the criterion invisibly, inside a scalar that gets thresholded
somewhere downstream.

→ **Output an explicit decision plus an explicit confidence. Separate "how good" from "how sure".
Bias UNCERTAIN toward passing at the screen stage.**

### 2.7 The crowd within - Vul & Pashler; dialectical bootstrapping - Herzog & Hertwig

Ask one person the same estimate twice; average them; accuracy improves measurably. Independent
noisy estimates cancel. Herzog & Hertwig get more by making the second estimate *deliberately
adversarial* to the first.

This reframes the 53-vs-63 flakiness. It is not noise to widen bands around. It is **an
unexploited ensemble**.

→ **Sample the final judgment 2-3 times with different stances - skeptic, advocate, neutral.
Take the median. Disagreement between stances IS the confidence signal.** Wide spread = escalate
to a human. That is strictly better information than a widened assertion band.

### 2.8 Weber-Fechner / relative judgment

Basic psychophysics: humans discriminate *differences* far better than they estimate *magnitudes*.
Models inherit this. "Is A better than B?" is a much more reliable question than "rate A out of
100".

→ **Rank the shortlist by pairwise comparison (Bradley-Terry over ~O(n log n) pairs), not by
sorting absolute scores.**

---

## 3. The engineering side

### 3.1 TRIZ - the contradiction

Every hard problem is a contradiction between two wanted properties. Here:

> Want **high recall** (never miss the unconventional great candidate)
> AND **high precision** (never waste an interview slot).

A single linear score resolves this badly - it collapses both onto one axis and satisfies neither.
TRIZ separation principles apply directly:

- **Separate in time** → two stages. Cheap high-recall gate, then expensive high-precision rank.
- **Separate in condition** → different logic for must-haves (boolean) and differentiators (graded).

### 3.2 FMEA

A red flag is not "score < 10". It is a failure mode with a **severity**, an **occurrence**, and a
**detectability**.

- Missing work authorization: severity 10, non-recoverable, blocks.
- Light on Kubernetes: severity 3, trainable, does not block.

The current code assigns severity from the *weight* the model happened to invent. Backwards.

### 3.3 Axiomatic design - Suh

**Axiom 1, independence.** Keep functional requirements independent. Violated: default
`location_weight: 50` silently dominates the technical signal. One parameter, two effects.

**Axiom 2, information.** Prefer the design with least information content that meets the
requirements. Violated hard: 41 labels (`Legendary Unicorn`, `Substantial Gap`, `Wrong Field`)
over a signal whose reproducibility is ±10 points. Delete all but 5.

### 3.4 Failure-first - Petroski

*To Engineer Is Human*: designs advance by studying failures, not successes. The analogue:
define what a **bad hire for this specific role** looks like, and check for that, instead of only
checking for good.

---

## 4. Proposed architecture - five passes, funnel-shaped

Mirrors the brain's own attention economy: cheap parallel filtering, expensive serial attention
on the few survivors.

```
  all resumes
      │
      ▼
 [P0] Perception        deterministic parse. 0 LLM calls.
      │                 years, titles, tech tokens, locations, gaps, tenure pattern
      ▼
 [P1] Gate              fast-and-frugal tree. 1 cheap call, tiny output.
      │                 3-5 binary must-pass, each severity-rated.
      │                 PASS / FAIL(reason) / UNCERTAIN → UNCERTAIN passes (recall bias)
      │
      ├──── FAIL ──────► decision card, exit, ~0 further cost
      ▼
 [P2] Evidence          BARS-anchored coding. evidence quoted BEFORE the rating.
 [P3] Surprise          diff vs expected-candidate model. positive + negative.
 [P4] Simulation        90-day sim + pre-mortem.
      │                 P2+P3+P4 = 1 structured call
      ▼
 [P5] Adjudication      2-3 stances (skeptic / advocate / neutral). median + spread.
      │                 spread = confidence. wide spread → flag for human.
      ▼
  shortlist
      │
      ▼
 [P6] Pairwise rank     Bradley-Terry, ~O(n log n) tiny comparisons. shortlist only.
```

### 4.1 Pass 0 - Perception (V1 feature extraction)

Deterministic where possible. No judgment. Folds into the existing `unifyResume` step.
Output: structured candidate object.

### 4.2 Pass 1 - Gate (amygdala short-circuit)

3-5 binary questions derived from the JD once, cached. Each carries an FMEA severity.
Not scored. **Decided.**

```
Q: Can this person legally work at the required location?  severity 10, blocking
Q: Do they hold the single non-negotiable skill?           severity 8,  blocking
Q: Does stated availability match the role?                severity 6,  blocking
```

`FAIL` on any blocking question → exit with reason. Score not reported (a score on a
disqualified candidate is misinformation - that is the Brazil-80 bug).
`UNCERTAIN` → forward. Recall bias at the screen stage, per §2.6.

Runs on a small model. Kills most of the pool for near-nothing.

### 4.3 Pass 2 - Evidence coding (Kahneman's six traits, blind)

Two changes from today, both load-bearing.

**(a) Behaviorally Anchored Rating Scales instead of naked 0-100.** 4-5 levels, each with a
concrete description. This is the I/O-psych answer to anchorless scales and the direct fix for D3.

```
Task-domain evidence:
 4 - shipped this exact work, named artifacts, quantified outcome
 3 - shipped adjacent work, named artifacts
 2 - claims the skill, no artifact
 1 - skill absent or inferred only from a keyword list
 0 - contradicted by the record
```

**(b) Evidence before number.** The schema forces a quoted resume span first, rating second.
Kills hallucinated evidence and anchorless magnitude in one move.

**Dimensions - replace the HR taxonomy.** The predictive-validity literature (Schmidt & Hunter;
magnitudes revised downward by Sackett et al. 2022, but the *ordering* holds) puts work samples
and structured behavioural evidence near the top, and puts years-of-experience and
education near the bottom. The current system weights the weak predictors heavily.

| Old (HR taxonomy) | New (predictive) |
|---|---|
| technical_skills | **task-domain evidence** - has done the actual work, with artifacts |
| experience_years | **scope trajectory** - what size of problem owned, and is the arc rising |
| certifications | **recency** - used in last 2 years vs 8 years ago |
| education | **evidence quality** - quantified specifics vs generic claims (proxy for self-awareness and honesty) |
| soft_skills | **role-shape fit** - IC-depth person for an IC-depth job |
| language, location | → moved to the Gate, where they belong (they are constraints, not scores) |

Five dimensions. Unit weights, or coarse 3-level.

### 4.4 Pass 3 - Surprise (prediction error)

Derive the *expected* candidate from the JD once, cache it. Diff each resume against it.

- **Positive surprise** - unexpected strength a keyword filter would miss.
- **Negative surprise** - gap the category scores hide.

New capability. No equivalent today.

### 4.5 Pass 4 - Simulation

*"Describe this person's first 90 days. Where do they struggle?"*
*"Pre-mortem: it is six months later and this hire failed. Why?"*

Surfaces risk that scoring structurally cannot.

### 4.6 Pass 5 - Delayed holistic + adjudication

Only now, with all evidence coded, allow the gestalt. One 5-point decision:

`STRONG_NO / NO / MAYBE / YES / STRONG_YES`

Run 2-3 stances. Median = decision. Spread = confidence. Split panel → `needs_human: true`.

### 4.7 Pass 6 - Pairwise rank

Shortlist only. Bradley-Terry over ~O(n log n) pairwise comparisons. Answers the question the
user actually has - *who do I interview first* - which absolute scores answer badly.

### 4.8 Output - a decision card, not a number

```
decision:      YES
confidence:    0.82          (from stance agreement × evidence density)
rank_in_pool:  3 of 47
for:           3 items, each with a resume citation
against:       3 items, each with a resume citation
disqualifiers: []            (or [{q, severity, reason}])
surprise:      "+ built the payments reconciliation the JD is silent about"
premortem:     "fails if the role is 70% stakeholder work"
needs_human:   false
```

Score, if kept at all, ships as an **interval** (`64 ± 8`), never a point value with a
`Legendary Unicorn` label attached.

---

## 5. Cost

| | now | proposed |
|---|---|---|
| per resume, screened out | 3 calls | 1 tiny call |
| per resume, survivor | 3 calls | ~4 calls (1 tiny + 1 large + 2 tiny) |
| shortlist rank | 0 | ~n log n tiny calls |

Pool of 100 at 30% gate pass: **100 tiny + 30×4 ≈ 1.4x current spend**, and it scales *better*
than linear as pools grow, because bad candidates get cheaper, not more expensive. Funnel shape =
the brain's attention economy.

Existing token-cost instrumentation (`getTokenUsage`, `estimateCostUsd`) already measures this.

---

## 6. Three ways to land it

### Option A - Gate + Anchors (small, ~1 loop iteration)

Ship P1 (real gate) and P2a (BARS anchors + evidence-before-number). Keep everything else.

- Fixes D1, D3. Partially D2.
- `location-hard-miss` becomes `decision: NO, disqualifier: work authorization` instead of 80.
- Variance should drop, which makes the deferred calibration fixture in LOOP.md affordable.
- Cost: roughly flat.
- Risk: low. Two prompt+schema changes. Existing evals still meaningful.

### Option B - Full cognitive pipeline (large, ~4-6 iterations)

All six passes plus the decision card plus pairwise ranking.

- Fixes everything. Genuinely new capability in surprise and pre-mortem.
- Cost: ~1.4x, and every existing eval assertion needs rewriting (score bands → decisions).
- Risk: high. Big-bang rewrite against an eval suite that would be rewritten in the same commit.
  That is how you lose the signal the harness exists to give you.

### Option C - Staged, gate-first (recommended)

Option A first, as one loop item. Then each remaining pass as its own item, each landing green:

1. **Gate + FMEA severities** → `location-hard-miss` asserts a decision, not a band
2. **BARS anchors + evidence-before-number** → unblocks the calibration fixture
3. **Kill hallucinated weights → 3-level coarse** → Dawes; removes a failure mode
4. **Collapse 41 bands → 5; report interval** → instrumentation honesty, trivial
5. **Surprise pass** → the new capability
6. **Multi-stance adjudication** → converts variance into confidence
7. **Pairwise shortlist rank** → answers the real question
8. **Calibration corpus** → the somatic marker, once labeled data exists

**Recommend C.** It matches the LOOP.md discipline already in the repo - one verifiable item at a
time, eval green at every step. B is the same destination with the safety rail removed.

Items 1-4 are the ones that fix present bugs. Items 5-8 are the ones that make it better than a
human screener rather than a faster one.

---

## 7. What each eval fixture becomes

| fixture | now | after |
|---|---|---|
| strong-frontend-match | `scoreMin: 60` | `decision ∈ {YES, STRONG_YES}` |
| backend-vs-frontend-mismatch | `scoreMax: 68` (widened once already) | `decision ∈ {NO, MAYBE}` - band jitter stops mattering |
| location-hard-miss | `scoreMax: 80` <- the bug | `decision: NO`, `disqualifier: location/authorization` |
| prompt-injection-resistance | unchanged | unchanged |
| website / email | unchanged | unchanged |

The Known-flaky entry in LOOP.md dissolves. `53 vs 63` both map to `NO`. Discretising at the
decision layer instead of the score layer is what kills the flake - not a wider band.

---

## 8. Open questions

1. **Calibration data.** Does any labeled history exist (past screens with outcomes)? Item 8 is
   the highest long-term value and it is blocked without ~10-20 labeled rows. If none exists, the
   loop should start *collecting* now, even by hand.
2. **Who sets the coarse weights?** Item 3 replaces model-invented weights with human-set
   critical/important/nice-to-have. That needs a place to live - JD frontmatter, a sidecar file,
   or a CLI flag.
3. **Does the score survive at all?** A decision + confidence + rank may make the 0-100 number
   dead weight. Keeping it as an interval is the conservative call; deleting it is the honest one.
