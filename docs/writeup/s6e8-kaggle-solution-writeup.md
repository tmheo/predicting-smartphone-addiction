# Kaggle S6E8 solution write-up (English source)

This file is the source for the Kaggle `New Solution Writeup` form.
The first section holds the form fields.
Everything under `Content` is pasted into the Content editor.
Facts come from the Korean retrospective, `docs/research/s6e8-our-final-solution.md` and the technical evidence appendix that lived at `docs/presentation/s6e8-retrospective-technical-evidence.md` until commit `cdcd677`.

## Form fields

| Field | Value |
| --- | --- |
| Title (80 max) | 14th Place Solution - Gated Stacking of 36 Own Models and 278 Public OOF Sets |
| Subtitle (140 max) | Fixed 5-fold, one sealed fold for every ensemble decision, rank-logit logistic stacking over 314 columns. Private 0.97109. |
| Tags (2-5) | tabular, ensembling, binary classification, neural networks, gradient boosting |
| Project links | GitHub repository `https://github.com/tmheo/predicting-smartphone-addiction` |
| Project links | Final solution reconstruction `https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/research/s6e8-our-final-solution.md` |
| Project links | Experiment adoption contract (ADR-0001) `https://github.com/tmheo/predicting-smartphone-addiction/blob/main/docs/adr/0001-experiment-adoption-contract.md` |
| Card and thumbnail (560 x 280) | `assets/writeup-thumbnail.png` (to be made in #609) |
| Media gallery | `assets/writeup-score-progression.png`, `assets/writeup-nested-oof.png` |

Figures referenced below do not exist yet.
They are English versions of the Korean retrospective figures and are produced in #609.

## Content

Congratulations to the winners, and thanks to everyone who shared OOF predictions and notebooks during the competition.
A large part of my final score came from your work, so this write-up spends as much time on how I validated and combined public predictions as on my own models.

I finished 14th with Private 0.97109 (Public 0.97135).
The final submission is a logistic stack over 314 prediction columns: 36 of my own models refit on full train, plus 278 public OOF/test pairs that passed an integrity check.
The part I care most about is not the stack itself but the rule that every ensemble decision had to survive a sealed fold before it was adopted.

![Score progression from the first LightGBM baseline to the final stack](assets/writeup-score-progression.png)

## TL;DR

- Fixed `StratifiedKFold(5, shuffle=True, random_state=42)` saved to a file on day one, never regenerated.
- Every single model is a 3-seed average (42, 43, 44) on those folds.
- Biggest single-model gains: keeping numeric columns and adding exact-value categorical copies (+0.00329 on LightGBM), and fixing a float32/float64 dtype bug in my RealMLP port (+0.00461).
- Best own single model: a Lookup-Transformer variant at OOF AUC 0.96941.
- Ensemble decisions were scored with nested OOF: seal one fold, choose members and combiner settings on the other four, score the sealed fold, repeat five times.
- Adoption gate for a pool change: nested OOF delta above a pre-set threshold and the sealed-fold delta positive in at least 3/5 (later 5/5) folds.
- Final combiner: L2 logistic regression on standardized rank and logit of every column, with `C` and a shrinkage factor toward the plain rank mean chosen inside the outer training folds.
- Nested OOF: own 35 models 0.96981, plus public predictions 0.97029, after removing 120 harmful public columns 0.97035, final 314 columns 0.97038.
- Public LB was never used for any decision.

## Data

Train has 691,369 rows, test 296,302, with 9 numeric and 3 categorical features and about 70.9% positives.
Every one of the 12 columns has missing values.

Two things shaped the whole approach.

First, the numeric columns are not continuous.
Values repeat on a small set of exact numbers, and the target rate is stable within each exact value.
I read this as a trace of how the synthetic data was generated, not as a fact about people, and treated exact values as keys.

Second, there is a budget relation between the screen-time columns.
`daily_screen_time_hours` is roughly `social_media_hours + gaming_hours + work_study_hours + something else`.
The residual of that relation, and the ratio of each part to the total, turned out to be useful features and also gave a way to bound imputed values.

## Validation

### Fixed folds and OOF

Folds were generated once with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, written to `artifacts/folds.parquet` and committed.
Every experiment reads that file.
Anything that touches the target (target encoding, imputation models, vocabularies for the transformers, rank-gauss quantiles) is fit inside the training part of each fold.
All numbers below are OOF AUC on those folds unless stated otherwise, averaged over seeds 42, 43, 44.

Two more rules ran from the first week.

- A placebo feature is always present.
  A new feature is only accepted if its fold-averaged gain importance beats the placebo.
  Non-tree models convert their own importance to the same scale so the gate applies to everything.
- Screening happens on fold 0 with seed 42 against the current champion on the same fold.
  A candidate that loses on fold 0 is dropped without a full 5-fold run.
  Champion replacement always requires the full 3-seed confirmation.

### Nested OOF for ensemble decisions

A single model's OOF AUC is honest because the model never saw the labels of the rows it scores.
An ensemble chosen by comparing many combinations on OOF is not, because the choice already looked at those labels.

So every ensemble judgment used a sealed fold.

![Nested OOF: seal one fold, choose members and combiner settings on the other four, score the sealed fold, repeat](assets/writeup-nested-oof.png)

1. Each member's OOF predictions already exist from the fixed 5-fold run.
2. Seal fold k. Take the OOF rows and labels of the other four folds.
3. Choose members, combiner weights and combiner hyperparameters on those rows only.
4. Apply the frozen combiner to the OOF rows of fold k.
5. Repeat for k = 1..5, concatenate in the original row order, score ROC AUC.

No model is retrained in this loop and there is no inner re-splitting.
It is not a full double cross-validation.
It only removes the optimism that comes from selecting the ensemble on the same rows you score.

From this point every experiment answered two separate questions.
Is it good alone (its own OOF AUC)?
Does it help together (the nested OOF delta when added to the current pool)?
A model with a lower solo score stayed as a candidate when it was wrong on different rows than the pool.
A model with a high solo score was dropped when it was wrong on the same rows.
The duplicate rule was Spearman rank correlation above 0.998 with any existing member.

### Adoption gate

A pool change was adopted only if two things held, both fixed before looking at the result.

- The nested OOF delta cleared a threshold (+0.00002 for late pool changes).
- The delta on the sealed fold was positive in at least 3 of 5 folds early on, and 5 of 5 for the final pool.

Example: replacing five pool members with their missingness-augmented versions gave +0.0000469 nested OOF, positive on 5/5 sealed folds, and was adopted.
Widening the final stack from 314 to 327 columns gave +0.0000047, positive on 3/5, and was rejected.
That 327 version was submitted for the record anyway and scored Private 0.97108 against 0.97109 for the 314 version, so the gate was not wrong.

## Features

The 12 raw columns were observed in several representations and different model families got different subsets.
Counting feature providers across the final 36 configs: derived features 30, constraint-based imputation 27, target encoding 18, XGBoost-based imputation 14, exact-value categorical copies 3, frequency encoding 2, lattice pair target encoding 1, and five configs with original-dataset proxies.

### Exact-value categorical copies

The first big jump was not a new model.
Keeping the 9 numeric columns and adding 9 categorical copies (same value = same category) moved LightGBM from 0.96276 to 0.96605.
Converting everything to categorical and dropping the numeric columns went the other way, to 0.95859.
The problem was never categorical treatment, it was throwing away order and distance.
Both views stayed side by side for the rest of the competition.

| Representation | OOF AUC | Delta |
| --- | ---: | ---: |
| Numeric only (baseline) | 0.96276 | |
| All 12 columns as categorical | 0.95859 | -0.00417 |
| Numeric + exact-value categorical copies | 0.96605 | +0.00329 |

Same LightGBM settings for all three (`num_leaves=255`, `learning_rate=0.05`, early stopping 200, seed 42).

### Screen-time budget

- `other_screen = daily - (social + gaming + work)`, NaN if any part is missing.
- `screen_slack = daily - sum of observed parts`, NaN only if daily is missing.
- Ratios of each part to daily, weekday/weekend differences, `weekend - slack`, and a count of how many parts were observed for the slack.

### Imputation as auxiliary columns

Raw columns were never overwritten.
Imputed values were added as `<col>_recon` columns next to the raw ones.

- Constraint-based: an `IterativeImputer` fit on the training fold, with the screen-time estimates clipped to the interval allowed by `daily >= social + gaming + work`.
- XGBoost-based: per-column XGBoost imputers fit on the training fold, plus compositions built on the imputed values (fractions, awake-time screen fraction, weekend minus daily, and so on).

### Target encoding

Exact-value target encoding, fit inside each training fold with an inner split, was used by half the pool.
One config used a lattice of pairwise exact-value target means.

The placebo gate did real work throughout.
For example, a batch of nine composition features passed the AUC gate as a group, but four of them had lower gain than the placebo and were cut before the confirmation run.

### Original dataset proxies

Five configs used the public `Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv` as a proxy for the generator's source: nearest-neighbour features, prior means, class-conditional CDF differences, and a first-stage prediction from the original data.
They were modest alone but were wrong on different rows than the rest of the pool.

### Missingness augmentation

Six of the final 36 members were trained on 3x rows: the original training fold plus two copies where each observed cell was masked with probability 0.25.
Existing NaNs stay NaN, copies inherit fold and label, and all preprocessing state is fit on the original rows only.
The gain came from mask diversity rather than from realism of the masks.
A version that copied the empirical row-level mask patterns from train and test instead of independent masking did not pass the gate.

## Models

Final own pool: 36 members, each a 3-seed average.

| Family | Members | Best OOF AUC |
| --- | ---: | ---: |
| Lookup-Transformer | 5 | 0.96941 |
| RealMLP | 4 | 0.96855 |
| LightGBM | 14 | 0.96884 |
| CatBoost | 3 | 0.96875 |
| XGBoost | 2 | 0.96833 |
| Contextual Spline Transformer | 1 | 0.96814 |
| TabM | 1 | 0.96838 |
| Scalar Token Transformer | 1 | 0.96782 |
| TabPFN-3 | 1 | 0.96724 |
| TabCNN | 3 | 0.96795 |
| One-hot exact-value logistic regression | 1 | 0.95962 |

The best LightGBM member is a config taken from an AutoGluon-style hyperparameter portfolio, trained on missingness-augmented rows.

Solo score was not the entry ticket.
The one-hot logistic regression at 0.9596 stayed because it was the most different model in the pool, and its OOF logit was also used as an init score for one of the LightGBM members.

### Lookup-Transformer

This is a port of tamerlanomralinov's public Lookup-Transformer notebook from this competition, with the vocabulary and rank-gauss quantiles fit on the training fold only.
Each column becomes one token.
The token is the sum of an exact-value lookup embedding (the repeated keys of the synthetic data) and a PLR embedding of the rank-gauss value (the smooth trend).
Values that only appear in validation or test map to a per-column UNK id, distinct from the NA id.
Tokens go through a 4-layer, width-128, 8-head encoder and a CLS head.

Training: 32 epochs, batch 2,048, OneCycle with peak LR 2e-3, EMA, value dropout.
The best variants used Muon instead of AdamW, averaged several initializations per fold, and added 5 PLR-only composition tokens on the imputed screen-time columns.

The first version scored 0.96892 against a champion of 0.96854 (+0.00038, positive on 5/5 folds).
Its nearest neighbour in the pool by Spearman was an XGBoost at 0.98149, well under the 0.998 duplicate line, and adding it to the ensemble gave +0.00025.
Trees and this model were wrong on different people, which is exactly what the pool needed.

I then spent 17 configs on its learning rate, schedule and optimizer.
None beat the champion on fold 0, so none were promoted.
The same fold-0 gate stopped TabR-S, TabICLv2, AMFormer and Trompt at -0.027 to -0.244 against the champion.

### The RealMLP dtype bug

The largest single-model gain of the competition was a bug fix.
My RealMLP port also used numeric columns as exact-value categories.
The vocabulary was built from float64 values, but the encoding step cast to float32 before the lookup.
Any value with a fractional part changed slightly under the cast and missed the vocabulary, so the categorical channel for 6 columns was effectively dead during training.

| | Before | After |
| --- | ---: | ---: |
| Validation cells mapped to UNK (one fold, 12 categorical columns) | 800,896 | 23 |
| 3-seed OOF AUC | 0.96371 | 0.96832 |

Moving the cast after the lookup was the whole fix, worth +0.00461.
After that I spent as much time checking that existing configs did what they claimed as searching for new ones.

## Ensembling

### Combiner

For each column, compute the empirical rank and the logit of the raw probability, standardize both, and fit an L2 logistic regression on all columns.
Negative weights are allowed.
The output rank is shrunk toward the plain rank mean of the members with a factor `lambda`.

`C` in {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0} and `lambda` in {0.25, 0.5, 0.75, 1.0} are chosen by leave-one-fold-out inside the outer training folds, so they are part of what the sealed fold judges.
On the final full-OOF fit the choice was `C = 0.03`, `lambda = 1.0`, meaning the plain rank mean was not mixed in at all.
Ties go to the smaller `C`, then the smaller `lambda`.

For the test prediction the combiner is refit once on the full 314-column OOF matrix and applied to the test matrix in the same column order.

### Public OOF/test pairs as members

Many participants published OOF and test prediction pairs as datasets or notebook outputs.
I treated every pair as a candidate member with the same checks as my own models.

- 691,369 OOF rows and 296,302 test rows, all finite.
- Re-scored AUC on our labels within 1e-5 of the declared AUC.
- Deduplicated by a hash of the prediction arrays.
- Fold evidence recorded: author statement (152), public code (98), sibling code by the same author (13), an included fold vector matching ours (12), none (3).
- Two members were excluded for code-level target-mean leakage.

Licence was recorded per member: CC0 203, CC BY 4.0 6, Apache 2.0 5, unknown 61, other 3.
The 64 unknown/other arrays were used only as combiner inputs and are not committed to the repository or redistributed.
Sources include the OOF libraries and fusion sets from szymonkapiski, hboyang, paiky1995, najiama, beicicc, raykkretzschmar, masayakawamata and outputs of about 30 public notebooks.

The public predictions do not all share our fold assignment.
Nested OOF removes the selection bias of the combiner, but it cannot remove selection bias that happened upstream when the authors built their models.
That caveat stands.

### Progression

| Step | Columns | Nested OOF AUC |
| --- | ---: | ---: |
| Own pool | 35 | 0.96981 |
| Plus validated public predictions (ledger v1) | 242 | 0.97029 |
| Plus new public predictions, minus 120 harmful columns | 313 | 0.97035 |
| `C` selected inside outer folds | 313 | 0.97036 |
| Own pool updated to 36 members | 314 | 0.97038 |

The 120 removed columns were a family of weak classical probabilistic models from one library.
Adding all 433 columns gave +0.0000063 over 242 on 3/5 folds.
Adding the same set without those 120 gave +0.0000633 on 5/5 folds.
Width helps only when the new columns carry something the pool does not already have.

Own members' test predictions come from full-train refits (each config refit on all rows with the training length observed in CV).
Public members' test predictions are whatever the author published, normally a fold average.
Since the combiner works in rank space per column, the mismatch in scale does not matter.

## Submissions

| Submission | Columns | Nested OOF | Public | Private | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| Own 35, safety pick | 35 | 0.96981 | 0.97099 | 0.97063 | yes |
| Own 35 + public 207 | 242 | 0.97029 | 0.97134 | 0.97106 | no |
| Own 35 + public 278 | 313 | 0.97035 | 0.97135 | 0.97108 | no |
| Same 313 with `C` selection | 313 | 0.97036 | 0.97135 | 0.97109 | no |
| Own 36 refit + public 278 (final) | 314 | 0.97038 | 0.97135 | 0.97109 | yes |
| 314 + 13 strict candidates | 327 | 0.97039 | 0.97133 | 0.97108 | no, for the record |

Public and Private moved together with nested OOF across the whole ladder, with an offset of about +0.0010 between nested OOF and Public.

## What did not work

- All-categorical representation (-0.00417).
- 17 Lookup-Transformer learning-rate, schedule and optimizer variants: none beat the champion on fold 0.
- TabR-S, TabICLv2, AMFormer, Trompt: -0.027 to -0.244 on fold 0. Trompt would also have needed about 40 hours for 5 folds.
- Row-mask missingness augmentation copied from real missing patterns: below the independent-mask version.
- DAE latent representations fed to Lookup-Transformer and CatBoost: -0.00002 and -0.00016.
- CatBoost on GPU was 5.7x faster but -0.00013 OOF, so CPU stayed the reference. XGBoost on GPU was 3.8x slower on this data.
- Rebuilding the own pool from scratch with an exact search over 6 ranges: the proposed 13-member pool scored -2.5e-06 under the incumbent, so the incumbent stayed.
- The extra 13 strict external candidates at the end (+0.0000047, 3/5).

## Infrastructure

One experiment is one YAML config, committed to git with the fixed folds, seeds and locked dependencies.
The same `pipeline.run <config>` command runs on a laptop, on Kaggle CPU/GPU, on Vast.ai and on Runpod.
No environment ever had its own training loop.

![Experiment pipeline: one committed config runs anywhere, the returned bundle is verified locally before it enters MLflow](assets/writeup-experiment-pipeline.png)

Remote runs come back as a bundle (config, predictions, metrics, diagnostics) with a SHA-256.
On import the laptop checks the input data hashes, that the commit exists and contains that config, that the tree was clean, and re-scores the OOF against local labels before the run is accepted.
Only accepted runs go into the local MLflow, which ended at 455 runs.
Remote instances and volumes are deleted right after import and the billing is checked.

Kaggle's free T4 was where the first Lookup-Transformer and TabM runs happened, but 6-7 hour runs against a 30 hour weekly quota and a 9 hour session limit did not scale.
Vast.ai became the primary GPU provider (a matched run cost $0.12 there versus $0.24 on Runpod), with Runpod as fallback.
The final refit of the one changed neural member was 3 seeds on 3 GPUs for about $0.39.
Wide stacking judgments were memory bound: 400-column jobs took 10-16 GB each and five in parallel rebooted the laptop, so three became the cap.

### Working with coding agents

Most of the repetitive work went to Claude Code and Codex: reading prior write-ups and notebooks, turning a question into a GitHub issue with the adoption criteria written down before running anything, implementing, running, and recording results.
I decided what to test and what to adopt.

![Loop between the human, the coding agents, and the shared GitHub issues, research notes and MLflow records](assets/writeup-agent-loop.png)

Over the competition that loop handled 606 issues, 92 pull requests and 964 commits.
The research side covered 12 top write-ups from 11 similar tabular competitions, 14 blog posts, all 25 early discussion threads with incremental updates afterwards, and a catalogue of 267 upvoted public notebooks of which the top 37 were read at cell level.
Findings went straight into experiment candidates and were judged with the same nested rule as everything else.

## Takeaways

- Fix the folds and the stopping rules before seeing results, then stop early on small evidence.
- Judge "good alone" and "helps together" separately. The second one is what the leaderboard pays for.
- Re-score everything locally, whatever machine it ran on.
- The biggest single-model gain was a dtype bug. Verifying that a config does what it says is worth as much as a new idea.
- Next time I would spend the first week searching wider across model families with different inductive biases, and less time tuning the one that is currently ahead.

Code, configs, ADRs and the full decision history are in the repository linked below.
