# Modelling decisions

The choices below are the ones a reviewer would question. Each is stated with the
alternative that was rejected and why.

## The interpretable model is the champion, not the fallback

The Weight-of-Evidence logistic scorecard is the champion; the gradient-boosted
model is the challenger. That is the opposite of the usual portfolio-project
instinct, and it is the right way round for lending.

A scorecard collapses into a printable points table - see
`reports/woe_bin_table.csv` and the app's "The scorecard" tab. Every decline can
be explained exactly, not approximately, and a credit committee can read the whole
model on two sides of paper. The challenger exists to price that: it quantifies
what interpretability costs in discrimination.

On the held-out sample it costs about **two Gini points** (0.712 against 0.734).
At that price,
interpretability is essentially free and the scorecard ships. Had the gap been
fifteen points, the honest conclusion would have been different - which is the
point of measuring rather than assuming.

## Why the top Information Values look alarming

Seven features have IV above 0.5, which the conventional Siddiqi reading labels
"suspicious - check for leakage". They are not leakage, and the label in
`reports/information_values.csv` is deliberately left in view rather than
suppressed.

The target is default within a **two-year forward window**. The delinquency
counters describe payment behaviour **before** that window opens. Past arrears
predicting future default is the entire basis of consumer credit scoring, not a
leak: bureau delinquency variables routinely carry IVs above 1.0 on real books.

The check that actually rules leakage out is temporal, not numeric: no feature in
`creditrisk.features` reads anything recorded after the observation point. There is
no post-outcome information in the file to leak.

## Correlated features are pruned, and it barely costs anything

The engineered delinquency features are near-restatements of one another -
`total_delinquencies` correlates 0.93 with `worst_delinquency_severity` on the WoE
scale. Left in, they split one signal across several coefficients and make the
points table unreadable.

Five of the eighteen features that clear the IV floor are dropped for pairwise
correlation above 0.85, keeping the higher-IV member of each pair. A sixth,
`times_60_89_dpd`, is dropped as **degenerate**: 95% of applicants sit at zero, so
after minimum-bin merging it has one populated bin and separates nobody. Its IV
came entirely from 149 sentinel-coded records, and left in it would have ranked
near the top of the points table on a range only 0.2% of the book can reach. That
is also why `global_importance` ranks on a **population-weighted** spread rather
than raw points range.

Twelve features survive, at a cost of about **0.4 Gini points**, for a model a
third smaller and a points table a human can actually read. The exclusion reason
for every dropped feature is recorded in `reports/information_values.csv`.

## Calibration is fitted on validation, not on train

The raw model output is turned into a usable probability by an isotonic
regression fitted on the **validation** split. Calibrating on training data
produces a curve that is already fitted to those residuals and reports a
calibration quality that will not survive contact with new applicants.

This matters more than the ranking metric for a lender. A model that ranks
perfectly but predicts 3% where the true rate is 9% will price the entire book
wrong. On the held-out sample, predicted and observed default rates agree to
within 1.9 percentage points in every decile - see `reports/calibration_table.csv`.

## The split is hashed, not sampled

Train / validation / test membership is a hash of the applicant id, assigned in the
gold layer. A random sample with a seed is reproducible only as long as nobody
changes the row order, the row count, or the library version. A hash is stable
against all three, so an applicant cannot drift across the boundary between runs
and quietly leak.

## The cut-off is a business decision, not an F1 score

There is no threshold that "maximises accuracy" in lending, because approving a
bad loan and declining a good one do not cost the same. `creditrisk.metrics`
sweeps the approval rate and prices each point with a stated loss-given-default
and revenue-per-good-loan, both declared in `conf/config.yaml` rather than buried
in code.

The threshold is chosen on the **validation** split and then applied unchanged to
test. Maximising profit on the test set and then quoting that test set as an
unbiased estimate would be selecting a parameter on the holdout. The chosen
threshold approves 87.0% of validation and 87.1% of test, at a 3.2% bad rate and
£2.2m more expected profit than approving everyone.

Those economics are illustrative figures for an unsecured consumer book - the
number to take from it is the method, not the £ figure.

## What this model does not do

Stated plainly, because a portfolio project that claims completeness is less
credible than one that knows its own limits:

- **No reject inference.** The data only contains accepted applicants, so the
  model is trained on a population already filtered by somebody else's credit
  policy. A production scorecard would need reject inference to correct for that.
- **No temporal validation.** The extract carries no origination dates, so there
  is no out-of-time holdout. Discrimination on a future vintage would very likely
  be lower than the number reported here. The drift monitor exists precisely
  because that decay cannot be measured up front.
- **No fairness audit against protected characteristics.** The dataset contains
  none, so none can be tested. The segment analysis by age band in
  `reports/metrics.json` is a partial substitute and no more than that. Age is
  itself a protected characteristic in several jurisdictions, and its presence in
  the feature set would need a policy decision before this went anywhere near
  production.
- **The Open Banking track is simulated** and contributes nothing to the headline
  metrics. See `docs/open_banking.md`.
