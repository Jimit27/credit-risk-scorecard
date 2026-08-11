# Data quality: what is wrong with this extract, and what was done about it

The Give Me Some Credit file is real consumer credit data, and it is messy in the
specific ways real credit data is messy. Every rule in `creditrisk.clean` exists
because of one of the defects below. The counts come from
`reports/data_quality.json`, which the pipeline regenerates on every run.

Nothing here is a judgement call hidden in code: the rules live in
`conf/config.yaml`, and each one reports how many records it touched.

| Defect | Records | What was done | Why |
|---|---|---|---|
| `MonthlyIncome` written as the literal text `NA` | 29,731 | Cast with `TRY_CAST` in silver, counted, then flagged | Bronze lands everything as text. Casting is a decision, so it belongs where it can be audited, not in a reader option |
| `NumberOfDependents` written as `NA` | 3,924 | As above | Same |
| Byte-identical duplicate applications | 609 | Dropped | An application counted twice inflates whatever pattern it carries |
| Sentinel codes `96` and `98` in the three delinquency counters | 225 | Nulled, and a `flag_delinquency_sentinel` retained | These are administrative codes, not counts. Untreated, they tell the model that 225 people missed ninety-eight payments and mostly repaid |
| `age` of 0, or above 100 | 14 | Nulled, imputed to the median | A newborn does not hold nine credit lines |
| `DebtRatio` above 100 where income is missing | 22,691 | Moved into `monthly_debt_amount`, original nulled, `flag_debt_ratio_is_amount` set | The column switches units. Where there is no income to divide by, the field holds an absolute monetary amount, so the same column means two different things |
| `RevolvingUtilizationOfUnsecuredLines` up to 50,708 | 371 | Winsorised to [0, 2] | Utilisation is bounded by definition. Winsorising rather than dropping keeps the rest of the record, which is still informative |
| `DebtRatio` above 5 (max 61,107 among rows that reach this rule) | 6,950 | Winsorised to [0, 5] | Same. The larger values, up to 329,664, belong to the income-missing rows above and are nulled by rule 4 before winsorisation ever sees them |
| `MonthlyIncome` up to 3,008,750 | 301 | Winsorised to [0, 50,000] | Same |

## The one that is easy to get wrong

The sentinel codes are the interesting case. The obvious move is to drop those 225
rows, or to treat 96 and 98 as counts and let the model deal with it. Both are
mistakes.

Kept as their own WoE bin, those records turn out to have an observed default rate
around **60%**, against a book average of 6.7%. Whatever the bureau means by
coding a record rather than counting it, it is one of the strongest single signals
in the dataset. Dropping the rows throws it away; treating 98 as a count buries it
in a tail the binning will smooth over.

This is the general argument for giving missingness its own bin rather than
imputing it: in credit data, the absence of a record is a fact about the applicant.

## Missingness is not random

19.6% of incomes are absent, and those applicants are not a random sample - the
absence correlates with the debt-ratio unit problem above, which is why both flags
survive into the feature set even though neither cleared the Information Value
floor on its own. The imputation is a median within age band rather than a global
median, because income and age are strongly related and a single global figure
would flatten that structure across 30,000 records.
