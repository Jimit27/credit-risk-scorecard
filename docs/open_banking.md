# The Open Banking track

## The honest framing, first

The transaction ledger in this project is **simulated**. Nothing in this track
feeds the scorecard, and **no predictive claim is made from it**.

That is not a shortcut, it is the constraint. There is no public, licensable
dataset of real current-account transactions linked to real default outcomes -
that data is precisely the kind that cannot be released. Any project claiming a
"cashflow uplift over bureau data" on public data is either using a synthetic
generator and not saying so, or measuring an artefact of its own simulation.

So the ledger is generated from a documented process in
`creditrisk.openbanking.generate_ledger`, and this module demonstrates the part
that actually transfers: the engineering an affordability model requires.

## Why it is here at all

Lenders like Abound build affordability decisions on Open Banking data rather
than bureau files alone. The bureau tells you what someone has *repaid*; the
transaction stream tells you what they can *afford*. Two applicants with identical
credit files can have completely different capacity once you can see that one is
salaried and the other is on variable gig income with three overdraft fees a
month.

Getting from a raw transaction feed to that judgement is mostly two problems.

## Problem 1: merchant strings do not generalise

Real bank descriptors are truncated, upper-case, and welded to store numbers:
`TESCO STORES 3452`, `UBER *TRIP 88121`, `DD HOUSING ASSOC`. Categorising them is
the messiest step in any Open Banking product.

The evaluation here is on **held-out merchants** - whole merchant stems removed
from training, so the model meets strings it has never seen. A random split of
transactions would score far higher and mean nothing, because the same merchant
would sit on both sides of it.

| Model | Accuracy on unseen merchants |
|---|---|
| Uniform guess across 12 categories | 8.3% |
| **Always guess the most common category** | **21.3%** |
| Character n-grams on the description | **13.2%** |
| Description **plus transaction behaviour** | **36.7%** |

The majority-class row is the baseline that matters. The held-out transactions are
heavily skewed - groceries 21%, shopping 20%, eating out 20% - so measuring against
a uniform 1-in-12 guess would flatter every model in the table.

The finding is the interesting part, and the first half of it is negative. Text
alone lands *below* the majority-class baseline on a merchant it has not
memorised - nothing in `OCTOPUS ENERGY` resembles `EDF ENERGY` at the character
level, and the n-gram model does worse than simply always guessing "groceries"
because it confidently maps unfamiliar strings onto whichever category shares
incidental character patterns with them. Memorisation that looks like learning,
until the merchant changes.

Adding how the transaction *behaves* - amount, sign, recurrence, amount stability,
share of the month's outflow - takes it to 36.7%, or 1.7x the majority-class
baseline, because an unseen merchant is still recognisable as "monthly, fixed
amount, large" versus "frequent, variable, small".

37% is still poor, and that is the honest conclusion: **unseen-merchant
categorisation is genuinely hard**, which is exactly why commercial providers
maintain large curated merchant dictionaries and lean on the model only for the
tail. Across the full ledger agreement is 87%, but that figure is **in-sample** -
most of those merchants were in training - and it is reported for context, not as
a result. A project reporting 95% on unseen merchants has almost certainly leaked
its merchants across the split.

Two caveats on the numbers above, since the doc criticises other projects for
exactly this: they come from a **single seed** with 16 held-out merchants and no
variance estimate, and the difficulty is set by the author's own ~60-string
merchant catalogue. The *direction* of the finding is robust and the mechanism is
real; the precise percentages are a property of this simulation.

## Problem 2: turning a stream into an affordability judgement

Eighteen features are derived per applicant, grouped by the question they answer:

**Is the income real and dependable?**
`ob_income_monthly_median`, `ob_income_volatility` (coefficient of variation),
`ob_income_is_regular`, `ob_income_payment_count`. A delivery rider and a salaried
nurse can have the same mean income and completely different affordability.

**Does income cover the non-negotiables?**
`ob_essential_spend_monthly`, `ob_essential_spend_ratio`,
`ob_surplus_after_essentials`, `ob_essential_cover_ratio`. Housing, utilities,
groceries and transport are treated as essential; the surplus after them is the
headline affordability number.

**How much flex is there?**
`ob_discretionary_spend_monthly`, `ob_discretionary_spend_ratio`,
`ob_spend_volatility`. Discretionary spending is what can be cut if a repayment
gets tight.

**Is the account already under strain?**
`ob_min_balance`, `ob_mean_balance`, `ob_days_in_overdraft`, `ob_fee_events`,
`ob_fees_monthly`, `ob_cash_monthly`, `ob_debt_repayment_monthly`. Overdraft fees
in the simulation are a *consequence* of the balance actually going negative
rather than an independent coin flip, which is what makes the derived features
behave like real ones.

## What the generator does

Stated so the output cannot be mistaken for real data. Each applicant draws:

- a log-normal monthly income, and a pay rhythm (monthly or four-weekly);
- a latent "financial pressure" level, which raises discretionary spending, cash
  withdrawals, income irregularity and overdraft exposure;
- a housing cost as a share of income, rising with pressure;
- **fixed counterparties** for recurring categories - one landlord, one energy
  supplier, one employer, a handful of subscriptions each billed monthly at the
  same amount. Drawing a fresh merchant each month would destroy the recurrence
  structure the behavioural features depend on, and no real ledger looks like
  that.

Balances are carried forward transaction by transaction, so the overdraft
features are computed from a coherent balance series rather than asserted.

## What would make this real

With a genuine Open Banking feed the same code would run unchanged; what would
change is the evaluation. The question worth answering - does cashflow data add
discrimination over a bureau file for thin-file applicants? - needs real
transactions joined to real outcomes, and can only be answered inside a lender.
