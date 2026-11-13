    # Error Analysis — ChurnGuard Week 7

## Summary

At the chosen threshold (0.342) on the untouched test set (1,409 customers), the model produced **445 errors**:
- **52 False Negatives** (missed churners)
- **393 False Positives** (false alarms)

**Recall is 86.1%, precision 45.0%.** The high FP count is expected and acceptable given the recall-first threshold — but the systematic patterns below reveal where the model still fails.

## Sample of 10 Misclassified Customers

### 5 False Negatives (churned but predicted safe)

| tenure | Contract | MonthlyCharges | InternetService | PaymentMethod | proba |
|---|---|---|---|---|---|
| 45 | One year | $20.40 | No | Electronic check | 0.124 |
| 12 | One year | $74.05 | DSL | Mailed check | 0.270 |
| 17 | One year | $98.60 | Fiber optic | Bank transfer | 0.231 |
| 54 | One year | $106.55 | Fiber optic | Bank transfer | 0.301 |
| 63 | One year | $99.90 | Fiber optic | Bank transfer | 0.291 |

### 5 False Positives (stayed but predicted churn)

| tenure | Contract | MonthlyCharges | InternetService | PaymentMethod | proba |
|---|---|---|---|---|---|
| 8 | Month-to-month | $100.15 | Fiber optic | Credit card | 0.724 |
| 18 | Month-to-month | $78.20 | Fiber optic | Electronic check | 0.516 |
| 21 | Month-to-month | $99.85 | Fiber optic | Credit card | 0.632 |
| 21 | Month-to-month | $99.15 | Fiber optic | Bank transfer | 0.534 |
| 70 | Month-to-month | $106.05 | Fiber optic | Electronic check | 0.472 |

## Patterns Identified

### Pattern 1: Model ignores churn risk for One-Year contract customers (FN pattern)

**Observation:** All 5 sampled false negatives had **One year contracts**, with high tenure (mean 38.2 months) and moderate-to-high monthly charges ($79.90 average). Four of five used **Bank transfer (automatic)** payment — the "safest" payment method.

**Why:** `ContractRisk` (a key engineered signal) assigns 0 to One-year and Two-year contracts, and the model learned during training that One-year customers rarely churn. But when a One-year customer has **Fiber optic + high charges + automatic payment**, this is actually a latent churn signal the model under-weights.

**Business impact:** These are high-value customers ($80-$106/month) — missing them is expensive. **52 FN total**, likely dominated by this pattern.

**Mitigation ideas:**
- Add interaction feature `ContractRisk × MonthlyCharges` — capture "safe contract but expensive" outliers
- Assign higher sample weight to churned One-year customers during training
- Manual review queue for One-year + Fiber optic + high charges, even if model says safe

### Pattern 2: Over-fires on Month-to-month + Fiber optic regardless of tenure (FP pattern)

**Observation:** All 5 sampled false positives had **Month-to-month contracts**, **Fiber optic internet**, and **high charges ($96.68 average)** — but tenures ranged from 8 to 70 months. The model treats tenure as secondary when ContractRisk and InternetService are strong positive signals.

**Why:** `ContractRisk=1` (month-to-month) combined with high charges is the model's strongest learned churn pattern. It over-fires on this combination even when tenure (which normally protects) is very high (70 months in one case).

**Business impact:** 393 FP → 393 wasted retention offers (~$19,650 in offer cost at $50 each). Not catastrophic, but significant.

**Mitigation ideas:**
- Add `ContractRisk × Tenure` interaction — "month-to-month but long-tenured" should reduce churn risk
- Cap `ContractRisk` effect for tenure > 48 months
- Split evaluation by tenure bucket; report FP rate separately for tenured vs. new customers

### Pattern 3: Payment method carries under-weighted signal

**Observation:** Most false negatives use **Bank transfer (automatic)** — the model may learn "automatic = safe" too strongly. Meanwhile, **Electronic check** appears in both FN and FP — this payment method is genuinely ambiguous, correlating with churn but also common in long-tenured customers.

**Mitigation:** Evaluate model fairness across `PaymentMethod` segments. If electronic-check customers are disproportionately misclassified, consider a payment-method-specific threshold.

## Systematic Weakness (Summary)

The model has learned **Contract type + InternetService + MonthlyCharges** as its three dominant decision axes. When these three signals disagree with **tenure** (the model's 4th-most-important feature), the model tends to:

1. **Under-predict churn** for high-tenure One-year customers who show churn behavior elsewhere (Fiber optic + high charges)
2. **Over-predict churn** for high-tenure Month-to-month customers who simply happen to have expensive plans

This is the classic "static rule + borderline case" failure mode: the model relies on the strongest historical pattern (contract type), and can't represent the rare-but-real cases where tenure or payment method reverses it.

## Recommendations for Production

1. **Do not deploy at threshold 0.342 for high-value customers without human review** — 52 FN are concentrated among expensive Fiber optic One-year subscribers. High-value misses are expensive.
2. **Build a two-tier system:**
   - Tier 1 (auto-offer): probability ≥ 0.65
   - Tier 2 (human review): probability 0.30–0.65, especially One-year + Fiber optic + high charges
3. **Retrain quarterly**, but **evaluate FN distribution by segment** — if One-year FN keep growing, add segment-specific sample weights.
4. **Add the two interaction features** (`ContractRisk × Tenure`, `ContractRisk × MonthlyCharges`) in the next iteration.

## Artifacts

- `week7/output/error_analysis_sample.csv` — 10 misclassified rows (5 FN + 5 FP) with proba
- `week7/output/precision_recall_curve.png` — PR curve used for threshold selection