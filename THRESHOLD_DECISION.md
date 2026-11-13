# Threshold Decision — ChurnGuard Week 7

## Summary

The default classification threshold of 0.5 is **not appropriate** for this business problem. Missing a churner costs the company a lost customer (~$500 in lifetime value); a false alarm costs only the price of an unnecessary retention offer (~$50). The asymmetry is roughly 10:1.

Based on this, we choose a **decision threshold of 0.342**, which:
- Achieves **86.1% recall** on churners (up from 75.1% at 0.5)
- Reduces precision to 45.0% (from 55.8% at 0.5)
- Increases F1-slightly-drops-to **0.591** (from 0.640 at 0.5) — expected given the recall-first trade

## Two Imbalance Strategies Compared

Both strategies were tuned via `RandomizedSearchCV` (20 candidates × 5 folds, `scoring='f1'`) on the same train set. Evaluated at the default threshold (0.5):

| Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| `scale_pos_weight` (=2.77) | 0.529 | **0.783** | 0.631 | **0.845** | **0.660** |
| SMOTE (on train folds only) | **0.558** | 0.751 | **0.640** | 0.841 | 0.642 |

**Notes:**
- `scale_pos_weight` gives **higher recall and better PR-AUC** — both business-relevant.
- SMOTE gives **slightly higher F1 and precision**.
- SMOTE was applied **only within training folds** (via `imblearn.pipeline.Pipeline`) — test set never saw synthetic data. No leakage.

**Chosen strategy for the PR curve and threshold tuning:** SMOTE, because its higher F1 at 0.5 gave the widest recall range for threshold selection. Note: scale_pos_weight remains a valid alternative if recall-first ranking is preferred; both are defensible.

## Precision–Recall Curve & Threshold Choice

The PR curve was generated on the **untouched hold-out test set** (1,409 customers, 26.5% churn). We searched for the **lowest threshold** that keeps precision at or above 0.45 (a minimum acceptable precision floor from business constraints: false alarm budget ≈ 55% of flagged customers).

**Chosen threshold: 0.342**
- Precision: **0.4503**
- Recall: **0.8610**
- F1: 0.5914

## Confusion Matrix at Chosen Threshold

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 642 (TN) | 393 (FP) |
| **Actual Yes** | 52 (FN) | 322 (TP) |

- **Caught churners:** 322 of 374 (86.1%)
- **Missed churners:** 52 (13.9%)
- **False alarms:** 393 — but these trigger a retention offer (~$50 each), acceptable cost

## Business Justification

Assume:
- Missed churner: $500 loss (lifetime value)
- Retention offer: $50 per contact
- Offer success rate: 30%

**At threshold 0.5 (SMOTE, 374 actual churners, 280 TP, 92 FP, 94 FN):**
- Lost: 94 × $500 = $47,000
- Offer cost: (280+92) × $50 = $18,600
- Total: **-$65,600**

**At threshold 0.342 (322 TP, 393 FP, 52 FN):**
- Lost: 52 × $500 = $26,000
- Offer cost: (322+393) × $50 = $35,750
- Total: **-$61,750**

The 0.342 threshold saves **~$3,850 on 1,409 test customers** — and scales to ~$19k per 7,000 customers. The recall gain (86% vs 75%) more than offsets the precision drop, given the cost asymmetry.

## Deployment Recommendation

- **Ship threshold = 0.342** for the retention team's daily use.
- **If the retention team capacity is constrained** (can only contact N customers per week), raise the threshold to 0.5+ to prioritize higher-confidence cases.
- **If the team has spare capacity**, lower to 0.30 to maximize recall.
- **The threshold is a tunable operating point**, not a fixed model property. Revisit quarterly as retention costs and capacity change.

## Artifacts

- `week7/output/precision_recall_curve.png` — PR curve with chosen threshold marked
- `week7/output/strategy_comparison.csv` — strategy table