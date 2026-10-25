# Measured findings

Validation selected **gradient_boosting**. On the later held-out period, average precision is **0.782** against prevalence **0.0012**; ROC AUC is 0.974.

At **10 reviews per closed hour**, the model flags **78 of 114** labelled fraud transactions (68.4% recall) with 65.0% precision and 120 total reviews. Flagged fraud amount share is 61.0%; this is not a prevented-loss estimate. The source amount's currency is not assumed in this project.

A validation-selected probability threshold yields 0.55% test review share, which need not respect an hourly budget. The strict closed-hour queue and fixed-threshold policy are separate operational designs.

Recommendation: treat this as a reproducible offline card-fraud triage benchmark. Data covers about two days, PCA feature semantics are unavailable, and realistic label delays are not represented. A one-hour gap prevents adjacent windows touching but does not simulate delayed chargebacks. No AML detection, real-time system, customer-level explanation or deployment performance is claimed.
