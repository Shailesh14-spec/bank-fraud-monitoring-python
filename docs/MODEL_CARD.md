# Card fraud model card

**Purpose:** reproducible offline card-fraud ranking and closed-hour analyst-review simulation on public labelled data. Not an AML model, a live transaction engine or a production fraud-control claim.

**Inputs:** CSV `Time`, `Amount`, and `V1`…`V28`. Values must be numeric/nonmissing; Amount cannot be negative. Class is used only during evaluation; unlabelled replay does not require it. Relative time drives batching and splits. Original PCA semantics and entity IDs are unavailable.

**Outputs:** local fraud research scores and an analyst queue capped per completed hour. No card is blocked, no account is frozen, and no customer is contacted. Scores are not known to be calibrated for another bank's fraud prevalence or label process.

**Selection:** weighted logistic regression and histogram gradient boosting, with separate calibration and chronological validation. Validation recall at a ten-per-hour review capacity is primary; average precision breaks ties. Selected model, uncertainty and test queue metrics are reported from execution, not estimates invented for the portfolio.

**Limits:** two-day sample, sparse validation frauds, unknown label delays, no card- or customer-group holdout, unknown upstream PCA fit procedure, and no realistic streaming infrastructure. Closed-hour top-K has review latency. Anonymised component importance does not create customer-level adverse-action reasons or proof of legal/regulatory suitability.

**Operational extension:** a genuine pilot would require recent consented data, realistic delayed-label backtesting, event schemas, entity-safe feature stores, latency measurements, drift response, access control and analyst feedback. None is represented as deployed here.
