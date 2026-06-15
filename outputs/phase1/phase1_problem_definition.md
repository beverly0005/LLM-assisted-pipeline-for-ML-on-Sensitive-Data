# Phase 1: ML Problem Definition
## Pharmacy Intervention Document Classifier

**Document Version:** 1.0  
**Date:** 2026-06-12  
**Domain:** Healthcare / Clinical NLP  
**Task Type:** Multi-class text classification

---

## 1. Precise ML Task Definition

This is a **supervised multi-class text classification** task. Given a free-text pharmacy intervention document, the model must assign it to exactly one of five mutually exclusive intervention categories at the time of document submission.

**Input:** Raw free-text content of a single pharmacy intervention document, optionally enriched with structured metadata (drug name, dose, patient age, ward).

**Output:** A single predicted class label from {Drug-Drug Interaction, Dose Adjustment, Therapeutic Substitution, Allergy Alert, Other}, along with a calibrated probability distribution over all five classes for auditability and threshold adjustment.

**Inference trigger:** Real-time, on document submission. The model must complete inference within 2 seconds on CPU-only hospital infrastructure.

**Interpretability requirement:** The model must surface token/phrase-level evidence for each prediction to support pharmacist review and audit. SHAP values, LIME explanations, or attention-weight summaries are acceptable mechanisms depending on model architecture chosen.

---

## 2. Prediction Target Definition

**Target variable:** `intervention_category` — a five-class nominal label assigned by the reviewing pharmacist at the time of documentation or retrospectively during audit.

**Class definitions and disambiguation guidance:**

| Class | Definition | Key distinguishing features |
|---|---|---|
| Drug-Drug Interaction (DDI) | Intervention triggered by a clinically significant interaction between two or more co-prescribed drugs | Mentions two or more drug names; interaction risk language (e.g., "QT prolongation", "bleeding risk", "serotonin syndrome") |
| Dose Adjustment | Intervention recommending a change in dose, frequency, or route for a single agent | Single drug focus; dose/weight/renal function language; "reduce", "increase", "adjust" |
| Therapeutic Substitution | Recommendation to replace one drug with a therapeutically equivalent alternative | Two drug names with substitution intent; "switch", "substitute", "equivalent", formulary language |
| Allergy Alert | Intervention flagging a documented allergy or cross-reactivity risk | Allergy terms; "contraindicated", "cross-reactive", patient allergy history references |
| Other | Interventions not fitting the above (e.g., documentation errors, supply issues, patient counselling) | Catch-all; should be reviewed periodically for emergent sub-categories |

**Label ambiguity note:** A single document may reflect overlapping clinical events (e.g., a dose adjustment prompted by a DDI). The labelling protocol must define a primary-reason rule to ensure consistent single-label assignment. This rule should be documented as part of the annotation guide and applied consistently to ground-truth labels.

**Ground truth source:** Pharmacist-assigned labels from historical records. The quality and consistency of these labels must be assessed before training; inter-annotator agreement (Cohen's Kappa) should be measured on a sample of ≥200 records reviewed by two independent pharmacists.

---

## 3. Label Leakage Risks

Label leakage is the primary data integrity risk in this project. The following sources must be investigated and controlled:

**High risk — direct leakage:**

- **Category keywords embedded in document templates:** If the pharmacy system auto-populates category fields or uses structured templates that include the intervention type as a text heading (e.g., "Intervention Type: Dose Adjustment — ..."), this directly encodes the label. Strip or mask any such fields before feature extraction.
- **Pharmacist abbreviations or codes:** Pharmacists may habitually write shorthand (e.g., "DDI noted", "allergy flag") that directly mirrors the label taxonomy. Audit training documents for such patterns.
- **Auto-generated document metadata:** If the record system attaches category tags as document metadata or in structured fields adjacent to the free text, these must be excluded from model inputs.

**Medium risk — indirect leakage:**

- **Document creation timestamp patterns:** If certain intervention categories are documented at predictable times (e.g., allergy alerts at admission), temporal features could proxy for the label. Avoid using timestamp features unless formally tested for leakage.
- **Pharmacist identifier as a proxy:** If specific pharmacists only document specific intervention types, pharmacist ID would encode label information. Do not include pharmacist ID as a feature.
- **Drug name lists from dispensing system:** Joining specific drug combinations as features is legitimate, but care is needed if the drug-pair lookup itself was derived from the labelled dataset.

**Recommended leakage audit procedure:**
1. Train a simple bag-of-words logistic regression on the raw text and inspect the top 20 features per class.
2. Flag any features that are direct synonyms of the class label.
3. Remove or mask such features and retrain baseline before proceeding.

---

## 4. Suitable Evaluation Metrics

Given the five-class setting with potential class imbalance (the "Other" category is likely over-represented), the following metrics are recommended:

**Primary metrics (must meet success thresholds):**

| Metric | Rationale | Target |
|---|---|---|
| Macro-averaged F1 | Treats all classes equally regardless of frequency; penalises poor performance on minority classes | > 0.80 |
| Macro-averaged ROC-AUC | One-vs-rest AUC averaged across classes; measures ranking quality independent of threshold; robust to imbalance | > 0.85 |
| Per-class sensitivity (recall) | Clinically critical — missed safety-critical interventions (DDI, Allergy Alert) carry patient harm risk | > 0.70 per class |

**Secondary / diagnostic metrics:**

- **Confusion matrix:** Inspect systematic confusions (e.g., DDI vs Therapeutic Substitution) to guide feature engineering.
- **Calibration (Brier score, reliability diagram):** Since the model outputs probabilities for pharmacist review, calibration quality matters. A model with poor calibration may mislead users even if point-prediction accuracy is high.
- **Matthews Correlation Coefficient (MCC):** A single-number multi-class summary metric robust to imbalance; useful for model comparison.
- **Per-class precision:** Monitor precision on Allergy Alert and DDI specifically — false positives on safety-critical classes create alert fatigue and erode trust.

**Baseline comparison:**
All metrics must be compared against a TF-IDF (unigrams + bigrams, top 5,000 features) + Logistic Regression (L2, class-weight balanced) baseline trained and evaluated on identical data splits. This baseline must be documented in any publication.

---

## 5. Potential Biases

**Data representation biases:**

- **Ward/unit skew:** Intervention patterns vary by ward (ICU vs general medicine vs oncology). If historical data overrepresents certain wards, model performance may degrade for underrepresented clinical settings. Stratify performance analysis by ward.
- **Temporal drift:** Clinical practice evolves (new drugs, formulary changes, guideline updates). A model trained on records from 2020–2023 may underperform on 2025 records. Assess label distribution and vocabulary shift over time.
- **Pharmacist writing style variation:** Individual pharmacists may document the same intervention type very differently. If certain pharmacists are overrepresented in training data, the model may learn idiosyncratic language patterns rather than generalising.

**Label quality biases:**

- **Retrospective relabelling:** If historical labels were assigned retrospectively rather than prospectively, they may reflect hindsight knowledge not available at documentation time, artificially inflating apparent predictability.
- **"Other" category underspecification:** This catch-all class likely contains heterogeneous subtypes. If pharmacists inconsistently assign "Other" (versus a specific class), the label noise in this class will propagate into model uncertainty.

**Demographic and equity considerations:**

- **Age and ward as features:** Patient age and ward can be legitimate clinical features but may correlate with demographic groups. Evaluate model performance stratified by age group and ward to detect disparate performance.
- **Language and literacy effects:** If documentation quality varies by patient literacy or care setting, text length and complexity may confound model predictions.

---

## 6. Clinical and Business Usefulness

**Clinical value:**

- Automated categorisation supports prospective safety surveillance — a real-time stream of labelled DDI and Allergy Alert interventions can trigger escalation workflows or dashboard alerts before a pharmacist manually reviews the queue.
- Consistent categorisation removes inter-pharmacist labelling variability from quality audit reports, improving the reliability of workload and safety metrics reported to pharmacy leadership and accreditation bodies.
- Per-class confidence scores allow the model to flag low-confidence predictions for mandatory pharmacist review, creating a human-in-the-loop safety net rather than fully autonomous classification.

**Operational value:**

- Monthly quality reports currently requiring manual document review can be partially automated, reducing pharmacist administrative burden.
- Workload analysis by intervention type (e.g., tracking DDI volume over time) becomes possible at scale without additional effort.
- Longitudinal category trends can surface signals of emerging drug safety issues (e.g., a spike in DDI interventions associated with a newly listed drug).

**Limitations to communicate to stakeholders:**

- The model should be positioned as a decision-support tool, not a replacement for pharmacist judgement. Final categorisation authority remains with the pharmacist.
- "Other" predictions should always be routed to manual review given the class's heterogeneity.
- Model performance should be monitored quarterly and retraining triggered if macro F1 drops more than 5 percentage points from the validation baseline.

---

## 7. Recommended Validation Strategy

**Data splitting:**

Given ~10,000 records and the need for temporally valid evaluation, use a **time-based split** rather than random stratified splitting:

- **Training set:** Earliest 70% of records by document submission date (~7,000 records)
- **Validation set (for hyperparameter tuning):** Next 15% chronologically (~1,500 records)
- **Test set (held-out, evaluated once):** Most recent 15% (~1,500 records)

Rationale: Random splitting would allow future documents to inform training, overstating real-world performance. Temporal splitting simulates the prospective deployment scenario.

**Cross-validation:**

For model selection and hyperparameter tuning, apply **time-series-aware k-fold cross-validation** (e.g., sklearn `TimeSeriesSplit` with 5 folds) on the training set. Do not use standard k-fold, which violates temporal ordering.

**Class imbalance handling:**

- Apply `class_weight='balanced'` in all classifiers.
- If the "Other" class constitutes > 40% of records, consider downsampling it to 2× the second-largest class size for training only (not evaluation).
- Do not apply oversampling (SMOTE) on raw text; if oversampling is used, apply it to embeddings only and only on the training fold.

**Held-out test set protocol:**

- The test set must not be examined until all modelling decisions are finalised.
- Report all primary and secondary metrics on the test set in the final paper.
- Report 95% confidence intervals using bootstrap resampling (n=1,000) on all test set metrics.

---

## 8. Suggested Study Design for Publication-Quality Work

**Recommended study design:** Retrospective development and prospective internal validation study, following TRIPOD reporting guidelines (Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis).

**Development phase:**

1. **Data curation and annotation audit:** Sample 200 records for dual-pharmacist re-labelling. Report inter-annotator agreement (Cohen's Kappa). Resolve disagreements by consensus and update ground truth labels accordingly.
2. **Exploratory analysis:** Report class distribution, document length statistics (tokens, characters), vocabulary size, temporal trends in category frequency, and missing data rates.
3. **Feature engineering and model comparison:** Evaluate the following in order of increasing complexity:
   - Baseline: TF-IDF (1–2 gram) + Logistic Regression
   - TF-IDF + SVM (RBF kernel)
   - TF-IDF + Random Forest / Gradient Boosting
   - Static word embeddings (fastText or GloVe) + LSTM
   - **Recommended primary model:** BioClinicalBERT or PubMedBERT fine-tuned for sequence classification (CPU inference feasible for short documents with quantisation or distillation)
   - If BERT-based models exceed 2-second inference budget, use DistilBERT or a quantised variant

4. **Interpretability implementation:** For the selected model, implement SHAP token-level explanations (for linear/tree models) or attention visualisation (for transformer models). Validate that explanations surface clinically plausible evidence with a pharmacist subject-matter expert.

**Validation phase:**

- Report all metrics on the temporally held-out test set.
- Conduct **subgroup analysis** by class, ward, time period, and document length.
- Report **calibration curves** and Brier score.
- If feasible, conduct a **prospective pilot** over 1–3 months in which model predictions are shown to pharmacists alongside confidence scores, and pharmacist agreement rates are recorded (without modifying pharmacist decisions).

**Ethical and governance requirements:**

- Obtain institutional data governance approval for use of patient-linked records even within the hospital network.
- Document the data lineage, feature set, and model version under which any reported results were produced.
- Pre-register the study protocol (e.g., on OSF or ClinicalTrials.gov) prior to test set evaluation to establish credibility of primary endpoint reporting.

**Recommended manuscript structure:**
Introduction → Methods (data, labelling, model development, validation) → Results (performance table, calibration, subgroup analysis, example explanations) → Discussion (clinical implications, limitations, future work) → Conclusion

**Target journals:** Journal of the American Medical Informatics Association (JAMIA), JAMIA Open, International Journal of Medical Informatics, Applied Clinical Informatics.

---

## Pipeline Metadata

**Pipeline.Domain:** Healthcare / Clinical NLP  
**Pipeline.TaskType:** Multi-class text classification
