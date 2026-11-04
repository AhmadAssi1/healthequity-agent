# Reading Notes

## 1. Obermeyer et al. (2019)

**Title:** Dissecting racial bias in an algorithm used to manage the health of populations

Obermeyer et al. (2019) found that a widely used healthcare algorithm underestimated Black patients’ needs because it predicted healthcare spending rather than illness burden. At the same risk score, Black patients were sicker than White patients, showing that accurate cost predictions do not necessarily lead to fair allocation of care. For HealthEquity Agent, this highlights why we must examine what the prediction target represents and assess performance across demographic groups, alongside overall model metrics.

**Source:** https://doi.org/10.1126/science.aax2342

## 2. 45 CFR §92.210 — Patient Care Decision Support Tools

**Reading date:** 2026-11-02

The text of 45 CFR §92.210 prohibits covered entities from discriminating through patient care decision support tools on the basis of race, color, national origin, sex, age, or disability. It describes an ongoing duty to make reasonable efforts to identify uses of tools employing variables or factors that measure these characteristics and to mitigate the resulting risk of discrimination. HealthEquity Agent can support this review by documenting differences in performance across demographic groups, but a statistical disparity alone does not establish unlawful discrimination and an audit report does not guarantee compliance.

**Source:** https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-A/part-92/subpart-C/section-92.210

**Scope:** These notes summarize the regulatory text reviewed; they do not assess current enforceability or litigation, and the separate HHS Section 1557 overview remains to be read.

## 3. Fairlearn - Fairness Assessment and Common Fairness Metrics

**Reading dates:** 2026-11-02 to 2026-11-03

Fairness assessment starts by identifying potential harms and affected groups, then choosing suitable measures and comparing them across groups using tools such as MetricFrame. Demographic parity compares selection rates, equal opportunity compares true positive rates, and equalized odds compares both true positive and false positive rates; choosing among them requires understanding the intended use and the validity of the recorded outcomes. HealthEquity Agent will report group performance and disparities alongside sample sizes and uncertainty, treating findings as prompts for review rather than proof of discrimination or guarantees of fairness, and will not use the 80% rule as a universal healthcare fairness threshold.

**Sources:**

- https://fairlearn.org/main/user_guide/assessment/index.html
- https://fairlearn.org/main/user_guide/assessment/perform_fairness_assessment.html
- https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html

**Scope:** Assessment workflow, MetricFrame overall and by-group results, differences and ratios, demographic parity, equalized odds, equal opportunity, and limitations of the four-fifths rule. Other assessment chapters have not been marked as completed.
