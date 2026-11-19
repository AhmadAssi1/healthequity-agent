# PDF reporting

Reports can be rendered from completed, saved audit artifacts without another
provider request. Activate the project environment and install requirements first.

```powershell
python -m healthequity.report outputs/agent_runs/RUN_ID --output output/pdf/audit.pdf --model-metadata PATH_TO_MATCHING_METADATA.json
```

For saved evaluation artifacts, add `--scenario CASE_ID`. The optional metadata
must describe the audited sample: its row count must match, and any operating
threshold must be finite and within [0, 1]. Supplied model provenance is labeled
as supplied, rather than independently verified. Missing information stays unknown.

To render after a new audit, add `--pdf` to the existing `healthequity audit`
command. An optional `--model-metadata PATH` supplies the same provenance. The
validated audit completes first; PDF generation invokes the `render_report` tool.
A rendering failure preserves the completed audit and records a separate failed
report status (CLI exit code 3).

The renderer validates structured findings against saved evidence before writing.
It produces a PDF, a standalone responsive HTML companion, and a review JSON
sidecar. Existing destinations are never overwritten. The sidecar retains the
original prose, revised summary and limitations, provenance, and artifact hashes.
The sidecar records human review separately for each edition. New management
reports start with review pending.

The report includes dataset and model context, gaps and bootstrap intervals,
eligible group rates, descriptive TPR/FPR charts, component rate tests, suppression
reasons, quality warnings, limitations, the current methodology appendix, and
provenance. Suppressed groups have no count or rate table entries. Percentages and
percentage-point gaps are labeled separately. Values are formatted from existing
evidence; this is not an independent recalculation of the audit.

Calibration plots are unavailable from saved aggregate evidence: calibration bins
cannot be reconstructed from AUROC or group rates. Raw-score calibration remains
a reporting follow-up. The PDF explicitly discloses this limitation.

## Rendering choice and verification

New management reports use **fpdf2**, the fallback named in the project brief
(page 19). The Windows project environment lacks WeasyPrint's native runtime.
Jinja2 renders the HTML companion from the same validated content. The historical
A4 editions selected by the page-review flags continue to use ReportLab.

The management PDF uses 96 × 176 mm pages with section bookmarks. Each
headline comparison keeps the attribute, measure, gap, confidence interval
and review priority together. Group rates use up to three labelled values
across. The HTML preserves its desktop tables and displays labelled values
in a single column on a narrow screen.

The saved 14,998-stay readmission audit produced
`output/pdf/HealthEquity_Readmission_Mobile.pdf` in 1.80 seconds with
fpdf2 2.9.0. It has 52 short pages and nine bookmarked sections. All 15
tables were checked against the prior management report, including all
12 headline comparisons and 26 eligible group records. Only the renderer
entry in the provenance table changed. All 52 pages were rasterized at
390 pixels wide. Representative results, charts, group values and provenance
were also rendered at 360 pixels wide and at a 720-pixel laptop viewing
height. At fit-to-width the gap values measure 13.9 pixels on a 360-pixel
screen and 15.0 pixels on a 390-pixel screen. The checks used simulated
viewports. Generation reads saved audit artifacts locally.

Ahmad Assi visually reviewed and approved this delivered edition on November 19,
2026. The approval is recorded in its review sidecar and the
[mobile report review](reviews/mobile-report-review-20261119.md).

The first full draft uses the completed 14,998-row audit in
`outputs/agent_runs/714405af53d549fba5a0a105c13a966d`, preserving its original
November 12, 2026 source timestamp. The November 17 draft contains 12 pages and
rendered locally in 0.795 seconds. All pages were rasterized and visually checked.
This timing measures saved-artifact reporting, not a new end-to-end audit.

Automated tests cover evidence validation, suppression, escaping, provenance,
metadata validation, failed-render cleanup, overwrite protection, saved-run
rendering, and the audit CLI's separate report status. Human review of the final
PDF is required for each new report; approval is never inherited from an older audit.

## Reviewed management edition (November 17, 2026)

Ahmad Assi reviewed and approved the management pages and visually inspected the
technical appendix. Methodology was compared with the implementation. The saved
audit's metrics, all 12 bootstrap intervals, and all 16 rate tests were reproduced
from the 14,998-row sample with identical structured results.

The reviewed edition uses clearer terminology, descriptive charts in separate
panels, age ordering, an executive summary, and a concise interpretation-limits
page. Detailed tests and full limitations remain in the appendix. Actual audit
settings are distinguished from documented defaults. The review record states
author editorial/visual approval, not independent clinical or legal certification.

```powershell
python -m healthequity.report outputs/agent_runs/714405af53d549fba5a0a105c13a966d --output output/pdf/HealthEquity_Management_Report.pdf --model-metadata data/samples/diabetes_audit_sample_metadata.json --review-complete
```

Use a fresh output name if it already exists. `--review-complete` applies only to
this reviewed audit and its pinned source trace/findings hashes. It cannot approve
a new dataset. Ordinary reports remain drafts. The management prose and reviewed
page overrides are specific to this readmission audit. The November 19 generic
management template below handles new binary tasks without these overrides.
Calibration plotting remains incomplete.

## General management template (November 19, 2026)

Ordinary saved-run and `audit --pdf` reports now use a task-neutral management
template by default. Executive summaries, observed group-rate ranges, severity
counts, test interpretations, tables, chart axes and exclusions are derived from
validated evidence. They do not assume hospital stays, readmission, race, age,
gender or payer columns. Missing task information is explicitly unavailable.

Optional metadata fields:

| Field | Meaning |
|---|---|
| `row_count` | Required when supplying metadata; must match the audited sample |
| `task_type` | If supplied, must be `binary_classification` |
| `task_name` | Supplied task title; used in report title and executive summary |
| `positive_class` | Meaning of outcome 1; otherwise not supplied |
| `observation_unit` | Plural unit description, such as `applications` |
| `model_name`, `model_run_id` | Supplied model provenance |
| `decision_threshold` | Optional finite number in [0, 1]; not inferred or reapplied |
| `group_display_names` | Unique display names keyed by actual mapped group columns |

See `docs/examples/report-metadata.json` for a synthetic non-healthcare example.
Replace all example values to describe the actual audit; these are supplied
descriptions rather than independently verified task semantics. Regression and
multiclass metadata are rejected. Binary label validation remains mandatory.

Charts stack the TPR and FPR panels, annotate values, and share a scale across
each attribute (0-50% or 0-100%, determined by the actual rates). Each chart
shows up to four groups. Numeric interval labels are sorted by their
lower bound; other source labels are preserved. Unavailable rates remain
unavailable, including groups with no positive predictions or no negative outcomes.

The new management wording and full limitations are retained in the review
sidecar. The domain-neutral methodology presentation retains its source hash.
The PDF and HTML open with the audit title and executive summary. The sidecar
records the review status for each new report.
