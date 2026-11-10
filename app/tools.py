"""Nine validated tool interfaces bound to one in-memory audit dataset."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.agent_profile import build_agent_profile
from app.analysis_counts import compute_analysis_counts
from app.analysis_selection import _rate_counts, select_analyses
from app.audit_schemas import AuditFindings, GroupMetrics, MetricResult
from app.claims_validation import validate_claims
from app.plan_guard import guard_analysis_plan
from app.profiling import profile_dataset
from app.schemas import ColumnMapping
from app.tool_contracts import (
    AttributeMetricsOutput,
    AttributeToolInput,
    AuditEvidence,
    BootstrapCIInput,
    BootstrapOutput,
    BootstrapResult,
    ClaimsValidationOutput,
    DraftFindingsOutput,
    EmptyToolInput,
    FairnessMetricsOutput,
    GroupSizesOutput,
    HeadlineGap,
    RenderReportOutput,
    SelectAnalysesInput,
    SignificanceOutput,
    SignificanceResult,
    SignificanceTestInput,
    ToolDefinition,
    ValidateClaimsInput,
)
from app.tool_schemas import AgentDatasetProfile, GuardedAnalysisPlan
from fairness.metrics import compute_fairness_gaps, compute_group_metrics
from fairness.policy import (
    SeverityPolicy,
    SuppressionPolicy,
    load_severity_policy,
    load_suppression_policy,
)
from fairness.stats import bootstrap_fairness_intervals, compare_proportions

RATES = ("selection_rate", "tpr", "fpr", "ppv")
METRICS = (
    "demographic_parity_difference", "equalized_odds_difference", "ppv_difference",
)


class ToolExecutionError(ValueError):
    """A tool request conflicts with the audit context or its prerequisites."""


class ToolUnavailableError(ToolExecutionError):
    """The required implementation has not been connected."""


@dataclass(frozen=True)
class ToolBackends:
    draft_findings: Callable[[AuditEvidence], AuditFindings] | None = None
    validate_claims: Callable[[AuditFindings, AuditEvidence], ClaimsValidationOutput] | None = validate_claims
    render_report: Callable[[AuditFindings], RenderReportOutput] | None = None


TOOL_SPECS = {
    "profile_dataset": (
        EmptyToolInput, AgentDatasetProfile,
        "Return the policy-filtered dataset profile; patient rows stay in Python.",
    ),
    "check_group_sizes": (
        EmptyToolInput, GroupSizesOutput,
        "Return reportable groups and suppressed group reason codes, without suppressed counts.",
    ),
    "select_analyses": (
        SelectAnalysesInput, GuardedAnalysisPlan,
        "Build or correct a proposed analysis plan using trusted group counts.",
    ),
    "compute_fairness_metrics": (
        AttributeToolInput, FairnessMetricsOutput,
        "Compute per-group rates and the three headline gaps for selected or all attributes.",
    ),
    "run_significance_test": (
        SignificanceTestInput, SignificanceOutput,
        "Run permitted rate tests; omitted attributes and rates select all. No combined EOD p-value.",
    ),
    "bootstrap_ci": (
        BootstrapCIInput, BootstrapOutput,
        "Compute seeded 95% bootstrap intervals for selected or all headline metrics and attributes.",
    ),
    "draft_findings": (
        EmptyToolInput, DraftFindingsOutput,
        "Ask the configured drafting backend for findings using complete aggregate evidence.",
    ),
    "validate_claims": (
        ValidateClaimsInput, ClaimsValidationOutput,
        "Check submitted findings against trusted aggregate evidence using the configured validator.",
    ),
    "render_report": (
        EmptyToolInput, RenderReportOutput,
        "Render only findings that passed claim validation in this session.",
    ),
}


class AuditToolSession:
    """One dataset snapshot and policy snapshot, with no dataframe arguments over JSON."""

    def __init__(
        self, df: pd.DataFrame, mapping: ColumnMapping,
        suppression_policy: SuppressionPolicy | None = None,
        severity_policy: SeverityPolicy | None = None,
        backends: ToolBackends | None = None,
    ):
        self._df = df.copy(deep=True)
        self._mapping = mapping.model_copy(deep=True)
        self._policy = (
            suppression_policy if suppression_policy is not None else load_suppression_policy()
        ).model_copy(deep=True)
        self._severity = (
            severity_policy if severity_policy is not None else load_severity_policy()
        ).model_copy(deep=True)
        self._profile = profile_dataset(self._df, self._mapping)
        self._agent_profile = build_agent_profile(self._profile, self._policy)
        self._backends = backends or ToolBackends()
        self._analysis_counts = None
        self._metrics = {}
        self._significance = {}
        self._bootstrap = {}
        self._validated_findings = None

    def definitions(self) -> list[ToolDefinition]:
        """Export provider-neutral JSON Schemas; provider translation is separate."""
        return [ToolDefinition(
            name=name, description=description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            available=(
                getattr(self._backends, name) is not None
                if name in ("draft_findings", "validate_claims", "render_report") else True
            ),
        ) for name, (input_model, output_model, description) in TOOL_SPECS.items()]

    def invoke(self, name: str, arguments: dict):
        if name not in TOOL_SPECS:
            raise ToolExecutionError(f"Unknown audit tool: {name}")
        input_model, output_model, _ = TOOL_SPECS[name]
        request = input_model.model_validate(arguments)
        result = getattr(self, f"_tool_{name}")(request)
        # Validate serialized output too; model instances can have been mutated by a backend.
        payload = result.model_dump(mode="python") if hasattr(result, "model_dump") else result
        return output_model.model_validate(payload)

    def _counts(self):
        if self._mapping.prediction is None:
            raise ToolExecutionError("Analysis tools require mapped predictions.")
        if self._analysis_counts is None:
            self._analysis_counts = compute_analysis_counts(self._df, self._mapping, self._policy)
        return self._analysis_counts

    def _names(self, request):
        names = self._mapping.groups if request.attributes is None else request.attributes
        if any(name not in self._mapping.groups for name in names):
            raise ToolExecutionError("Only mapped group attributes may be analyzed.")
        return names

    def _plan(self):
        return select_analyses(self._counts(), self._mapping.score is not None)

    def _eligible_frame(self, attribute):
        """Integer group codes prevent pandas from merging categories such as True and 1."""
        profile = next(item for item in self._agent_profile.attributes if item.attribute == attribute)
        labels = {i: item.group_value for i, item in enumerate(profile.reportable_groups)}
        identities = {(type(label), label): code for code, label in labels.items()}
        codes = []
        for label in self._df[attribute]:
            if pd.isna(label):
                label = None
            elif isinstance(label, np.generic):
                label = label.item()
            codes.append(identities.get((type(label), label), -1))
        frame = pd.DataFrame({
            "outcome": self._df[self._mapping.outcome].to_numpy(),
            "prediction": self._df[self._mapping.prediction].to_numpy(),
            "_group": codes,
        })
        if self._mapping.score is not None:
            frame["score"] = self._df[self._mapping.score].to_numpy()
        return frame.loc[frame["_group"] >= 0].copy(), labels, profile

    def _tool_profile_dataset(self, request):
        return self._agent_profile.model_copy(deep=True)

    def _tool_check_group_sizes(self, request):
        return GroupSizesOutput(
            suppression_policy=self._policy.model_copy(deep=True),
            attributes=[item.model_copy(deep=True) for item in self._agent_profile.attributes],
        )

    def _tool_select_analyses(self, request):
        expected = self._plan()
        self._validated_findings = None
        return guard_analysis_plan(
            expected if request.proposed_plan is None else request.proposed_plan,
            self._counts(), self._mapping.score is not None,
        )

    def _tool_compute_fairness_metrics(self, request):
        self._counts()
        results = []
        for name in self._names(request):
            frame, labels, profile = self._eligible_frame(name)
            groups, gaps = [], []
            if frame.empty:
                gaps = [HeadlineGap(
                    metric=metric, value=None,
                    reason="At least two eligible groups are required.",
                ) for metric in METRICS]
            else:
                table = compute_group_metrics(
                    frame, "_group", "score" if self._mapping.score is not None else None,
                )
                comparisons = compute_fairness_gaps(table)
                for code, row in table.iterrows():
                    values = {}
                    for metric in ("base_rate", "selection_rate", "tpr", "fpr", "ppv", "auroc"):
                        if metric not in row:
                            values[metric] = MetricResult(value=None, reason="No score column was provided.")
                        elif pd.isna(row[metric]):
                            values[metric] = MetricResult(
                                value=None,
                                reason=comparisons["undefined_metric_reasons"].loc[code, metric],
                            )
                        else:
                            values[metric] = MetricResult(value=float(row[metric]), reason=None)
                    groups.append(GroupMetrics(
                        group_value=labels[int(code)], count=int(row["count"]), **values,
                    ))
                for metric in METRICS:
                    gap = comparisons["gaps"].loc[metric]
                    gaps.append(HeadlineGap(
                        metric=metric,
                        value=None if pd.isna(gap["value"]) else float(gap["value"]),
                        reason=None if pd.isna(gap["reason"]) else gap["reason"],
                    ))
            results.append(AttributeMetricsOutput(
                attribute=name, groups=groups, gaps=gaps,
                suppressed_groups=[item.model_copy(deep=True) for item in profile.suppressed_groups],
            ))
        self._validated_findings = None
        self._metrics.update({item.attribute: item for item in results})
        return FairnessMetricsOutput(results=results)

    def _tool_run_significance_test(self, request):
        plan = {item.attribute: item for item in self._plan().attributes}
        counts = {item.attribute: item for item in self._counts()}
        results = []
        for name in self._names(request):
            tests = {item.rate: item for item in plan[name].statistical_tests}
            for rate in request.rates or RATES:
                planned = tests[rate]
                if request.test_name is not None and request.test_name != planned.method:
                    raise ToolExecutionError(f"Requested test is not permitted; expected {planned.method}.")
                if planned.method == "unavailable":
                    result = SignificanceResult(
                        attribute=name, rate=rate, method="unavailable", statistic=None,
                        p_value=None, min_expected_count=planned.min_expected_count,
                        reason=planned.reason,
                    )
                else:
                    numerators, totals = zip(*[
                        _rate_counts(group, rate) for group in counts[name].groups
                    ], strict=True)
                    tested = compare_proportions(
                        numerators, totals, n_resamples=request.n_resamples, seed=request.seed,
                    )
                    if tested.method != planned.method:
                        raise ToolExecutionError("The statistical engine disagrees with the approved plan.")
                    result = SignificanceResult(
                        attribute=name, rate=rate, method=tested.method,
                        statistic=tested.statistic, p_value=tested.p_value,
                        min_expected_count=tested.min_expected_count, reason=tested.reason,
                    )
                results.append(result)
        self._validated_findings = None
        self._significance.update({(item.attribute, item.rate): item for item in results})
        return SignificanceOutput(results=results)

    def _tool_bootstrap_ci(self, request):
        plans = {item.attribute: item for item in self._plan().attributes}
        results = []
        for name in self._names(request):
            frame, _, _ = self._eligible_frame(name)
            planned = {item.metric: item for item in plans[name].fairness_metrics}
            intervals = (
                bootstrap_fairness_intervals(
                    frame, "_group", n_resamples=request.n_resamples, seed=request.seed,
                ) if plans[name].eligible_group_count >= 2 else {}
            )
            for metric in request.metrics or METRICS:
                interval = intervals.get(metric)
                reason = None
                if not planned[metric].available:
                    reason = planned[metric].reason
                elif interval.lower is None:
                    reason = "No valid bootstrap interval could be computed."
                results.append(BootstrapResult(
                    attribute=name, metric=metric,
                    observed=None if interval is None else interval.observed,
                    lower=None if interval is None else interval.lower,
                    upper=None if interval is None else interval.upper,
                    valid_resamples=0 if interval is None else interval.valid_resamples,
                    total_resamples=request.n_resamples, seed=request.seed, reason=reason,
                ))
        self._validated_findings = None
        self._bootstrap.update({(item.attribute, item.metric): item for item in results})
        return BootstrapOutput(results=results)

    def _evidence(self):
        names = self._mapping.groups
        if (
            set(self._metrics) != set(names)
            or set(self._significance) != {(name, rate) for name in names for rate in RATES}
            or set(self._bootstrap) != {(name, metric) for name in names for metric in METRICS}
        ):
            raise ToolExecutionError("Compute metrics, all rate tests, and all bootstrap gaps before drafting or validation.")
        return AuditEvidence(
            profile=self._agent_profile, plan=self._plan(), severity_policy=self._severity,
            metrics=FairnessMetricsOutput(results=[self._metrics[name] for name in names]),
            significance=SignificanceOutput(results=[self._significance[name, rate] for name in names for rate in RATES]),
            bootstrap=BootstrapOutput(results=[self._bootstrap[name, metric] for name in names for metric in METRICS]),
        ).model_copy(deep=True)

    def _tool_draft_findings(self, request):
        self._validated_findings = None
        if self._backends.draft_findings is None:
            raise ToolUnavailableError("A drafting backend is not configured.")
        findings = self._backends.draft_findings(self._evidence())
        payload = findings.model_dump() if hasattr(findings, "model_dump") else findings
        return DraftFindingsOutput(findings=AuditFindings.model_validate(payload))

    def _tool_validate_claims(self, request):
        self._validated_findings = None
        if self._backends.validate_claims is None:
            raise ToolUnavailableError("A claim-validation backend is not configured.")
        verdict = self._backends.validate_claims(request.findings.model_copy(deep=True), self._evidence())
        payload = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict
        verdict = ClaimsValidationOutput.model_validate(payload)
        if verdict.passed:
            self._validated_findings = request.findings.model_copy(deep=True)
        return verdict

    def _tool_render_report(self, request):
        if self._backends.render_report is None:
            raise ToolUnavailableError("A report-rendering backend is not configured.")
        if self._validated_findings is None:
            raise ToolExecutionError("Report rendering requires findings that passed claim validation in this session.")
        return self._backends.render_report(self._validated_findings.model_copy(deep=True))

    def get_evidence(self) -> AuditEvidence:
        return self._evidence()
