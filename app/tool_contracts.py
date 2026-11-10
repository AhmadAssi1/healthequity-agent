"""Provider-independent, JSON-only input/output contracts for the nine tools."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, model_validator

from app.audit_schemas import (
    AuditBaseModel,
    AuditFindings,
    GroupMetrics,
    MetricResult,
    StatisticalTestResult,
)
from app.tool_schemas import (
    AgentAttributeProfile,
    AgentDatasetProfile,
    AnalysisPlan,
    SuppressedGroupSummary,
)
from fairness.policy import SeverityPolicy, SuppressionPolicy

Name = Annotated[str, Field(min_length=1, pattern=r"\S")]
Rate = Literal["selection_rate", "tpr", "fpr", "ppv"]
HeadlineMetric = Literal[
    "demographic_parity_difference", "equalized_odds_difference", "ppv_difference",
]
TestMethod = Literal[
    "two_proportion_z", "chi_square", "fisher_exact", "fisher_monte_carlo",
]


class EmptyToolInput(AuditBaseModel):
    """Dataset, mappings, and policies come from the bound Python session."""


class AttributeToolInput(AuditBaseModel):
    attributes: list[Name] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def unique_attributes(self):
        if self.attributes is not None and len(set(self.attributes)) != len(self.attributes):
            raise ValueError("Requested attributes must be unique.")
        return self


class SelectAnalysesInput(AuditBaseModel):
    proposed_plan: dict[str, JsonValue] | None = None


class SignificanceTestInput(AttributeToolInput):
    rates: list[Rate] | None = Field(default=None, min_length=1)
    test_name: TestMethod | None = None
    n_resamples: int = Field(default=10_000, ge=1, le=100_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def validate_selection(self):
        if self.rates is not None and len(set(self.rates)) != len(self.rates):
            raise ValueError("Requested rates must be unique.")
        if self.test_name is not None and (
            self.attributes is None or len(self.attributes) != 1
            or self.rates is None or len(self.rates) != 1
        ):
            raise ValueError("An explicit test name requires one attribute and one rate.")
        return self


class BootstrapCIInput(AttributeToolInput):
    metrics: list[HeadlineMetric] | None = Field(default=None, min_length=1)
    n_resamples: int = Field(default=1000, ge=1, le=100_000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def unique_metrics(self):
        if self.metrics is not None and len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Requested metrics must be unique.")
        return self


class GroupSizesOutput(AuditBaseModel):
    suppression_policy: SuppressionPolicy
    attributes: list[AgentAttributeProfile] = Field(min_length=1)


class HeadlineGap(MetricResult):
    metric: HeadlineMetric


class AttributeMetricsOutput(AuditBaseModel):
    attribute: Name
    groups: list[GroupMetrics]
    gaps: list[HeadlineGap] = Field(min_length=3, max_length=3)
    suppressed_groups: list[SuppressedGroupSummary]

    @model_validator(mode="after")
    def unique_gaps(self):
        if len({gap.metric for gap in self.gaps}) != 3:
            raise ValueError("Each headline gap must appear once.")
        return self


class FairnessMetricsOutput(AuditBaseModel):
    results: list[AttributeMetricsOutput] = Field(min_length=1)


class SignificanceResult(StatisticalTestResult):
    attribute: Name
    min_expected_count: float | None = Field(ge=0)

    @model_validator(mode="after")
    def require_expected_counts(self):
        if self.method != "unavailable" and self.min_expected_count is None:
            raise ValueError("Available tests require expected cell counts.")
        return self


class SignificanceOutput(AuditBaseModel):
    results: list[SignificanceResult] = Field(min_length=1)


class BootstrapResult(AuditBaseModel):
    attribute: Name
    metric: HeadlineMetric
    observed: float | None = Field(ge=0, le=1)
    lower: float | None = Field(ge=0, le=1)
    upper: float | None = Field(ge=0, le=1)
    valid_resamples: int = Field(ge=0)
    total_resamples: int = Field(ge=1)
    confidence_level: Literal[0.95] = 0.95
    seed: int = Field(ge=0, le=2**32 - 1)
    reason: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_interval(self):
        if self.valid_resamples > self.total_resamples:
            raise ValueError("Valid resamples cannot exceed total resamples.")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Both confidence bounds must be available or unavailable.")
        if self.lower is None:
            if self.reason is None or not self.reason.strip():
                raise ValueError("Unavailable intervals require a reason.")
        else:
            if self.observed is None or self.valid_resamples == 0:
                raise ValueError("An interval requires an observed gap and valid resamples.")
            if self.lower > self.upper or self.reason is not None:
                raise ValueError("Available intervals need ordered bounds and no reason.")
        return self


class BootstrapOutput(AuditBaseModel):
    results: list[BootstrapResult] = Field(min_length=1)


class AuditEvidence(AuditBaseModel):
    profile: AgentDatasetProfile
    plan: AnalysisPlan
    severity_policy: SeverityPolicy
    metrics: FairnessMetricsOutput
    significance: SignificanceOutput
    bootstrap: BootstrapOutput


class DraftFindingsOutput(AuditBaseModel):
    findings: AuditFindings


class ValidateClaimsInput(AuditBaseModel):
    findings: AuditFindings


class ClaimIssue(AuditBaseModel):
    code: Literal[
        "unsupported_number", "unsupported_group", "causal_language",
        "missing_insufficient_evidence", "evidence_mismatch",
    ]
    message: Name


class ClaimsValidationOutput(AuditBaseModel):
    passed: bool
    issues: list[ClaimIssue]
    scope: Literal["structured_evidence_and_english_rules"] = "structured_evidence_and_english_rules"
    requires_human_review: Literal[True] = True

    @model_validator(mode="after")
    def consistent_verdict(self):
        if self.passed != (not self.issues):
            raise ValueError("A passing verdict requires no issues; a failure requires issues.")
        return self


class RenderReportOutput(AuditBaseModel):
    pdf_path: Name
    run_id: Name
    trace_id: Name


class ToolDefinition(AuditBaseModel):
    name: Name
    description: Name
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    available: bool
