"""Structured output schemas for fairness audits."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class AuditBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class MetricResult(AuditBaseModel):
    value: float | None = Field(ge=0, le=1)
    reason: str | None = Field(
        min_length=1,
        description=(
            "Required unavailability reason: null when value is available; "
            "otherwise explain why value is null. Never use for commentary."
        ),
    )

    @model_validator(mode="after")
    def validate_availability(self):
        if self.value is None:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "An unavailable metric requires a reason."
                )
        elif self.reason is not None:
            raise ValueError(
                "An available metric must not have an unavailable reason."
            )

        return self

class BootstrapConfidenceInterval(AuditBaseModel):
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    valid_resamples: int = Field(ge=1)
    total_resamples: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.lower > self.upper:
            raise ValueError(
                "The lower bound must not exceed the upper bound."
            )

        if self.valid_resamples > self.total_resamples:
            raise ValueError(
                "Valid resamples must not exceed total resamples."
            )

        return self


class GroupMetrics(AuditBaseModel):
    group_value: str | int | float | bool | None
    count: int = Field(ge=1)

    base_rate: MetricResult
    selection_rate: MetricResult
    tpr: MetricResult
    fpr: MetricResult
    ppv: MetricResult
    auroc: MetricResult


class SuppressedGroup(AuditBaseModel):
    group_value: str | int | float | bool | None
    suppression_reason: str = Field(min_length=1)

    @field_validator("suppression_reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "A suppressed group requires a non-blank reason."
            )

        return value

class FairnessFinding(MetricResult):
    metric: Literal[
        "demographic_parity_difference",
        "equalized_odds_difference",
        "ppv_difference",
    ]
    confidence_interval: BootstrapConfidenceInterval | None
    confidence_interval_reason: str | None = Field(
        min_length=1,
        description=(
            "Required unavailability reason: null when confidence_interval "
            "is available; otherwise explain why it is null."
        ),
    )
    severity: Literal[
        "no_evidence_found",
        "watch",
        "substantial",
        "unavailable",
    ]

    @model_validator(mode="after")
    def validate_finding(self):
        interval = self.confidence_interval
        interval_reason = self.confidence_interval_reason

        if interval is None:
            if interval_reason is None or not interval_reason.strip():
                raise ValueError(
                    "An unavailable interval requires a reason."
                )
        elif interval_reason is not None:
            raise ValueError(
                "An available interval must not have an unavailable reason."
            )

        if self.value is None and interval is not None:
            raise ValueError(
                "An unavailable gap must not have a confidence interval."
            )

        can_classify = self.value is not None and interval is not None
        if can_classify == (self.severity == "unavailable"):
            raise ValueError(
                "Severity availability must match gap and interval availability."
            )

        return self


class StatisticalTestResult(AuditBaseModel):
    rate: Literal["selection_rate", "tpr", "fpr", "ppv"]
    method: Literal[
        "two_proportion_z",
        "chi_square",
        "fisher_exact",
        "fisher_monte_carlo",
        "unavailable",
    ]
    statistic: float | None
    p_value: float | None = Field(ge=0, le=1)
    min_expected_count: float | None = Field(ge=0)
    reason: str | None = Field(
        min_length=1,
        description=(
            "Required unavailability reason: null for every available test; "
            "explain why the test is unavailable only when method is unavailable. "
            "Never use for interpretation or commentary."
        ),
    )

    @model_validator(mode="after")
    def validate_test_result(self):
        if self.method == "unavailable":
            if self.p_value is not None or self.statistic is not None:
                raise ValueError(
                    "An unavailable test must not contain test results."
                )

            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "An unavailable test requires a reason."
                )
        else:
            if self.min_expected_count is None:
                raise ValueError("An available test requires expected cell counts.")
            if self.p_value is None:
                raise ValueError(
                    "An available test requires a p-value."
                )

            if self.reason is not None:
                raise ValueError(
                    "An available test must not have an unavailable reason."
                )

            if (
                self.method in ("two_proportion_z", "chi_square")
                and self.statistic is None
            ):
                raise ValueError(
                    "Z and chi-square tests require a statistic."
                )

        return self

class AttributeFindings(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    groups: list[GroupMetrics]
    suppressed_groups: list[SuppressedGroup]
    fairness_findings: list[FairnessFinding] = Field(
        min_length=3,
        max_length=3,
    )
    statistical_tests: list[StatisticalTestResult]

    @model_validator(mode="after")
    def validate_attribute_findings(self):
        metrics = [
            finding.metric for finding in self.fairness_findings
        ]
        if len(set(metrics)) != 3:
            raise ValueError(
                "Each headline fairness metric must appear exactly once."
            )

        rates = [test.rate for test in self.statistical_tests]
        if len(rates) != len(set(rates)):
            raise ValueError(
                "Each rate must have at most one statistical test."
            )

        identities = [
            (type(group.group_value), group.group_value)
            for group in [*self.groups, *self.suppressed_groups]
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "Groups must be unique and cannot also be suppressed."
            )

        if len(self.groups) < 2:
            if any(
                finding.value is not None
                for finding in self.fairness_findings
            ):
                raise ValueError(
                    "Fairness comparisons require two eligible groups."
                )

            if any(
                test.method != "unavailable"
                for test in self.statistical_tests
            ):
                raise ValueError(
                    "Statistical comparisons require two eligible groups."
                )

        return self

class DatasetSummary(AuditBaseModel):
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    group_columns: list[str] = Field(min_length=1)
    score_available: bool

    @field_validator("group_columns")
    @classmethod
    def validate_group_columns(cls, value: list[str]) -> list[str]:
        if any(not name.strip() for name in value):
            raise ValueError(
                "Group column names must not be blank."
            )

        if len(value) != len(set(value)):
            raise ValueError(
                "Group column names must be unique."
            )

        return value

    @model_validator(mode="after")
    def validate_column_count(self):
        if len(self.group_columns) > self.column_count:
            raise ValueError(
                "Group columns must not exceed the dataset column count."
            )

        return self

class AuditFindings(AuditBaseModel):
    dataset: DatasetSummary
    attributes: list[AttributeFindings] = Field(min_length=1)
    summary: str = Field(min_length=1, pattern=r"\S")
    warnings: list[str]
    limitations: list[str] = Field(min_length=1)

    @field_validator("warnings", "limitations")
    @classmethod
    def validate_messages(cls, value: list[str]) -> list[str]:
        if any(not message.strip() for message in value):
            raise ValueError(
                "Warnings and limitations must not contain blank messages."
            )

        return value

    @model_validator(mode="after")
    def validate_audit_consistency(self):
        names = [item.attribute for item in self.attributes]

        if len(names) != len(set(names)):
            raise ValueError(
                "Each group attribute must appear exactly once."
            )

        if set(names) != set(self.dataset.group_columns):
            raise ValueError(
                "Attribute findings must match the declared group columns."
            )

        for attribute in self.attributes:
            if sum(group.count for group in attribute.groups) > (
                self.dataset.row_count
            ):
                raise ValueError(
                    "Eligible group counts must not exceed the row count."
                )

            if not self.dataset.score_available and any(
                group.auroc.value is not None
                for group in attribute.groups
            ):
                raise ValueError(
                    "AUROC must be unavailable without scores."
                )

        return self

    
