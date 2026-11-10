"""Typed contracts for audit tools."""
from math import isclose
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from app.audit_schemas import AuditBaseModel
from fairness.policy import SuppressionPolicy


class GroupCounts(AuditBaseModel):
    group_value: str | int | float | bool | None
    count: int = Field(ge=1)
    positive_outcomes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self):
        if self.positive_outcomes > self.count:
            raise ValueError(
                "Positive outcomes must not exceed the group count."
            )

        return self

class AttributeProfile(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    groups: list[GroupCounts] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_groups(self):
        identities = [
            (type(group.group_value), group.group_value)
            for group in self.groups
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "Each group must appear exactly once per attribute."
            )

        return self

class DatasetProfile(AuditBaseModel):
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    positive_outcomes: int = Field(ge=0)
    outcome_prevalence: float = Field(ge=0, le=1)
    score_available: bool
    attributes: list[AttributeProfile] = Field(min_length=1)
    data_quality_flags: list[str]

    @model_validator(mode="after")
    def validate_profile(self):
        if self.positive_outcomes > self.row_count:
            raise ValueError(
                "Positive outcomes must not exceed the row count."
            )

        expected_prevalence = self.positive_outcomes / self.row_count
        if not isclose(
            self.outcome_prevalence,
            expected_prevalence,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Outcome prevalence must match the outcome counts."
            )

        names = [item.attribute for item in self.attributes]
        if len(names) != len(set(names)):
            raise ValueError(
                "Attribute names must be unique."
            )

        if len(names) > self.column_count:
            raise ValueError(
                "Group attributes must not exceed the column count."
            )

        for attribute in self.attributes:
            if sum(group.count for group in attribute.groups) != self.row_count:
                raise ValueError(
                    "Each attribute must account for every row."
                )

            if sum(
                group.positive_outcomes for group in attribute.groups
            ) != self.positive_outcomes:
                raise ValueError(
                    "Each attribute must account for every positive outcome."
                )

        if any(not flag.strip() for flag in self.data_quality_flags):
            raise ValueError(
                "Data quality flags must not contain blank messages."
            )

        return self

class GroupEligibility(AuditBaseModel):
    group_value: str | int | float | bool | None
    reportable: bool
    suppression_reason: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self):
        if self.reportable:
            if self.suppression_reason is not None:
                raise ValueError(
                    "A reportable group must not have a suppression reason."
                )
        elif (
            self.suppression_reason is None
            or not self.suppression_reason.strip()
        ):
            raise ValueError(
                "A suppressed group requires a non-blank reason."
            )

        return self

class AttributeEligibility(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    groups: list[GroupEligibility] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_groups(self):
        identities = [
            (type(group.group_value), group.group_value)
            for group in self.groups
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "Each group must have exactly one eligibility decision."
            )

        return self

class GroupSizeCheckResult(AuditBaseModel):
    suppression_policy: SuppressionPolicy
    attributes: list[AttributeEligibility] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_attributes(self):
        names = [item.attribute for item in self.attributes]

        if len(names) != len(set(names)):
            raise ValueError(
                "Each attribute must have exactly one eligibility result."
            )

        return self

class SuppressedGroupSummary(AuditBaseModel):
    group_value: str | int | float | bool | None
    reason_codes: list[
        Literal[
            "below_min_group_size",
            "below_min_positive_outcomes",
        ]
    ] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def validate_unique_reasons(self):
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError(
                "Suppression reason codes must be unique."
            )

        return self

class AgentAttributeProfile(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    reportable_groups: list[GroupCounts]
    suppressed_groups: list[SuppressedGroupSummary]

    @model_validator(mode="after")
    def validate_groups(self):
        all_groups = [
            *self.reportable_groups,
            *self.suppressed_groups,
        ]

        if not all_groups:
            raise ValueError(
                "An attribute must contain at least one group."
            )

        identities = [
            (type(group.group_value), group.group_value)
            for group in all_groups
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "Groups must be unique and cannot also be suppressed."
            )

        return self

class AgentDatasetProfile(AuditBaseModel):
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    outcome_prevalence: float = Field(ge=0, le=1)
    score_available: bool
    suppression_policy: SuppressionPolicy
    attributes: list[AgentAttributeProfile] = Field(min_length=1)
    data_quality_flags: list[str]

    @model_validator(mode="after")
    def validate_agent_profile(self):
        names = [item.attribute for item in self.attributes]

        if len(names) != len(set(names)):
            raise ValueError(
                "Attribute names must be unique."
            )

        if len(names) > self.column_count:
            raise ValueError(
                "Group attributes must not exceed the column count."
            )

        for attribute in self.attributes:
            reported_count = sum(
                group.count for group in attribute.reportable_groups
            )

            if reported_count > self.row_count:
                raise ValueError(
                    "Reported group counts must not exceed the row count."
                )

            if (
                not attribute.suppressed_groups
                and reported_count != self.row_count
            ):
                raise ValueError(
                    "Without suppressed groups, every row must be accounted for."
                )

            for group in attribute.reportable_groups:
                if (
                    group.count < self.suppression_policy.min_group_size
                    or group.positive_outcomes
                    < self.suppression_policy.min_positive_outcomes
                ):
                    raise ValueError(
                        "Reported groups must meet the suppression policy."
                    )

        if any(not flag.strip() for flag in self.data_quality_flags):
            raise ValueError(
                "Data quality flags must not contain blank messages."
            )

        return self

class ConfusionCounts(AuditBaseModel):
    tn: int = Field(ge=0)
    fp: int = Field(ge=0)
    fn: int = Field(ge=0)
    tp: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_nonempty_group(self):
        if self.tn + self.fp + self.fn + self.tp == 0:
            raise ValueError(
                "Confusion counts must represent at least one observation."
            )

        return self

class GroupAnalysisCounts(GroupCounts):
    confusion: ConfusionCounts

    @model_validator(mode="after")
    def validate_confusion_counts(self):
        total = (
            self.confusion.tn
            + self.confusion.fp
            + self.confusion.fn
            + self.confusion.tp
        )

        if total != self.count:
            raise ValueError(
                "Confusion counts must sum to the group count."
            )

        actual_positives = self.confusion.tp + self.confusion.fn

        if actual_positives != self.positive_outcomes:
            raise ValueError(
                "TP plus FN must equal the positive outcome count."
            )

        return self

class AttributeAnalysisCounts(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    groups: list[GroupAnalysisCounts]

    @model_validator(mode="after")
    def validate_unique_groups(self):
        identities = [
            (type(group.group_value), group.group_value)
            for group in self.groups
        ]

        if len(identities) != len(set(identities)):
            raise ValueError(
                "Each group must have exactly one set of analysis counts."
            )

        return self

class RateTestPlan(AuditBaseModel):
    rate: Literal["selection_rate", "tpr", "fpr", "ppv"]
    method: Literal[
        "two_proportion_z",
        "chi_square",
        "fisher_exact",
        "fisher_monte_carlo",
        "unavailable",
    ]
    min_expected_count: float | None = Field(ge=0)
    reason: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_test_availability(self):
        if (
            self.method != "unavailable"
            and self.min_expected_count is None
        ):
            raise ValueError(
                "A planned test requires the minimum expected cell count."
            )

        return self

class FairnessMetricPlan(AuditBaseModel):
    metric: Literal[
        "demographic_parity_difference",
        "equalized_odds_difference",
        "ppv_difference",
    ]
    available: bool
    bootstrap: bool
    reason: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_bootstrap_availability(self):
        if self.bootstrap and not self.available:
            raise ValueError(
                "Bootstrap cannot be planned for an unavailable metric."
            )

        return self

class AttributeAnalysisPlan(AuditBaseModel):
    attribute: str = Field(min_length=1, pattern=r"\S")
    eligible_group_count: int = Field(ge=0)

    fairness_metrics: list[FairnessMetricPlan] = Field(
        min_length=3,
        max_length=3,
    )
    statistical_tests: list[RateTestPlan] = Field(
        min_length=4,
        max_length=4,
    )

    compute_auroc: bool
    auroc_reason: str = Field(min_length=1, pattern=r"\S")

    @model_validator(mode="after")
    def validate_plan(self):
        metrics = [item.metric for item in self.fairness_metrics]

        if len(set(metrics)) != 3:
            raise ValueError(
                "Each headline fairness metric must appear exactly once."
            )

        rates = [item.rate for item in self.statistical_tests]

        if len(set(rates)) != 4:
            raise ValueError(
                "Each rate must have exactly one test plan."
            )

        if self.eligible_group_count < 2:
            if any(item.available for item in self.fairness_metrics):
                raise ValueError(
                    "Fairness comparisons require two eligible groups."
                )

            if any(
                item.method != "unavailable"
                for item in self.statistical_tests
            ):
                raise ValueError(
                    "Statistical comparisons require two eligible groups."
                )

        if self.eligible_group_count == 0 and self.compute_auroc:
            raise ValueError(
                "AUROC requires at least one eligible group."
            )

        return self

class AnalysisPlan(AuditBaseModel):
    score_available: bool
    attributes: list[AttributeAnalysisPlan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_analysis_plan(self):
        names = [item.attribute for item in self.attributes]

        if len(names) != len(set(names)):
            raise ValueError(
                "Each attribute must have exactly one analysis plan."
            )

        if not self.score_available and any(
            item.compute_auroc for item in self.attributes
        ):
            raise ValueError(
                "AUROC must not be planned without a score column."
            )

        return self


class PlanCorrection(AuditBaseModel):
    operation: Literal["add", "replace", "remove"]
    field_path: str = Field(min_length=1, pattern=r"\S")
    proposed_value: JsonValue
    corrected_value: JsonValue
    reason: str = Field(min_length=1, pattern=r"\S")


class GuardedAnalysisPlan(AuditBaseModel):
    plan: AnalysisPlan
    corrections: list[PlanCorrection]
