"""Build policy-filtered profiles for the language model."""

from app.group_checks import check_group_sizes
from app.tool_schemas import (
    AgentAttributeProfile,
    AgentDatasetProfile,
    DatasetProfile,
    SuppressedGroupSummary,
)
from fairness.policy import SuppressionPolicy


def build_agent_profile(
    profile: DatasetProfile,
    policy: SuppressionPolicy | None = None,
) -> AgentDatasetProfile:
    eligibility = check_group_sizes(profile, policy)
    applied_policy = eligibility.suppression_policy
    attributes = []

    for source, decisions in zip(
        profile.attributes,
        eligibility.attributes,
        strict=True,
    ):
        reportable_groups = []
        suppressed_groups = []

        for group, decision in zip(
            source.groups,
            decisions.groups,
            strict=True,
        ):
            if decision.reportable:
                reportable_groups.append(
                    group.model_copy(deep=True)
                )
                continue

            reasons = []

            if group.count < applied_policy.min_group_size:
                reasons.append("below_min_group_size")

            if (
                group.positive_outcomes
                < applied_policy.min_positive_outcomes
            ):
                reasons.append("below_min_positive_outcomes")

            suppressed_groups.append(
                SuppressedGroupSummary(
                    group_value=group.group_value,
                    reason_codes=reasons,
                )
            )

        attributes.append(
            AgentAttributeProfile(
                attribute=source.attribute,
                reportable_groups=reportable_groups,
                suppressed_groups=suppressed_groups,
            )
        )

    allowed_flags = {
        "Predictions are not mapped; fairness analysis requires predictions.",
        (
            "Duplicate rows exist in the mapped audit columns; "
            "no rows were removed."
        ),
        "The outcome column contains only one class.",
        *(
            f"Missing values in '{item.attribute}' form a separate group."
            for item in profile.attributes
        ),
    }

    safe_flags = []

    for flag in profile.data_quality_flags:
        safe_flag = (
            flag
            if flag in allowed_flags
            else "Additional internal data quality flags require review."
        )

        if safe_flag not in safe_flags:
            safe_flags.append(safe_flag)

    return AgentDatasetProfile(
        row_count=profile.row_count,
        column_count=profile.column_count,
        outcome_prevalence=profile.outcome_prevalence,
        score_available=profile.score_available,
        suppression_policy=applied_policy,
        attributes=attributes,
        data_quality_flags=safe_flags,
    )
