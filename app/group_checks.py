"""Deterministic group eligibility checks using dataset profiles."""
from app.tool_schemas import (
    AttributeEligibility,
    DatasetProfile,
    GroupEligibility,
    GroupSizeCheckResult,
)
from fairness.policy import (
    SuppressionPolicy,
    check_group_eligibility,
    load_suppression_policy,
)


def check_group_sizes(
    profile: DatasetProfile,
    policy: SuppressionPolicy | None = None,
) -> GroupSizeCheckResult:
    if policy is None:
        policy = load_suppression_policy()

    applied_policy = policy.model_copy(deep=True)
    attributes = []

    for attribute in profile.attributes:
        decisions = []

        for group in attribute.groups:
            decision = check_group_eligibility(
                count=group.count,
                positive_outcomes=group.positive_outcomes,
                policy=applied_policy,
            )

            decisions.append(
                GroupEligibility(
                    group_value=group.group_value,
                    reportable=decision["reportable"],
                    suppression_reason=decision["suppression_reason"],
                )
            )

        attributes.append(
            AttributeEligibility(
                attribute=attribute.attribute,
                groups=decisions,
            )
        )

    return GroupSizeCheckResult(
        suppression_policy=applied_policy,
        attributes=attributes,
    )

