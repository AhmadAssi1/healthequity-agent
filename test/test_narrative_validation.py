"""Controlled group-reference parsing and clause-local causal-language checks."""

import pytest

from app.narrative_validation import (
    check_group_mentions,
    has_causal_claim,
    has_unsupported_approval,
)

GROUPS = {"race": ["Caucasian", "AfricanAmerican", None], "gender": ["Female", "Male"],
          "age": ["[70-80)", "[0-10)"], "cohort": [1, 1.0, True, "1", "A", 'O"Brien']}


@pytest.mark.parametrize("text", [
    'The largest group is race: "Caucasian".',
    'The comparison includes gender="Female".',
    'The group "Female" has an observed difference.',
    "The 'Male' group requires review.",
    'The age: "[70-80)" group requires review.',
    'Missing race: null remains separate.',
    'Compare cohort: 1 with cohort: 1.0 and cohort: true and cohort: "1".',
    'The group is cohort: "O\\\"Brien".',
    "Missing values in 'race' form a separate group.",
    'The metric "selection_rate" is reported.',
    'The group Female is present.',
    'The Male group is present.',
    'Groups [70-80) were compared.',
])
def test_supported_explicit_group_mentions(text):
    _, issues = check_group_mentions(text, GROUPS, "summary")
    assert not issues


@pytest.mark.parametrize("text", [
    'The group "Martian" has higher selection.',
    'race: "Female" has higher selection.',
    'ethnicity: "Caucasian" was compared.',
    'gender: null was included.',
    'gender: 1 was included.',
    'The group "Missing" is present.',
    'The group "watch" is present.',
    'The group Martian is present.',
    'The Martian group is present.',
    'The explanation is "everything is fine".',
])
def test_unknown_misattributed_or_unquoted_group_mentions_are_rejected(text):
    _, issues = check_group_mentions(text, GROUPS, "summary")
    assert issues
    assert all(issue.code == "unsupported_group" for issue in issues)


@pytest.mark.parametrize("text", [
    "The selection-rate test returned p=6.210520379526549e-7.",
    "The gap has a 95% CI: 0.105 to 0.251.",
    "The gap has a 95% CI [0.105, 0.251].",
])
def test_statistical_annotations_are_not_group_references_and_retain_numbers(text):
    prose, issues = check_group_mentions(text, GROUPS, "summary")
    assert not issues
    assert prose == text


def test_statistical_symbol_can_still_be_an_actual_mapped_group_attribute():
    assert check_group_mentions("p=0.5", {"p": [0, 1]}, "summary")[1]


@pytest.mark.parametrize(("text", "count"), [
    ("Six groups were included.", "6"), ("Two groups reported.", "2"),
    ("Eight age groups were analyzed.", "8"), ("Ten groups included.", "10"),
    ("Seven suppressed groups were identified.", "7"),
])
def test_word_form_counts_are_numeric_claims_not_group_labels(text, count):
    prose, issues = check_group_mentions(text, GROUPS, "summary")
    assert not issues and prose.startswith(count + " ")


def test_unknown_group_names_remain_rejected_while_known_bare_labels_are_masked():
    assert check_group_mentions("The group Martian is present.", GROUPS, "summary")[1]
    assert check_group_mentions("The group A is present.", {"one": ["A"], "two": ["A"]}, "summary")[1]
    prose, issues = check_group_mentions("Groups [70-80) were compared.", GROUPS, "summary")
    assert not issues and "70" not in prose and "80" not in prose


@pytest.mark.parametrize("determiner", ["The", "This", "That", "Each", "Every", "Any", "All", "No", "A"])
def test_generic_determiner_does_not_consume_the_following_group_reference(determiner):
    _, issues = check_group_mentions(f"{determiner} group Martian requires review.", GROUPS, "summary")
    assert issues


@pytest.mark.parametrize("text", ["The group requires review.", "Each group was considered.",
                                 "The group called Martian requires review."])
def test_generic_group_wording_and_named_group_are_distinguished(text):
    _, issues = check_group_mentions(text, GROUPS, "summary")
    assert bool(issues) == ("Martian" in text)


@pytest.mark.parametrize("description", [
    "Suppressed", "Excluded", "Included", "Eligible", "Ineligible", "Reportable",
    "Small", "Smaller", "Large", "Demographic", "Protected", "Reference", "Comparison",
    "Underrepresented", "Minority", "Majority", "Race", "Gender", "Age", "Some", "These",
    "Certain", "Several", "Many", "Few", "Both", "Multiple", "Various",
])
def test_sentence_initial_group_descriptions_are_not_category_names(description):
    _, issues = check_group_mentions(f"{description} groups have insufficient data for reliable metrics.", GROUPS, "limitations[2]")
    assert not issues


def test_descriptor_that_is_also_a_category_label_requires_disambiguation():
    assert check_group_mentions("Suppressed groups need review.", {"cohort": ["Suppressed"]}, "summary")[1]


def test_unquoted_name_feedback_identifies_the_actual_token():
    _, issues = check_group_mentions("The group Martian is present.", GROUPS, "summary")
    assert issues and '"Martian"' in issues[0].message


def test_live_style_suppression_limitation_is_not_a_group_claim():
    text = "Suppressed groups have insufficient data for reliable metrics."
    prose, issues = check_group_mentions(text, GROUPS, "limitations[2]")
    assert not issues and prose == text


def test_ambiguous_labels_require_attribute_qualification():
    groups = {"one": ["A"], "two": ["A"]}
    assert check_group_mentions('The group "A" is present.', groups, "summary")[1]
    assert not check_group_mentions('one: "A" is present.', groups, "summary")[1]


@pytest.mark.parametrize("text", [
    'For attribute race, groups "" and "Asian" were compared.',
    'For the race attribute, groups "" and "Asian" were compared.',
    'The attribute race includes groups "" and "Asian".',
    'For the attribute race, groups "" and "Asian" were compared.',
    'In attribute race, groups "" and "Asian" were compared.',
])
def test_explicit_sentence_attribute_resolves_shared_empty_label(text):
    groups = {"race": ["", "Asian"], "payer_code": ["", "PO"]}
    assert not check_group_mentions(text, groups, "summary")[1]


@pytest.mark.parametrize("text", [
    'For attribute race, findings are available. The group "" was compared.',
    'For attribute race and attribute payer_code, the group "" was compared.',
    'For attribute invented, the group "" was compared.',
    'The group "" was compared.',
])
def test_attribute_scope_cannot_cross_sentences_or_guess_between_attributes(text):
    groups = {"race": [""], "payer_code": [""]}
    _, issues = check_group_mentions(text, groups, "summary")
    assert issues and '""' in issues[0].message


@pytest.mark.parametrize(("token", "trusted"), [("1", True), ("1.0", 1), ("true", 1), ('"1"', 1), ("null", "Missing")])
def test_typed_labels_do_not_alias(token, trusted):
    assert check_group_mentions(f"cohort: {token}", {"cohort": [trusted]}, "summary")[1]


def test_valid_age_bands_and_numeric_labels_are_masked_from_numeric_claim_checks():
    text = 'For age: "[70-80)", cohort: 1 had a gap of 0.20.'
    prose, issues = check_group_mentions(text, GROUPS, "summary")
    assert not issues
    assert "70" not in prose and "80" not in prose
    assert "cohort: 1" not in prose and "0.20" in prose


def test_group_label_text_is_data_not_a_causal_claim():
    prose, issues = check_group_mentions('cohort: "Race causes bias" requires review.',
                                        {"cohort": ["Race causes bias"]}, "summary")
    assert not issues and not has_causal_claim(prose)


@pytest.mark.parametrize("text", [
    "The gap exists because of discrimination.",
    "The difference is due to racial bias.",
    "Race causes the disparity.",
    "Discrimination caused the gap.",
    "The result is caused by racism.",
    "The gap is driven by gender.",
    "The gap is attributed to insurance status.",
    "The disparity results from discrimination.",
    "The analysis establishes causation.",
    "No score exists, but race causes the gap.",
    "No score exists; the gap is because of discrimination.",
    "The model is not calibrated and race causes the gap.",
    "The gap is not only because of discrimination.",
])
def test_affirmative_causal_claims_are_rejected(text):
    assert has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "Association does not imply causation.",
    "The audit cannot prove causation.",
    "No causal effect can be inferred.",
    "Causal effects cannot be inferred.",
    "The gap is not because of discrimination.",
    "There is no evidence that race causes the disparity.",
    "Groups were suppressed due to insufficient sample size.",
    "Groups were suppressed because of below_min_positive_outcomes.",
    "The audit describes associations only.",
])
def test_explicit_limitations_and_policy_suppression_are_allowed(text):
    assert not has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "Some groups were suppressed due to small size or low positive outcome count, limiting insight for these groups.",
    "Some groups were suppressed due to small size or low positive outcome count, limiting conclusions for these groups.",
    "Findings for suppressed groups are unavailable due to insufficient group size or positive outcome counts.",
    "Suppressed group findings are unavailable due to insufficient group size and/or positive outcome counts as indicated.",
    "Groups were suppressed due to low positive outcome counts.",
    "Groups were suppressed due to both small size and insufficient positive outcomes.",
])
def test_saved_live_policy_explanations_are_not_causal_disparity_claims(text):
    prose, issues = check_group_mentions(text, GROUPS, "limitations[0]")
    assert not issues
    assert not has_causal_claim(prose)


@pytest.mark.parametrize("text", [
    "Some groups were suppressed due to discrimination.",
    "Certain groups have a higher rate because of racial bias.",
    "The small group has a disparity due to racism.",
])
def test_policy_vocabulary_does_not_exempt_discrimination_claims(text):
    assert has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "The disparity is due to insufficient group size.",
    "The gap is because of low positive outcome counts.",
    "The disparity is due to small size.",
])
def test_policy_cause_exemption_requires_a_suppression_or_unavailability_context(text):
    assert has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "Confidence intervals were computed using bootstrap resampling, which may have variability due to random sampling.",
    "Confidence intervals were computed using bootstrap resampling, which may exhibit variability due to random sampling.",
    "Bootstrap estimates vary because of random sampling.",
    "Interpret the analysis with caution due to data quality flags indicating duplicated rows.",
])
def test_procedural_uncertainty_is_not_a_cause_of_demographic_disparity(text):
    assert not has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "The race disparity is due to random sampling.",
    "Bootstrap resampling was used, but the gap exists because of discrimination.",
    "This result is because of racial bias.",
    "The groups differ due to ageism.",
])
def test_methodology_words_do_not_hide_claims_about_demographic_gap_causes(text):
    assert has_causal_claim(text)


@pytest.mark.parametrize("text", [
    "The model is fair.",
    "No scores exist, but the model is fair.",
    "The test did not fail and validation has passed.",
])
def test_unrelated_negation_does_not_hide_unsupported_approval(text):
    assert has_unsupported_approval(text)


@pytest.mark.parametrize("text", ["The model is not legally compliant.", "The audit cannot prove fairness."])
def test_negated_approval_claims_are_allowed(text):
    assert not has_unsupported_approval(text)
