"""Check fixture ground truth against independent arithmetic and engine outputs."""

import csv
import io
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.analysis_counts import compute_analysis_counts
from app.analysis_selection import select_analyses
from app.profiling import profile_dataset
from app.validation import validate_dataset
from evals import scenarios
from evals.scenarios import SOURCE, build_scenarios
from fairness.metrics import compute_fairness_gaps, compute_group_metrics
from fairness.policy import SuppressionPolicy


@pytest.fixture(scope="module")
def cases():
    return {case.name: case for case in build_scenarios()}


def test_suite_has_exact_brief_cases(cases):
    assert set(cases) == {
        "no_gap", "selection_gap", "tpr_gap", "ppv_gap", "tiny_group",
        "multiple_groups", "missing_score", "missing_outcome",
        "mislabelled_prediction", "duplicated_rows", "diabetes_female",
        "diabetes_male",
    }
    assert sum(case.expected["status"] == "refuse" for case in cases.values()) == 2


@pytest.mark.parametrize("name", ["missing_outcome", "mislabelled_prediction"])
def test_invalid_fixtures_are_detectable_contract_errors(cases, name):
    case = cases[name]
    with pytest.raises(ValueError, match=case.expected["error_contains"]):
        validate_dataset(case.data, case.mapping)


@pytest.mark.parametrize("name", [
    "no_gap", "selection_gap", "tpr_gap", "ppv_gap", "tiny_group",
    "multiple_groups", "missing_score", "duplicated_rows",
    "diabetes_female", "diabetes_male",
])
def test_valid_fixtures_match_frozen_counts_metrics_and_plan(cases, name):
    case = cases[name]
    before = case.data.copy(deep=True)
    expected = case.expected
    profile = profile_dataset(case.data, case.mapping)
    assert profile.row_count == expected["row_count"]
    assert profile.score_available == expected["score_available"]
    duplicate_warning = any("Duplicate rows" in flag for flag in profile.data_quality_flags)
    assert duplicate_warning == (expected["duplicate_rows"] > 0)
    audit_columns = [case.mapping.outcome, case.mapping.prediction, *case.mapping.groups]
    if case.mapping.score:
        audit_columns.append(case.mapping.score)
    assert int(case.data[audit_columns].duplicated().sum()) == expected["duplicate_rows"]
    policy = SuppressionPolicy(**expected["policy"])
    counts = compute_analysis_counts(case.data, case.mapping, policy)
    plan = select_analyses(counts, score_available=profile.score_available)
    for attribute, selected in zip(counts, plan.attributes, strict=True):
        answers = expected["attributes"][attribute.attribute]
        assert {g.group_value for g in attribute.groups} == set(answers["eligible_groups"])
        assert {t.rate: t.method for t in selected.statistical_tests} == answers["rate_tests"]
        assert selected.compute_auroc == expected["score_available"]
        raw_metrics = compute_group_metrics(case.data, attribute.attribute)
        assert set(raw_metrics.index) == set(answers["groups"])
        for label, group in answers["groups"].items():
            subset = case.data.loc[case.data[attribute.attribute].eq(label)]
            observed = {
                cell: int(((subset.outcome == y) & (subset.prediction == p)).sum())
                for cell, y, p in [("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0), ("tp", 1, 1)]
            }
            assert observed == group["confusion"]
            assert raw_metrics.loc[label, "count"] == group["count"]
            for rate, value in group["rates"].items():
                if value is not None:
                    assert raw_metrics.loc[label, rate] == pytest.approx(value, abs=1e-12)
        reportable = raw_metrics.loc[answers["eligible_groups"]]
        actual_gaps = compute_fairness_gaps(reportable)["gaps"]
        for metric, value in answers["gaps"].items():
            assert actual_gaps.loc[metric, "value"] == pytest.approx(value, abs=1e-12)
    assert_frame_equal(case.data, before)


@pytest.mark.parametrize("name, gaps", [
    ("no_gap", [0, 0, 0]),
    ("selection_gap", [0.2, 0.2, 0]),
    ("tpr_gap", [0.1, 0.2, 1 / 12]),
    ("ppv_gap", [0, 0.2, 0.2]),
])
def test_planted_headline_answers_have_hand_calculated_values(cases, name, gaps):
    answers = cases[name].expected["attributes"]["group"]["gaps"]
    assert list(answers.values()) == pytest.approx(gaps, abs=1e-12)


def test_suppression_checks_size_and_positive_count_separately(cases):
    answers = cases["tiny_group"].expected["attributes"]["group"]
    assert set(answers["suppressed_groups"]) == {"Tiny", "LowPositive"}
    assert answers["groups"]["Tiny"]["positive_outcomes"] >= 10
    assert answers["groups"]["LowPositive"]["count"] >= 50


def test_rebuilding_is_reproducible_and_returns_independent_frames(cases):
    rebuilt = build_scenarios()
    for fresh in rebuilt:
        assert_frame_equal(fresh.data, cases[fresh.name].data)
        assert fresh.expected == cases[fresh.name].expected
    rebuilt[0].data.loc[0, "outcome"] = 123
    assert cases["no_gap"].data.loc[0, "outcome"] in (0, 1)


def test_real_source_change_fails_instead_of_regenerating_answers(tmp_path):
    changed = tmp_path / "changed.csv"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_scenarios(changed)


def test_real_slices_are_disjoint_and_keep_fixed_model_scores(cases):
    female, male = cases["diabetes_female"], cases["diabetes_male"]
    assert set(female.data.gender) == {"Female"}
    assert set(male.data.gender) == {"Male"}
    # All audit columns come from the pinned baseline export, without retraining.
    assert female.expected["source_sha256"] == male.expected["source_sha256"]
    assert len(female.data) == len(male.data) == 1000


def test_fixture_hashes_do_not_depend_on_pandas_csv_float_serialization(monkeypatch):
    def unexpected_serialization(*args, **kwargs):
        pytest.fail("Fixture identity must not depend on pandas float serialization")
    monkeypatch.setattr(pd.DataFrame, "to_csv", unexpected_serialization)
    assert len(build_scenarios()) == 12


def test_frozen_real_slices_do_not_repeat_version_dependent_sampling(monkeypatch):
    original = pd.DataFrame.sample
    def sample(frame, *args, **kwargs):
        if len(frame) > 1000:
            pytest.fail("Real evaluation selections must use frozen source row indices")
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(pd.DataFrame, "sample", sample)
    assert len(build_scenarios()) == 12


def test_source_line_endings_do_not_change_fixture_identity(tmp_path, cases):
    source = tmp_path / "windows.csv"
    source.write_bytes(SOURCE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for rebuilt in build_scenarios(source):
        assert_frame_equal(rebuilt.data, cases[rebuilt.name].data)


@pytest.mark.parametrize("change", ["order", "out_of_bounds", "duplicate", "hash"])
def test_changed_frozen_selection_is_rejected(tmp_path, monkeypatch, change):
    manifest = json.loads(scenarios.MANIFEST.read_text())
    spec = manifest["slices"][0]
    if change == "order":
        spec["source_row_indices"].reverse()
    elif change == "out_of_bounds":
        spec["source_row_indices"][0] = 999999999
    elif change == "duplicate":
        spec["source_row_indices"][0] = spec["source_row_indices"][1]
    else:
        spec["source_rows_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(scenarios, "MANIFEST", path)
    with pytest.raises(ValueError, match="(?:indices|slice hash mismatch)"):
        build_scenarios()


def test_score_values_round_trip_from_original_csv_tokens(cases):
    rows = list(csv.reader(io.StringIO(SOURCE.read_text(encoding="utf-8"))))
    header, records = rows[0], rows[1:]
    score_column = header.index("score")
    manifest = json.loads(scenarios.MANIFEST.read_text())
    for spec in manifest["slices"]:
        expected = [float(records[i][score_column]) for i in spec["source_row_indices"]]
        assert cases[spec["name"]].data.score.tolist() == expected
