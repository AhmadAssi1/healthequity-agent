import pytest
from pydantic import ValidationError

from app.schemas import ColumnMapping


def test_valid_column_mapping():
    mapping = ColumnMapping(
        outcome="actual",
        prediction="predicted",
        groups=["gender", "race"],
    )

    assert mapping.outcome == "actual"
    assert mapping.groups == ["gender", "race"]
    assert mapping.score is None

def test_empty_groups_rejected():
    with pytest.raises(ValidationError):
        ColumnMapping(outcome="actual", groups=[])



@pytest.mark.parametrize("field", ["outcome", "prediction", "score"])
@pytest.mark.parametrize("name", ["", "   "])
def test_blank_column_names_rejected(field, name):
    values = {
        "outcome": "actual",
        "groups": ["gender"],
    }
    values[field] = name

    with pytest.raises(ValidationError):
        ColumnMapping(**values)

@pytest.mark.parametrize(
    "groups",
    [
        [""],
        ["   "],
        ["gender", "gender"],
        ["gender", ""],
    ],
)
def test_invalid_group_names_rejected(groups):
    with pytest.raises(ValidationError):
        ColumnMapping(outcome="actual", groups=groups)