from pydantic import BaseModel, Field, field_validator


class ColumnMapping(BaseModel):
    outcome: str = Field(min_length=1)
    prediction: str | None = None
    groups: list[str] = Field(min_length=1)
    score: str | None = None

    @field_validator("outcome", "prediction", "score")
    @classmethod
    def validate_column_name(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Column name must not be blank.")
        return value
    
    @field_validator("groups")
    @classmethod
    def validate_group_names(cls, value):
        if any(not name.strip() for name in value):
            raise ValueError("Group column names must not be blank.")
        if len(value) != len(set(value)):
            raise ValueError("Group column names must be unique.")
        return value


class DatasetFingerprint(BaseModel):
    row_count: int = Field(ge=1)
    columns_hash: str = Field(min_length=1)