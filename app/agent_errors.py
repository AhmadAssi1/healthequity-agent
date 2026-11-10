"""Audit-agent errors that preserve the execution trace."""

from app.agent_schemas import AgentRunTrace


class AgentRunError(RuntimeError):
    def __init__(
        self,
        message: str,
        trace: AgentRunTrace,
    ):
        if trace.status != "failed":
            raise ValueError(
                "AgentRunError requires a failed trace."
            )

        super().__init__(message)
        self.trace = trace.model_copy(deep=True)