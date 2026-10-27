import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def calculate_selection_rate(
    predicted_positive: int,
    total: int,
) -> float:
    if total <= 0:
        raise ValueError("Total must be greater than zero.")

    if not 0 <= predicted_positive <= total:
        raise ValueError("Positive count must be between zero and total.")

    return predicted_positive / total

tools = [
    {
        "type": "function",
        "name": "calculate_selection_rate",
        "description": (
            "Calculate the fraction of cases predicted positive "
            "from a positive count and a total count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_positive": {
                    "type": "integer",
                    "description": "Number of cases predicted positive.",
                },
                "total": {
                    "type": "integer",
                    "description": "Total number of cases.",
                },
            },
            "required": ["predicted_positive", "total"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = OpenAI(timeout=30.0, max_retries=0)

response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input=(
        "In a synthetic dataset, 20 out of 100 cases were predicted "
        "positive. Use the tool to calculate the selection rate."
    ),
    tools=tools,
    tool_choice="required",
    parallel_tool_calls=False,
    store=False,
)

conversation = list(response.output)

for item in response.output:
    if item.type == "function_call":
        if item.name != "calculate_selection_rate":
            raise ValueError(f"Unknown tool: {item.name}")

        arguments = json.loads(item.arguments)
        result = calculate_selection_rate(**arguments)

        print("Tool result:", result)

        conversation.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({"selection_rate": result}),
            }
        )

final_response = client.responses.create(
    model=os.environ["OPENAI_MODEL"],
    input=conversation,
    instructions=(
        "Explain the tool result briefly. "
        "Express the selection rate as a percentage."
    ),
    store=False,
)

print(final_response.output_text)