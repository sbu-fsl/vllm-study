import csv
import ast
import time
import requests
from typing import List

# ---------------- CONFIG ----------------
API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"
REQUEST_PERIOD = 30  # seconds
TEMPERATURE = 0.0
# ----------------------------------------


def build_prompt(question: str, choices: List[str]) -> str:
    formatted_choices = "\n".join(
        f"{idx}. {choice}" for idx, choice in enumerate(choices)
    )

    return f"""Answer the following multiple-choice question.
Respond with ONLY the number of the correct choice.

Question:
{question}

Choices:
{formatted_choices}
"""


def query_vllm(prompt: str, timeout: int = 30) -> str | None:
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        print("⏱️ Request timed out")
        return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


def process_dataset(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            start_time = time.time()

            question = row["question"]
            choices = ast.literal_eval(row["choices"])

            prompt = build_prompt(question, choices)

            print(f"\n▶ Processing question {idx + 1}")
            answer = query_vllm(prompt, timeout=REQUEST_PERIOD)

            elapsed = time.time() - start_time

            if answer is not None:
                print(f"✔ Model answer: {answer}")
            else:
                print("⚠️ No answer (timeout or error)")

            # Enforce fixed 30-second period
            remaining = REQUEST_PERIOD - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    process_dataset("sampled_mmlu_data.csv")
