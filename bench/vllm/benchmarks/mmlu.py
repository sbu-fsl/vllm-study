import ast

from tasks.chatbot import ChatBot
from datasets.csvdataset import CSVDataset
from src.benchmark import Benchmark


class MMLUBenchmark(Benchmark):
    def build_input(self, entry: dict) -> tuple[str, dict]:
        question = entry["question"]
        choices = ast.literal_eval(entry["choices"])

        formatted_choices = "\n".join(
            f"{idx}. {choice}" for idx, choice in enumerate(choices)
        )

        prompt = f"""Answer the following multiple-choice question.
        Respond with ONLY the number of the correct choice.

        Question:
        {question}

        Choices:
        {formatted_choices}
        """

        opts = {
            "temperature": 0.0,
            "max_tokens": 32
        }

        return prompt, opts

    @classmethod
    def main(cls) -> "MMLUBenchmark":
        dataset = CSVDataset("sampled_mmlu_data.csv", batch_size=10)
        task = ChatBot(model="facebook/opt-125m")

        return cls(dataset, task)
