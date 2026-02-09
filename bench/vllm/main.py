import requests

from benchmarks.mmlu import MMLUBenchmark


API_BASE = "http://localhost:8000/v1"
REQUEST_PERIOD = 30  # seconds


if __name__ == "__main__":
    benchmark = MMLUBenchmark.main()

    for result in benchmark.run():
        url = f"{API_BASE}{result["uri"]}"
        headers = {
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=result["payload"],
                timeout=REQUEST_PERIOD,
            )
            response.raise_for_status()
            print(response.json()["choices"][0]["message"]["content"].strip())

        except requests.exceptions.Timeout:
            print("⏱️ Request timed out")
        except Exception as e:
            print(f"❌ Request failed: {e}")
