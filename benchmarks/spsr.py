# Read the HumanEval dataset and send HTTP requests
# to the vLLM service.

import csv
import datetime
import requests


API_URL = "http://localhost:30101/v1/completions"
HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en;q=0.9,fa-IR;q=0.8,fa;q=0.7,en-US;q=0.6',
    'Connection': 'keep-alive',
    'Content-Type': 'application/json'
}
MODEL = "facebook/opt-125m"


def http_post(prompt: str, osize: int) -> int:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": osize,
        "temperature": 0
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            verify=False,
            timeout=30
        )

        print(response.json())
        print()

        return response.status_code
    except Exception as e:
        print(e)
        return -1


def read_csv_as_list_of_dicts(filename):
    list_of_dicts = []
    with open(filename, mode='r', newline='', encoding='utf-8') as csv_file:
        dict_reader = csv.DictReader(csv_file)
        for row in dict_reader:
            list_of_dicts.append(row)
    return list_of_dicts


def main():
    print(f"starting: {datetime.datetime.now()}")
    
    data = read_csv_as_list_of_dicts("datasets/humaneval_20_samples.csv")

    for d in data:
        print(http_post(d["prompt"], len(d["canonical_solution"])))
    
    print(f"finished: {datetime.datetime.now()}")


if __name__ == "__main__":
    main()
