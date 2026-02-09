from benchmarks.mmlu import MMLUBenchmark


if __name__ == "__main__":
    benchmark = MMLUBenchmark.main()

    for result in benchmark.run():
        print(result)
