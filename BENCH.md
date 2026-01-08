# LLM Benchmarks

6 separate tasks, using 11 different datasets. Using OpenAI API to send our requests. Sampling 20 data from these datasets for our benchmarks.

- Single Prompt Single Response (4 test cases)
    - alpac
    - triviaqa
    - narrativeqa
    - wikitext
- Beam Search Evaluation (4 test cases)
    - longbench_gov
    - longbench_qmsum
    - narrativeqa
    - triviaqa
- Shared Prefix (4 test cases)
    - kvprobe
    - sharegpt
    - leval
    - longchat
- Chatbot Evaluation (2 test cases)
    - sharegpt
    - longchat
- Question Answering (2 test cases)
    - triviaqa
    - narrativeqa
- Summarization (3 test cases)
    - longbench_gov
    - longbench_qmsum
    - loogle
