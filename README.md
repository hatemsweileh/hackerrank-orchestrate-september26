# Buy or Wait? — HackerRank Orchestrate (September 2026)

An affordability decision agent: for each request in `dataset/requests.csv` it decides how much the user can
safely pay today, whether and how the request is affordable (full, partial, installments, wait or not
recommended), the exact payment plan, the earliest safe full-payment date and any flexible spending changes.

AI reads the unstructured evidence (payslips, bills and receipts via a vision model or local OCR; unfamiliar
notices via an LLM with grounding checks). Deterministic code owns every number: state reconstruction, a
day-by-day cash simulation, plan validation and the published ranking rules.

## Layout

```text
code/
  main.py                 entry point -> output.csv (+ evaluation/usage_report.md)
  agent/                  data loading, evidence parsing, images, state, forecast, planner, explanations, validation
  evaluation/main.py      sample scoring + output contract validation
  evaluation/usage_report.md
  tests/                  unit tests (engine, parsers, mocked OpenAI/Anthropic/Gemini paths)
  README.md               design and details
  requirements.txt, .env.example
output.csv                predictions for dataset/requests.csv
```

## Run

Place the challenge `dataset/` folder at the repository root (same layout as the starter repository), then:

```bash
pip install -r code/requirements.txt       # optional OCR fallback; core engine is standard library only
python code/main.py                        # writes output.csv and code/evaluation/usage_report.md
python code/evaluation/main.py             # scores sample_requests.csv and validates output.csv
python -m unittest discover -s code/tests  # unit tests
```

Optional model access: set one of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` (see
`code/.env.example`). Without a key the agent runs fully offline.

See [`code/README.md`](code/README.md) for the architecture, decision rules and security notes.
