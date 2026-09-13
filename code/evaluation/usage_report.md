# Token Usage and Cost Report

Generated automatically by `code/main.py` at the end of the full-dataset run that produced `output.csv`.

- Requests processed: **250**
- Runtime: **63.4 s**
- LLM provider configured: **none (offline mode)**

## Per model

| provider/model | calls | live calls | cache hits | failures | input tokens | output tokens | total tokens | est. cost (USD) |
|---|---|---|---|---|---|---|---|---|
| (no model calls) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |

## Overall

- Model calls: **0**
- Input tokens: **0**, output tokens: **0**, total: **0**
- Average tokens per request: **0.0**
- Estimated total cost: **$0.0000**; per request: **$0.000000**

## Offline (non-LLM) AI components

| component | invocations | tokens | cost |
|---|---|---|---|
| image_ocr (local rapidocr ONNX model) | 11 | 0 | $0 |

Token counts come from the provider usage fields returned by each API response; cache hits re-use the
recorded counts of the original call and are not billed again. Costs use the list prices in
`code/agent/llm.py::PRICING` and are estimates. Deterministic components (financial forecast, plan
ranking, validation, message template parser) use no tokens.
