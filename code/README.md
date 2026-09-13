# Buy or Wait? — Affordability Decision Agent

For every request in `dataset/requests.csv` the agent decides whether the user can safely afford an expense
and produces `output.csv` (amount safe today, status, method, payment plan, earliest safe full-payment date,
spending changes, explanation).

**Design principle:** AI reads the unstructured evidence (payslips, bills, receipts, notices); deterministic code
owns every number, date, safety check and ranking decision.

```
dataset/*.csv + media/images/*.png
        │
        ▼
  data.py ── typed loading + validation (fails loudly on malformed rows)
        │
        ▼
  evidence.py (messages)            images.py (blank event amounts)
  EN/ID template parser             vision LLM (if key) ─┐
  └─ LLM fallback, grounded         local OCR (rapidocr) ┴─ deterministic field selection
                                                          + amount-in-words / line-item checks
        │
        ▼
  state.py ── canonical FinancialState: recurring series, confirmed income, reserved
              pending/scheduled debits, FX conversion, conflict resolution, audit log
        │
        ▼
  forecast.py ── day-by-day simulator: headroom[t] = balance[t] − minimum_balance
        │
        ▼
  planner.py ── candidates (full / partial / each installment offer / wait, + flexible
                spending changes) → simulated safety check → published ranking rules
        │
        ▼
  explain.py + validate.py ── grounded explanation, output contract checks → output.csv
```

## Setup

Python 3.10+ (tested on 3.13). The core engine uses only the standard library.

```bash
pip install -r code/requirements.txt     # optional: offline OCR fallback for images
```

Optional model access (any one; nothing is required to run):

| variable | effect |
|---|---|
| `OPENAI_API_KEY` | use OpenAI (default model `gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | use Anthropic (default model `claude-haiku-4-5-20251001`) |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | use Google Gemini (default model `gemini-2.5-flash`) |
| `LLM_PROVIDER` | force `openai`, `anthropic` or `gemini` when several keys exist |
| `LLM_MODEL` | override the model name |
| `LLM_DISABLE=1` | force the fully offline path |

See `code/.env.example`. Keys are read from the environment only; nothing is written to disk except the
response cache in `code/cache/` (no keys stored).

## Run

From the repository root:

```bash
python code/main.py                         # dataset/requests.csv -> output.csv (+ evaluation/usage_report.md)
python code/evaluation/main.py              # score on sample_requests.csv + validate output.csv
python code/evaluation/main.py --verbose    # print every sample mismatch
python code/main.py --samples --only request_08 --trace   # full forecast trace for one request
python -m unittest discover -s code/tests -v              # unit tests
```

## How decisions are made

1. **Evidence.** Messages are parsed by exact English/Indonesian templates into typed facts (salary change from a
   date, salary date moved, first salary confirmed, employment ended, rent +12%, approved invoice, disputed
   charge, failed-debit retry, or *ignore* for pending/unconfirmed/non-cash/scam notices). Unknown formats go to
   the LLM, whose output is accepted only if every amount/date it returns literally occurs in the message.
   Images fill blank event amounts: the model/OCR only *reports* labelled amounts; code picks the right field
   (net pay for salary, balance due for outstanding bills, grand total otherwise) and cross-checks it.
2. **State.** Failed/cancelled/unrealized rows carry no cash; pending credits are ignored; pending and scheduled
   debits are reserved on their settlement date; scheduled salary is counted on its date. Monthly bills are
   projected on their day-of-month, spending streams (groceries/transport/dining) at their fixed interval,
   using the mean of history. Regular salary is projected monthly only when history shows a repeating amount
   and no message ends it; bonuses, commissions, arrears, gig and freelance credits are not projected.
3. **Forecast.** A daily balance path from `request_date` to `request_date + 86` days (window calibrated on the
   solved samples). `amount_safe_to_pay = min(headroom)`; the earliest full-payment date is the first day whose
   remaining-window minimum covers the full amount.
4. **Plans.** Eligible methods come from the user's preferences; installment offers must fit
   `max_installment_months`; every plan must finish by `desired_completion_date` and pass the simulator on every
   day. If a plan fails, flexible series in categories the user allows are stopped/reduced greedily (smallest
   saving first, max three, stop and reduce exclusive). Safe plans are ranked: deadline → no spending changes →
   lowest total paid → earliest start → fewer payments → lowest option id.

## Security

Message and image text is untrusted data: it is wrapped as data in prompts, can only produce facts from a
closed schema, model values must be grounded in the source text, and instruction-like content is flagged and
ignored (two scam "pay the release charge" notices in the dataset are classified as `ignore`). API keys come
from environment variables only.

## Evaluation

`code/evaluation/main.py` reports per-column accuracy on the 25 solved samples and validates `output.csv`
against the contract. `code/evaluation/usage_report.md` is regenerated by every full run with model calls,
tokens and estimated cost.
