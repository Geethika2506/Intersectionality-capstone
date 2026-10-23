# Intersectionality capstone — GREVIO policy brief pilot

Python tooling for a fixed protocol: two coded GREVIO passages, six independent OpenAI Responses API calls (three 400-word policy briefs per passage), with reproducible artifacts and no conversation history between requests.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example          # copy to .env (never commit .env)
├── .gitignore
├── run_pilot.py          # CLI entry point
├── pilot/                # library code
├── data/                 # original passage source text (no coding labels)
├── prompts/              # fixed instruction prompts per passage
├── evaluation/           # reference coding — never sent to the model
├── runs/                 # request/response artifacts and run_log.jsonl
└── tests/
```

## Mac setup (Terminal or Cursor)

### 1. Clone and open in Cursor

Open this folder in Cursor (`File → Open Folder…`).

### 2. Python virtual environment

Requires **Python 3.10+**.

```bash
cd /path/to/Intersectionality-capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In Cursor, choose the interpreter: `Cmd+Shift+P` → **Python: Select Interpreter** → `.venv/bin/python`.

### 3. API key

```bash
cp .env.example .env
```

Edit `.env` and set your key (do not commit this file):

```
OPENAI_API_KEY=sk-...
```

### 4. Fill study materials

Before any paid run:

1. Replace placeholders in `data/passage_01.txt` and `data/passage_02.txt` with the **original** GREVIO text (context and footnotes as supplied). Do **not** put coding labels or annotations here.
2. Replace placeholders in `prompts/passage_01.txt` and `prompts/passage_02.txt` with each passage’s **fixed** brief instructions (policy question, length target, etc.). Do not duplicate full GREVIO source into prompt files.
3. Optionally store team coding in `evaluation/` for your records only.

The script refuses `--run` while `[PLACEHOLDER` markers remain.

### 5. Dry run (free — no API)

Lists all six planned requests (model, reasoning effort, slug, input size). Does **not** print full passage text or call OpenAI.

```bash
source .venv/bin/activate
python run_pilot.py --dry-run
```

### 6. Paid run (six Responses API calls)

Only after placeholders are removed and `.env` is configured:

```bash
source .venv/bin/activate
python run_pilot.py --run
```

Each run creates a new UTC-stamped directory under `runs/`, for example `runs/20261023T120000Z/`.

**Protocol enforced by code**

| Setting | Value |
|--------|--------|
| API | OpenAI **Responses** (`client.responses.create`) |
| Model | `gpt-6.1-sol` |
| Reasoning | `medium` |
| Requests | 6 (passage_01 ×3, passage_02 ×3) |
| Input per request | That passage’s prompt + source only (no prior messages or response IDs) |
| Tools | None (no web search) |
| Output handling | Saved exactly as returned; no length retries or edits |

**Artifacts per request**

- `{passage_id}_repeat_{01-03}_request.json` — exact input text and metadata
- `{passage_id}_repeat_{01-03}_response.json` — full API response JSON
- `run_log.jsonl` — one JSON line per attempt with UTC time, IDs, models, reasoning, tokens, files, word count

If a response file already exists in the run directory, the script **stops** instead of overwriting or re-billing. API errors are written to `*_failure.json` and logged; there is **no** automatic retry.

Optional: use a pre-created empty directory:

```bash
mkdir -p runs/my_manual_run
python run_pilot.py --run --run-dir runs/my_manual_run
```

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Tests use mocks only; they do not call OpenAI.

## Evaluation folder

`evaluation/` holds reference coding separate from model inputs. The pilot code reads only `data/` and `prompts/`.
