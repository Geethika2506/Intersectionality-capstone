# Intersectionality capstone — GREVIO policy brief pilot

Python tooling for a fixed protocol: two coded GREVIO passages, six independent OpenAI Responses API calls (three 400-word policy briefs per passage), with reproducible artifacts and no conversation history between requests.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .env.example          # copy to .env (never commit .env)
├── .gitignore
├── run_pilot.py                      # generation CLI
├── evaluate_intersectionality.py     # human-review CLI (no API calls)
├── pilot/                            # library code
├── data/                             # original passage source text (no coding labels)
├── prompts/                          # fixed instruction prompts per passage
├── evaluation/                       # reference coding — never sent to the model
├── runs/                             # response JSON, then review forms
└── tests/
```

## Mac setup

### 1. Clone the repository

```bash
git clone https://github.com/Geethika2506/Intersectionality-capstone.git
cd Intersectionality-capstone
```

### 2. Python virtual environment

Requires **Python 3.10+**.

```bash
cd /path/to/Intersectionality-capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

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
3. Copy your coding tables into `evaluation/` when you review the briefs. Those files are not part of the model input. See [Review the six briefs](#review-the-six-briefs).

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

## Review the six briefs

Generation and review are separate. Review reads the saved response JSON and your coding tables. It does not call the API, and it does not change the six response files.

### 1. Copy your coding tables

Open `evaluation/passage_01_coding_REFERENCE.txt`. The last line is a header. In your spreadsheet, arrange the passage 01 coding in three columns named `category`, `item_id`, and `reference_text`. Copy those rows and paste them under the header. Leave a tab between columns.

Do the same for passage 02 in `evaluation/passage_02_coding_REFERENCE.txt`.

`category` is one of:

- `social_position`
- `intersectional_relationship`
- `institutional_mechanism`
- `policy_problem`
- `responsible_actor`
- `participation_monitoring`
- `essential_relationship`

Each passage needs at least one row in each of the first three categories. The other four are used when that element is in the coding.

`item_id` is your own code for that row, such as `U1-SP`. Use each id only once in the file. `reference_text` is the coded item. Keep passage 02 rows out of the passage 01 file.

A pasted row looks like this (the gaps are tabs):

```
social_position	SP1	One coded item from your table
intersectional_relationship	IR1	One coded item from your table
institutional_mechanism	IM1	One coded item from your table
```

Replace that sample wording with your own rows. Lines that start with `#` are notes and are ignored.

### 2. Create the six review forms

After `python run_pilot.py --run` has written six `*_response.json` files:

```bash
python evaluate_intersectionality.py templates --run-dir runs/YOUR_RUN_FOLDER
```

This writes `passage_01_repeat_01_review.json` through `passage_02_repeat_03_review.json` in `runs/YOUR_RUN_FOLDER/reviews/`. Each form includes the brief text taken from that response JSON. Running the command again does not replace a form that is already there. Delete a form file first if you want a new blank copy. Completed forms are left unchanged.

### 3. Fill in each form

Open each review file. For every reference item, set `retention`:

- `0` — omitted or distorted
- `1` — partly preserved
- `2` — preserved

For `1` or `2`, paste an `excerpt` from that brief and set `supported_by_passage` to `true` or `false`. `true` means you judge the claim to be supported by the GREVIO passage.

```json
"retention": 2,
"excerpt": "short quote from this brief",
"supported_by_passage": true
```

For `0`, leave `excerpt` as `""` and `supported_by_passage` as `null`.

If the brief adds a claim that is not in your coding, add it under `additional_unsupported_claims`:

```json
"additional_unsupported_claims": [
  {
    "claim": "What the brief added",
    "excerpt": "quote from this brief"
  }
]
```

Leave that list empty if there are none. Do not edit `passage_id`, `output_file`, `item_id`, `category`, or `reference_text`.

### 4. Write the summary

```bash
python evaluate_intersectionality.py summary --run-dir runs/YOUR_RUN_FOLDER
```

The command checks for missing labels, duplicate items, a passage id that does not match the brief, and output files it does not recognize. When the six forms are complete, it writes `runs/YOUR_RUN_FOLDER/reviews/summary.md`.

The summary reports retention for each category and describes how the three runs of each passage differ. It also reports the word count of each brief, counted from the response JSON. Three runs describe this pilot only. They are not a statistical result.

## Tests

```bash
source .venv/bin/activate
pytest -q
```

Tests use mocks and fake response JSON only. They do not call OpenAI.

`evaluation/` holds reference coding separate from model inputs. `run_pilot.py` reads only `data/` and `prompts/`.
