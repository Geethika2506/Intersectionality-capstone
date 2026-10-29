# Reference coding

Files in this directory are not used when generating briefs. `run_pilot.py` reads only `prompts/` and `data/`.

Paste each passage’s coding table into:

- `passage_01_coding_REFERENCE.txt`
- `passage_02_coding_REFERENCE.txt`

Both files end with a header row. Columns are `category`, `item_id`, and `reference_text`, separated by a tab. Each passage needs at least one `social_position`, one `intersectional_relationship`, and one `institutional_mechanism`. A row may also use `policy_problem`, `responsible_actor`, `participation_monitoring`, or `essential_relationship`. Item ids must be unique inside a file. Keep each passage in its own file.

Rows can also be grouped under headings:

```
## social_position
SP1	One coded item

## intersectional_relationship
IR1	One coded item

## institutional_mechanism
IM1	One coded item
```

The main README explains how to create the six review forms, fill the labels, and write the summary. That step reads these tables and the saved response JSON. It does not call the API and does not change the response files.
