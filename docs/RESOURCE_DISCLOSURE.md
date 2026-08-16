# Resource Disclosure — Durham-Jazan, AISA-ArabicFC (ArabicNLP 2026)

Prepared 30 July 2026 in response to the organizers' data-use clarification of the same
date. Every claim below is backed by an artefact in our working tree that can be re-run or
diffed. Note that the public repository does not carry `data/` or `runs/`, so the
training files, logs and registries named here are available on request rather than
by download.

## 1. Models

Two open-weights base models, both used unmodified and both general-purpose (neither
contains AISA test items with gold labels):

- `ALLaM-AI/ALLaM-7B-Instruct-preview`
- `Qwen/Qwen2.5-7B-Instruct`

Six QLoRA adapters (4-bit NF4 base, LoRA on
`q,k,v,o,gate,up,down_proj`, dropout 0.05):

| tag | base | rank / alpha | epochs |
|-----|------|--------------|--------|
| v10 | ALLaM-7B    | 32 / 64  | 2 |
| v12 | ALLaM-7B    | 64 / 128 | 2 |
| v7b | Qwen2.5-7B  | 32 / 64  | 2 |
| v7  | Qwen2.5-7B  | 32 / 64  | 1 |
| aC  | ALLaM-7B    | 32 / 64  | 1 |
| qC  | Qwen2.5-7B  | 32 / 64  | 1 |

No proprietary or API model was used to produce, label, augment, or distil any training
data or any submitted prediction. No retrieval, no external knowledge base, no
test-time lookup.

## 2. Training data — provided train split only

Every one of the six adapters was trained on `train_st_aug.jsonl`, which contains **only**
rows from the official shared-task **train** split plus synthetic negative examples derived
from those same rows. Verified source-field counts:

| file | sharedtask | augment | parent | dev |
|------|-----------:|--------:|-------:|----:|
| `data/processed/train_st_aug.jsonl`     | 10,191 | 1,108 | **0** | **0** |
| `data/processed_v11/train_st_aug.jsonl` | 10,176 | 1,108 | **0** | **0** |
| `data/processed_v13/train_st_aug.jsonl` | 10,161 | 2,208 | **0** | **0** |

**Independent confirmation from the training logs.** `train_qlora.py` records the tokenized
example count, which uniquely identifies the input file:

| adapter | log | tokenized | ⇒ file |
|---|---|---:|---|
| v10 | `runs/allam-v10-2ep.log`       | 11,284 | `processed_v11/train_st_aug.jsonl` |
| v12 | `runs/allam-v12-r64-2ep.log`   | 11,284 | `processed_v11/train_st_aug.jsonl` |
| v7b | `runs/qwen7b-v7b-2ep.log`      | 11,284 | `processed_v11/train_st_aug.jsonl` |
| v7  | `runs/qwen7b-v7-v11.log`       | 11,284 | `processed_v11/train_st_aug.jsonl` |
| aC, qC | `runs/chain_clean_zoo.log`  | 12,369 | `processed_v13/train_st_aug.jsonl` |

These counts cannot be confused with the parent-merged `train.jsonl` (39,899 / 41,118 rows).

**On the parent corpus.** `aisa-ar-functioncall-parent` was downloaded during early
exploration and a merged `train.jsonl` containing it exists in the repository. **No shipped
adapter was trained on it** (counts above). One discarded experiment, `v8`
(`scripts/autoeval_v8.sh`), did use the merged file; v8 is not in the submitted ensemble
(`scripts/ensemble_config.json`) and its predictions are not used.

**Dev split.** Never trained on. `prepare_data.py` additionally removes any training row
whose user query matches a dev query (648 rows dropped, recorded in `prepare_stats.json`).
Dev was used only to select the ensemble members and to measure the reported dev scores.

**Blind test.** No gold labels for the test split were sought, reconstructed, or used. The
test split was consumed inputs-only through `scripts/prepare_blind.py`.

## 3. Post-processing rules

Two deterministic rules, both estimated **from the provided train split** and applied
uniformly without reference to any dev or test label:

1. **Year rule** — for `search_hotels.check_in/check_out` and
   `book_doctor_appointment.date`, set the year to 2023 when the user query contains no
   year (train purity 80% and 48/48 respectively).
2. **Omit rule** — drop `recipient_iban` / `insurance_number` when the value does not
   appear verbatim in the user text (train keeps them 276-vs-2 and 371-vs-26 only when
   verbatim).

## 4. Combiner and its tool registry

Per-argument majority vote across the six adapters, plus an **enum-validity** rule that
prefers a candidate appearing in a tool schema's `Supported values` list. The rule reads
`data/processed_v13/tools_registry.json`.

That registry was originally built with the parent corpus preferred as the schema source.
To remove any ambiguity under the 30 July clarification, we rebuilt it two further ways —
from the shared-task **train** split alone, and from the **blind test rows' own
`candidate_tools` schemas** — and all three are **byte-identical**, including all six
`Supported values` lists the rule actually consults. The parent corpus therefore
contributed no information to the submitted system. A clean-provenance copy exists as
`data/processed_v13/tools_registry_sharedtask_only.json`, available on request.

## 5. Reproducibility

`bash scripts/run_blind.sh data/processed_blind/test.jsonl <out>` regenerates the
submission deterministically (greedy decoding) from the six adapters in
`scripts/ensemble_config.json`. The same command on the dev split reproduces our reported
dev figures exactly: ArgEM 0.822 / OverallA 0.8925 / OverallB 0.9104 under v1.4 gold.
Adapters are LoRA deltas of about 80M trainable parameters each, 160M for the rank 64 member. They are not carried in the public repository because of their size and are available on request.

## 6. Known environment caveat

The pipeline was developed on torch 2.6+cu124 and re-run for the blind test on
torch 2.10.0+cu128 / transformers 5.5 / peft 0.18 / bitsandbytes 0.49. Numerical
differences from kernel changes are possible; the pinned original versions are recorded in
`scripts/ncc/setup_env.sh` should exact bit-level reproduction be required.
