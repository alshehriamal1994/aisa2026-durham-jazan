# Durham-Jazan at AISA-ArabicFC 2026

Code for our entry to the AISA-ArabicFC shared task on Arabic function calling, held at ArabicNLP 2026. The task gives a request in Arabic along with about four candidate tools, and asks for the right tool and a filled argument dictionary, or for `none` where no tool applies.

Our system is a majority vote taken separately on each argument, over six QLoRA adapters trained on two open 7B models, ALLaM-7B-Instruct and Qwen2.5-7B-Instruct. All training and inference ran on one 16 GB consumer card. The entry finished 9th of 17 in Track A, 8th of 17 in Track B, and 11th of 21 in the Track C dialect diagnostic.

One property of the data is worth knowing before you read the code. The reference tool is the first of the four candidates on all 11,000 training and development rows that call a tool, so a system that always answers the first candidate scores the same FnAcc as our six models. 
Re-running inference with the four candidates shuffled costs the voted system 0.202 FnAcc and 0.184 ArgEM on development, against the 0.323 a purely positional system would score. The dependence is very uneven across members, from -0.064 for the strongest to -0.650 for the rank-64 ALLaM member, which answers with the first candidate on 90.5 per cent of rows where the reference sits there on 26.2 per cent. The shuffled input is in `predictions/shuffled/`, together with each member's predictions on both orderings, `*_control.jsonl` for the original order and `*_shuffled.jsonl` for the shuffled one, both produced in the same environment on the same day so the comparison carries no drift. `scripts/score_shuffle_vote.py` reproduces the whole comparison from them without a GPU.

Retraining one adapter per base on the same data with the candidate order permuted removes the dependence. Shuffled-order FnAcc rises from 0.839 to 0.985 for ALLaM and from 0.936 to 0.996 for Qwen, at a canonical-order cost of 0.016 and 0.006 and no loss of ArgEM. `scripts/permute_candidates.py` builds the permuted training file (seed 20260816) and the four resulting prediction files are in `predictions/permtrain/`, scored by `scripts/score_permtrain.py`.

## Layout

    scripts/            training, inference, voting and the analysis code
    src/aisa/           prompt construction, normalisation, canonicalisation
    predictions/        the four files we submitted for the blind test
    predictions/members the six per-member blind predictions the vote was taken over
    docs/               our data use disclosure

Worth knowing about in `scripts/`:

    ensemble_vote.py        the combiner, and both convention rules
    blind_determinacy.py    two gold-free diagnostics for asking whether two splits
                            of a benchmark are equally determinate. Ensemble unanimity
                            as a proxy for how far the input pins the answer, and
                            nearest-train-twin similarity by character n-gram retrieval
                            with exact rescoring. Both run on inputs and model outputs
                            only, so neither needs gold.
    run_baseline_track_a.py an independent reimplementation of the official Track A
                            baseline, including the FunctionGemma DSL parser

## Running it

Be warned that a full run needs the task data and the six adapters, and neither is in this repository. What follows is what we ran, not a turnkey pipeline.

    python3 scripts/prepare_data.py          # writes data/processed
    python3 scripts/augment_negatives.py --input train.jsonl \
            --output train_st_aug.jsonl --source sharedtask
    python3 scripts/prepare_blind.py         # writes data/processed_blind/test.jsonl
    bash scripts/run_blind.sh data/processed_blind/test.jsonl out.jsonl

`prepare_data.py` merges the parent corpus into `train.jsonl`, which is not what the submitted members were trained on. The file we used is `train_st_aug.jsonl`, produced by `augment_negatives.py` with `--source sharedtask`, which keeps the parent rows out. `run_blind.sh` runs the six members listed in its own array at the top of the file, which must match `scripts/ensemble_config.json`, and then votes. Pointed at the development split it reproduces ArgEM 0.822 and Overall A 0.8925 against v1.4 gold.

Training is driven by `scripts/train_qlora.py`. Of the six members, `aC` and `qC` are produced by `scripts/chain_clean_zoo.sh`. The other four were trained during earlier rounds and their `autoeval_*.sh` scripts wait for a run to appear rather than starting one, so they will hang if you invoke them directly.

Decoding is greedy throughout, so runs repeat given the same environment. We developed on torch 2.6 and re-ran the blind inference on torch 2.10 without seeing a difference, though we have not checked this at the level of individual bits. There is no seed variation between members. They differ by base model, rank, epochs and data release.

## The two rules

Two deterministic rules are applied after decoding. Both were estimated from the training split alone and neither consults a development or test label.

Neither rule needs Arabic, knowledge of the domains, or any learned component. Both are a count over the training split, and that is why the paper reports them.

The year rule sets the year on hotel and appointment dates to 2023 when the request itself states no year. In the training release this holds for 666 of the 816 dated `search_hotels` fields, 81.6 per cent, and for all 48 dated `book_doctor_appointment` rows, whatever the stated context date happens to be.

The omit rule drops `recipient_iban` and `insurance_number` when the value does not appear word for word in the request. In the training release `recipient_iban` is kept 278 times and `insurance_number` 397 times, and in every one of those the value appears in the user text once Arabic-Indic digits are folded to ASCII. There are no counter-examples. Our implementation compares raw strings without folding, which on development costs two rows where the user wrote the number in Arabic-Indic digits.

Either rule can be switched off with `--no-year-rule` or `--no-omit-rule`, passed through `run_blind.sh` via the `RULES` variable:

    RULES="--no-omit-rule" bash scripts/run_blind.sh test.jsonl V2_year_only.jsonl

That is how the four variants in `predictions/` were produced, and they differ from one another on between 17 and 53 of the 1,125 rows.

The four variants differ only in the two post-processing rules, and V1, with both rules, was the pre-registered primary.

## What is not here

The task data are not redistributed. They belong to the organisers and can be obtained from the shared task page.

The adapters run to about 314 MB each, close to 1.9 GB for the six, which is more than this repository ought to carry. We are glad to share them on request.

The per-member blind predictions are here, in `predictions/members/`. They carry no gold and mean the combiner and both rules can be checked on a CPU in seconds, without the adapters and without a GPU.

The official evaluation code belongs to the organisers and is not copied here. Our scripts import it from a local checkout rather than vendoring it.

The scripts were written for one machine over several months and they show it. The shell scripts now resolve their own location rather than an absolute path, but two of the older experimental chains still point at a local Hugging Face cache, twenty-nine files, including the voting step itself, expect a checkout of the organisers' evaluator under `baselines/`, which is not included, and `run_infer.py` expects `data/processed/tools_registry.json` to exist at import. We have listed these rather than quietly rewriting history.

## For verifiers

The combiner and both convention rules can be checked without a GPU and without the
adapters. You need the six files in `predictions/members/`, which are here, and a
checkout of the organisers' evaluator, which is not.

    python3 scripts/ensemble_vote.py \
        --preds predictions/members/blind_{v10,v12,v7b,v7,aC,qC}.jsonl \
        --names v10 v12 v7b v7 aC qC \
        --train-conventions --input <blind test.jsonl> --out check.jsonl

`check.jsonl` should match `predictions/V1_year_and_omit.jsonl` row for row. Dropping
`--train-conventions` should match `V3_no_rules.jsonl`.

To reproduce the adapters themselves we would additionally need to give you the six
LoRA deltas, `data/processed/tools_registry.json` and `data/processed_v13/tools_registry.json`,
the `train_st_aug.jsonl` files for the v1.1 and v1.3 builds, and `runs/*.log` for the
tokenised-count fingerprints. Ask and we will send them.

Two honest caveats. The training recipes for `v10`, `v12`, `v7b` and `v7` are not in
this repository, since those runs predate the cleanup and only their `autoeval_*.sh`
consumers survive. And `scripts/ensemble_vote.py` expects the **v1.3** evaluator at
`baselines/leaderboard-code-v1_3/normalize.py` specifically, while `run_infer.py` and
`evaluate.py` want `baselines/leaderboard-code`. Scores in the paper are computed
against v1.4 gold with the v1.3 normaliser unless stated otherwise.

## Data use

`docs/RESOURCE_DISCLOSURE.md` sets out what each adapter was trained on, including the tokenised example counts recorded in the training logs, which identify the input file for every member unambiguously. The short version is the provided training split plus negatives derived from it, and nothing further. The development split was used to select ensemble members and to report scores. The blind test was read as inputs only.

## Citation

The system description paper, *Durham-Jazan at AISA-ArabicFC: A Field-Wise QLoRA Ensemble and the Sensitivity of Exact-Match Scoring*, is under review for ArabicNLP 2026. See `CITATION.cff`. We will add the full reference once that is settled.
