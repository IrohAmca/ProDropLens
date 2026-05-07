# Phase 0 Tokenizer Reference

This folder contains the tokenizer-discovery outputs used by Phase 1.

## Why This Matters

Turkish person/agreement information is not represented with a consistent token
boundary in `ytu-ce-cosmos/turkish-gpt2`. Some inflected verbs are one merged
token, some split the person suffix into a separate token, and some split at a
non-morphological boundary. This means a raw target-token rank can mix at least
three effects:

- actual subject/person information,
- lexical continuation from the preceding verb fragment,
- tokenizer length and boundary advantages.

Phase 1 therefore treats Phase 0 as a required reference table rather than a
preprocessing detail. Person-form comparisons are reported with tokenization
type, token IDs, read positions, and both token-normalized (`mean`) and total
sequence (`sum`) scoring controls.

## Files

- `phase0_verb_classification.csv`: tokenization type for each verb/person/tense form.
- `phase0_position_reference.csv`: pilot sentence token positions and read positions.
- `phase0_tokenization_type_summary.csv`: aggregate tokenizer split counts.
- `phase0_tokenization_by_verb.csv`: tokenizer split counts by verb.
- `phase0_tokenization_by_person.csv`: tokenizer split counts by person.
- `phase0_tokenization_by_tense.csv`: tokenizer split counts by tense.

## Tokenization Type Summary

| type | count | share |
| --- | --- | --- |
| A_merged | 52 | 0.722 |
| B_suffix_split | 15 | 0.208 |
| C_multi_split | 5 | 0.069 |

## Tokenization Types and Consequences

| type | Meaning | Consequence |
| --- | --- | --- |
| `A_merged` | Whole inflected form is one token, e.g. `gidiyorum`. | Clean single-token rank/logit comparison is possible, but person information is lexicalized inside one token. |
| `B_suffix_split` | Verb stem/tense and person suffix split, e.g. `gidiyor | sun`. | Suffix rank can look good very early because it may be a local continuation from the previous token, not necessarily subject reasoning. |
| `C_multi_split` | The form splits across a less clean boundary, e.g. `dinlen | iyorum` or `git | tin`. | Full-form sequence scoring is required; single-token interpretations are unsafe. |

The main risk is comparing forms with different token counts. Sum scoring
penalizes longer forms because every extra token adds another probability term.
Mean scoring reduces that length penalty, but it can over-reward forms with an
easy suffix token. Phase 1 keeps both views in the final controls.

## Pilot Position Reference

| person | target_form | compare_mode | target_tokens | target_token_ids | overt_primary_read_pos | prodrop_primary_read_pos | suffix_split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | gidiyorum | single_token_logit_diff | gidiyorum | 23301 | 3 | 2 | False |
| 2s | gidiyorsun | suffix_probe_plus_sequence_logprob | gidiyor \| sun | 6215 \| 833 | 4 | 3 | True |
| 3s | gidiyor | self_contrast_baseline | gidiyor | 6215 | 3 | 2 | False |
| 1p | gidiyoruz | single_token_logit_diff | gidiyoruz | 19015 | 3 | 2 | False |
| 2p | gidiyorsunuz | suffix_probe_plus_sequence_logprob | gidiyor \| sunuz | 6215 \| 1711 | 4 | 3 | True |
| 3p | gidiyorlar | single_token_logit_diff | gidiyorlar | 41311 | 3 | 2 | False |
