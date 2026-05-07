# Phase 1 Context Experiments

This folder contains the Phase 1 follow-up experiments that separate target-token
rank from subject/person-conditioned form selection. Existing root-level
`results/phase1_*.csv` files are left unchanged.


## Phase 0 Reference Inputs

The Phase 0 tokenizer outputs are copied to `../phase0/` and are part of this
report package. They define which comparisons are clean single-token cases and
which ones are sensitive to suffix/token continuation.

### Phase 0 Tokenization Summary

| type | count | share |
| --- | --- | --- |
| A_merged | 52 | 0.722 |
| B_suffix_split | 15 | 0.208 |
| C_multi_split | 5 | 0.069 |

### Phase 0 Pilot Position Reference

| person | target_form | compare_mode | target_tokens | target_token_ids | overt_primary_read_pos | prodrop_primary_read_pos | suffix_split |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1s | gidiyorum | single_token_logit_diff | gidiyorum | 23301 | 3 | 2 | False |
| 2s | gidiyorsun | suffix_probe_plus_sequence_logprob | gidiyor \| sun | 6215 \| 833 | 4 | 3 | True |
| 3s | gidiyor | self_contrast_baseline | gidiyor | 6215 | 3 | 2 | False |
| 1p | gidiyoruz | single_token_logit_diff | gidiyoruz | 19015 | 3 | 2 | False |
| 2p | gidiyorsunuz | suffix_probe_plus_sequence_logprob | gidiyor \| sunuz | 6215 \| 1711 | 4 | 3 | True |
| 3p | gidiyorlar | single_token_logit_diff | gidiyorlar | 41311 | 3 | 2 | False |


## Analysis Flow

1. Keep Phase 0 tokenizer and position references visible.
2. Archive the original pilot person-margin run.
3. Run three natural context experiments.
4. Run three final controls:
   - tokenization-matched verb control
   - mixed-token verb control
   - sum-vs-mean scoring sensitivity

## Main Conditions

- `ambiguous_prodrop`: no person cue before the target verb.
  Example: `Her gün okula gidiyorum.`
- `prior_predicate_person`: a previous finite predicate supplies person.
  Example: `Az önce geldim. Biraz dinleniyorum.`
- `possessive_person`: a possessive phrase supplies discourse-person evidence.
  Example: `Çantam hazır. Yola çıkıyorum.`

## Final Controls

- `control_token_matched_yapmak`: finite-predicate cue with `yapmak`, whose six
  present-progressive forms were single-token in Phase 0. This checks whether
  the result survives when candidate tokenization is mostly matched.
- `control_mixed_token_dinlenmek`: a clearer causal discourse version using
  `Çok yoruldum. Bu yüzden dinleniyorum.` with mixed-token candidate forms.
  This checks whether the prior-predicate effect is robust but still sensitive
  to tokenization.
- `sum_vs_mean_sensitivity`: compares final selection under total sequence
  log-probability (`sum`) vs. token-normalized sequence log-probability (`mean`).

## Main Metric

`person_margin_mean = score(correct form) - score(best wrong person form)`

Scores are layer-wise teacher-forced mean sequence log-probabilities. Positive
margin means the correct person form beats all five alternatives in the same
context.

## Files

- `phase1_original_person_form_*.csv`: archived rerun of the original pilot person-margin analysis.
- `phase1_context_person_form_specs.csv`: candidate-form specs for the three new conditions.
- `phase1_context_person_form_scores.csv`: layer-wise candidate scores for all six forms.
- `phase1_context_person_margin_layers.csv`: layer-wise correct-vs-best-wrong margins.
- `phase1_context_person_margin_transition_summary.csv`: first positive margin and final selection per case.
- `phase1_context_condition_summary.csv`: aggregate success rates by condition.
- `phase1_context_final_selection.csv`: final-layer selected form per condition/person.
- `phase1_context_tokenization.csv`: tokenizer breakdown for new candidate forms.
- `phase1_control_*.csv`: final control outputs.
- `phase1_all_sum_vs_mean_sensitivity*.csv`: scoring sensitivity across main
  conditions and controls.

## Figures

- `person_margin_by_condition`: `figures/person_margin_by_condition.html`, `figures/person_margin_by_condition.png`
- `final_margin_by_condition`: `figures/final_margin_by_condition.html`, `figures/final_margin_by_condition.png`
- `final_selected_person`: `figures/final_selected_person.html`, `figures/final_selected_person.png`
- `control_person_margin`: `figures/control_person_margin.html`, `figures/control_person_margin.png`
- `control_final_margin`: `figures/control_final_margin.html`, `figures/control_final_margin.png`
- `sum_vs_mean_sensitivity`: `figures/sum_vs_mean_sensitivity.html`, `figures/sum_vs_mean_sensitivity.png`

## Aggregate Context Results

| condition | n_cases | n_final_correct | final_correct_rate | n_ever_positive_margin | ever_positive_rate | mean_final_margin | median_final_margin | median_first_positive_layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_prodrop | 6 | 1 | 0.167 | 1 | 0.167 | -0.814 | -0.528 | 0.0 |
| possessive_person | 6 | 3 | 0.5 | 5 | 0.833 | 0.385 | 0.12 | 6.0 |
| prior_predicate_person | 6 | 5 | 0.833 | 5 | 0.833 | 0.289 | 0.185 | 2.0 |

## Original Pilot Final Layer

| variant | actual_person | correct_candidate_form | person_margin_mean | correct_form_rank_mean | selected_person_mean | selected_form_mean | selected_is_correct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overt | 1p | gidiyoruz | 0.056 | 1 | 1p | gidiyoruz | True |
| prodrop | 1p | gidiyoruz | -1.396 | 5 | 2p | gidiyorsunuz | False |
| overt | 1s | gidiyorum | 0.697 | 1 | 1s | gidiyorum | True |
| prodrop | 1s | gidiyorum | -0.428 | 2 | 2p | gidiyorsunuz | False |
| overt | 2p | gidiyorsunuz | 0.591 | 1 | 2p | gidiyorsunuz | True |
| prodrop | 2p | gidiyorsunuz | 0.428 | 1 | 2p | gidiyorsunuz | True |
| overt | 2s | gidiyorsun | 0.334 | 1 | 2s | gidiyorsun | True |
| prodrop | 2s | gidiyorsun | -0.546 | 4 | 2p | gidiyorsunuz | False |
| overt | 3p | gidiyorlar | -0.682 | 2 | 3s | gidiyor | False |
| prodrop | 3p | gidiyorlar | -2.431 | 6 | 2p | gidiyorsunuz | False |
| overt | 3s | gidiyor | 2.208 | 1 | 3s | gidiyor | True |
| prodrop | 3s | gidiyor | -0.509 | 3 | 2p | gidiyorsunuz | False |

## New Context Final Layer

| condition | actual_person | correct_candidate_form | person_margin_mean | correct_form_rank_mean | selected_person_mean | selected_form_mean | selected_is_correct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_prodrop | 1p | gidiyoruz | -1.396 | 5 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 1s | gidiyorum | -0.428 | 2 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 2p | gidiyorsunuz | 0.428 | 1 | 2p | gidiyorsunuz | True |
| ambiguous_prodrop | 2s | gidiyorsun | -0.546 | 4 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 3p | gidiyorlar | -2.431 | 6 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 3s | gidiyor | -0.509 | 3 | 2p | gidiyorsunuz | False |
| possessive_person | 1p | çıkıyoruz | 0.818 | 1 | 1p | çıkıyoruz | True |
| possessive_person | 1s | çıkıyorum | 2.914 | 1 | 1s | çıkıyorum | True |
| possessive_person | 2p | çıkıyorsunuz | 0.746 | 1 | 2p | çıkıyorsunuz | True |
| possessive_person | 2s | çıkıyorsun | -0.675 | 4 | 1s | çıkıyorum | False |
| possessive_person | 3p | çıkıyorlar | -0.505 | 3 | 2p | çıkıyorsunuz | False |
| possessive_person | 3s | çıkıyor | -0.987 | 5 | 2p | çıkıyorsunuz | False |
| prior_predicate_person | 1p | dinleniyoruz | 0.292 | 1 | 1p | dinleniyoruz | True |
| prior_predicate_person | 1s | dinleniyorum | 0.074 | 1 | 1s | dinleniyorum | True |
| prior_predicate_person | 2p | dinleniyorsunuz | 1.268 | 1 | 2p | dinleniyorsunuz | True |
| prior_predicate_person | 2s | dinleniyorsun | 1.151 | 1 | 2s | dinleniyorsun | True |
| prior_predicate_person | 3p | dinleniyorlar | 0.078 | 1 | 3p | dinleniyorlar | True |
| prior_predicate_person | 3s | dinleniyor | -1.127 | 5 | 2p | dinleniyorsunuz | False |

## New Context Transition Summary

| condition | actual_person | correct_candidate_form | first_positive_margin_layer | final_margin | final_correct_form_rank_mean | final_selected_person_mean | final_selected_form_mean | final_selected_is_correct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_prodrop | 1p | gidiyoruz |  | -1.396 | 5 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 1s | gidiyorum |  | -0.428 | 2 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 2p | gidiyorsunuz | 0.0 | 0.428 | 1 | 2p | gidiyorsunuz | True |
| ambiguous_prodrop | 2s | gidiyorsun |  | -0.546 | 4 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 3p | gidiyorlar |  | -2.431 | 6 | 2p | gidiyorsunuz | False |
| ambiguous_prodrop | 3s | gidiyor |  | -0.509 | 3 | 2p | gidiyorsunuz | False |
| possessive_person | 1p | çıkıyoruz | 6.0 | 0.818 | 1 | 1p | çıkıyoruz | True |
| possessive_person | 1s | çıkıyorum | 8.0 | 2.914 | 1 | 1s | çıkıyorum | True |
| possessive_person | 2p | çıkıyorsunuz | 0.0 | 0.746 | 1 | 2p | çıkıyorsunuz | True |
| possessive_person | 2s | çıkıyorsun | 2.0 | -0.675 | 4 | 1s | çıkıyorum | False |
| possessive_person | 3p | çıkıyorlar |  | -0.505 | 3 | 2p | çıkıyorsunuz | False |
| possessive_person | 3s | çıkıyor | 6.0 | -0.987 | 5 | 2p | çıkıyorsunuz | False |
| prior_predicate_person | 1p | dinleniyoruz | 11.0 | 0.292 | 1 | 1p | dinleniyoruz | True |
| prior_predicate_person | 1s | dinleniyorum | 0.0 | 0.074 | 1 | 1s | dinleniyorum | True |
| prior_predicate_person | 2p | dinleniyorsunuz | 0.0 | 1.268 | 1 | 2p | dinleniyorsunuz | True |
| prior_predicate_person | 2s | dinleniyorsun | 2.0 | 1.151 | 1 | 2s | dinleniyorsun | True |
| prior_predicate_person | 3p | dinleniyorlar | 10.0 | 0.078 | 1 | 3p | dinleniyorlar | True |
| prior_predicate_person | 3s | dinleniyor |  | -1.127 | 5 | 2p | dinleniyorsunuz | False |

## Final Control Results

### Control Aggregate Results

| condition | n_cases | n_final_correct | final_correct_rate | n_ever_positive_margin | ever_positive_rate | mean_final_margin | median_final_margin | median_first_positive_layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mixed_token_dinlenmek | 6 | 4 | 0.667 | 5 | 0.833 | 0.325 | 0.23 | 9.0 |
| control_token_matched_yapmak | 6 | 3 | 0.5 | 3 | 0.5 | -0.04 | -0.459 | 1.0 |

### Control Final Layer

| condition | actual_person | correct_candidate_form | person_margin_mean | correct_form_rank_mean | selected_person_mean | selected_form_mean | selected_is_correct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| control_mixed_token_dinlenmek | 1p | dinleniyoruz | 0.231 | 1 | 1p | dinleniyoruz | True |
| control_mixed_token_dinlenmek | 1s | dinleniyorum | 0.228 | 1 | 1s | dinleniyorum | True |
| control_mixed_token_dinlenmek | 2p | dinleniyorsunuz | 1.127 | 1 | 2p | dinleniyorsunuz | True |
| control_mixed_token_dinlenmek | 2s | dinleniyorsun | 1.068 | 1 | 2s | dinleniyorsun | True |
| control_mixed_token_dinlenmek | 3p | dinleniyorlar | -0.281 | 3 | 2s | dinleniyorsun | False |
| control_mixed_token_dinlenmek | 3s | dinleniyor | -0.423 | 3 | 2s | dinleniyorsun | False |
| control_token_matched_yapmak | 1p | yapıyoruz | 1.799 | 1 | 1p | yapıyoruz | True |
| control_token_matched_yapmak | 1s | yapıyorum | 1.692 | 1 | 1s | yapıyorum | True |
| control_token_matched_yapmak | 2p | yapıyorsunuz | -1.326 | 4 | 1p | yapıyoruz | False |
| control_token_matched_yapmak | 2s | yapıyorsun | -1.486 | 4 | 1s | yapıyorum | False |
| control_token_matched_yapmak | 3p | yapıyorlar | -0.953 | 4 | 1p | yapıyoruz | False |
| control_token_matched_yapmak | 3s | yapıyor | 0.035 | 1 | 3s | yapıyor | True |

### Control Transition Summary

| condition | actual_person | correct_candidate_form | first_positive_margin_layer | final_margin | final_correct_form_rank_mean | final_selected_person_mean | final_selected_form_mean | final_selected_is_correct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mixed_token_dinlenmek | 1p | dinleniyoruz | 11.0 | 0.231 | 1 | 1p | dinleniyoruz | True |
| control_mixed_token_dinlenmek | 1s | dinleniyorum | 10.0 | 0.228 | 1 | 1s | dinleniyorum | True |
| control_mixed_token_dinlenmek | 2p | dinleniyorsunuz | 1.0 | 1.127 | 1 | 2p | dinleniyorsunuz | True |
| control_mixed_token_dinlenmek | 2s | dinleniyorsun | 8.0 | 1.068 | 1 | 2s | dinleniyorsun | True |
| control_mixed_token_dinlenmek | 3p | dinleniyorlar | 9.0 | -0.281 | 3 | 2s | dinleniyorsun | False |
| control_mixed_token_dinlenmek | 3s | dinleniyor |  | -0.423 | 3 | 2s | dinleniyorsun | False |
| control_token_matched_yapmak | 1p | yapıyoruz | 0.0 | 1.799 | 1 | 1p | yapıyoruz | True |
| control_token_matched_yapmak | 1s | yapıyorum | 8.0 | 1.692 | 1 | 1s | yapıyorum | True |
| control_token_matched_yapmak | 2p | yapıyorsunuz |  | -1.326 | 4 | 1p | yapıyoruz | False |
| control_token_matched_yapmak | 2s | yapıyorsun |  | -1.486 | 4 | 1s | yapıyorum | False |
| control_token_matched_yapmak | 3p | yapıyorlar |  | -0.953 | 4 | 1p | yapıyoruz | False |
| control_token_matched_yapmak | 3s | yapıyor | 1.0 | 0.035 | 1 | 3s | yapıyor | True |

### Tokenization-Matched vs. Mixed-Token Summary

| condition | candidate_type | count | mean_candidate_n_tokens |
| --- | --- | --- | --- |
| control_mixed_token_dinlenmek | B_suffix_split | 2 | 2.333 |
| control_mixed_token_dinlenmek | C_multi_split | 4 | 2.333 |
| control_token_matched_yapmak | A_merged | 6 | 1.0 |

### Sum vs. Mean Sensitivity

| condition | n_cases | mean_final_correct_rate | sum_final_correct_rate | selection_agreement_rate | correctness_agreement_rate | n_selection_disagreements | n_correctness_disagreements |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_prodrop | 6 | 0.167 | 0.167 | 0.0 | 0.667 | 6 | 2 |
| control_mixed_token_dinlenmek | 6 | 0.667 | 0.667 | 0.333 | 0.333 | 4 | 4 |
| control_token_matched_yapmak | 6 | 0.5 | 0.5 | 1.0 | 1.0 | 0 | 0 |
| possessive_person | 6 | 0.5 | 0.5 | 0.333 | 0.667 | 4 | 2 |
| prior_predicate_person | 6 | 0.833 | 0.5 | 0.5 | 0.667 | 3 | 2 |

## Interpretation

- `ambiguous_prodrop` should not be expected to select every target person:
  the prefix is genuinely person-ambiguous.
- `prior_predicate_person` tests whether person from a previous finite predicate
  transfers to the next pro-drop clause.
- `possessive_person` tests whether possessive morphology can act as a discourse
  person cue; this is weaker evidence than a finite predicate because the local
  subject of the first sentence is the possessed noun.
- Very early success for suffix-split or frequent forms should be treated as
  lexical/token-continuation evidence until controlled by the candidate-margin
  comparison.
- The controls should be read last. If a result only appears under `mean` but
  disappears under `sum`, or only appears for mixed-token forms, the claim is
  tokenization-sensitive and should not be used as direct evidence of an
  abstract subject/person mechanism.
