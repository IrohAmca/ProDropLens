# ProDropLens — Core Package
from .dataset import (
    MinimalPair, PatchingPair,
    Person, Tense, Verb,
    VERB_FORMS, SUBJECT_PRONOUNS, TENSE_CONTEXT,
    get_pilot, get_core, get_patching, get_all_minimal_pairs,
    filter_pairs,
)
from .utils import (
    VerbSpan,
    tokenize, analyze_sentence, compare_tokenizations,
    find_token_position, find_verb_span, classify_verb_tokenization,
    get_token_id, get_token_ids_for_forms,
    tokens_to_tensor, get_logit_diff, get_top_predictions, get_token_rank,
)
from .logit_lens import (
    build_logit_lens_specs, build_person_form_specs,
    compare_overt_prodrop,
    load_lens_model, run_logit_lens, run_logit_lens_for_spec,
    run_person_form_lens, run_person_form_lens_for_spec,
    summarize_person_margins, summarize_person_margin_transitions,
    summarize_rank_transitions,
)
from .phase1_experiments import (
    build_phase1_context_cases, build_phase1_context_person_specs,
    build_phase1_control_cases, build_phase1_control_person_specs,
    summarize_context_conditions, summarize_sum_vs_mean_sensitivity,
)


__all__ = [
    "MinimalPair", "PatchingPair",
    "Person", "Tense", "Verb",
    "VERB_FORMS", "SUBJECT_PRONOUNS", "TENSE_CONTEXT",
    "get_pilot", "get_core", "get_patching", "get_all_minimal_pairs",
    "filter_pairs",
    "VerbSpan",
    "tokenize", "analyze_sentence", "compare_tokenizations",
    "find_token_position", "find_verb_span", "classify_verb_tokenization",
    "get_token_id", "get_token_ids_for_forms",
    "tokens_to_tensor", "get_logit_diff", "get_top_predictions", "get_token_rank",
    "build_logit_lens_specs", "build_person_form_specs",
    "compare_overt_prodrop",
    "load_lens_model", "run_logit_lens", "run_logit_lens_for_spec",
    "run_person_form_lens", "run_person_form_lens_for_spec",
    "summarize_person_margins", "summarize_person_margin_transitions",
    "summarize_rank_transitions",
    "build_phase1_context_cases", "build_phase1_context_person_specs",
    "build_phase1_control_cases", "build_phase1_control_person_specs",
    "summarize_context_conditions", "summarize_sum_vs_mean_sensitivity",
]
