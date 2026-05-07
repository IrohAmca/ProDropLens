"""
Phase 1 context experiments.

These fixtures extend the original pilot with more natural pro-drop contexts:

- ambiguous_prodrop: no person cue before the target verb
- prior_predicate_person: a previous finite predicate supplies person
- possessive_person: a possessive phrase supplies discourse-person evidence

The output is intentionally compatible with ``run_person_form_lens`` in
``core.logit_lens``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dataset import Person, Tense, Verb, VERB_FORMS
from .logit_lens import _fmt_list, _form_encoding
from .utils import analyze_sentence


PERSON_ORDER: tuple[Person, ...] = (
    Person.S1,
    Person.S2,
    Person.S3,
    Person.P1,
    Person.P2,
    Person.P3,
)


@dataclass(frozen=True)
class Phase1ContextCase:
    """One natural-context person-selection test case."""

    condition: str
    condition_label: str
    cue_type: str
    person: Person
    prefix_text: str
    target_form: str
    candidate_forms: dict[Person, str]
    notes: str = ""

    @property
    def pair_id(self) -> str:
        return f"phase1_{self.condition}_{self.person.value}"

    @property
    def full_text(self) -> str:
        return f"{self.prefix_text}{self.target_form}."


DINLENMEK_SIMDIKI: dict[Person, str] = {
    Person.S1: "dinleniyorum",
    Person.S2: "dinleniyorsun",
    Person.S3: "dinleniyor",
    Person.P1: "dinleniyoruz",
    Person.P2: "dinleniyorsunuz",
    Person.P3: "dinleniyorlar",
}


CIKMAK_SIMDIKI: dict[Person, str] = {
    Person.S1: "çıkıyorum",
    Person.S2: "çıkıyorsun",
    Person.S3: "çıkıyor",
    Person.P1: "çıkıyoruz",
    Person.P2: "çıkıyorsunuz",
    Person.P3: "çıkıyorlar",
}


YAPMAK_SIMDIKI: dict[Person, str] = VERB_FORMS[Verb.YAPMAK][Tense.SIMDIKI]


PRIOR_PREDICATE_PREFIX: dict[Person, str] = {
    Person.S1: "Az önce geldim. Biraz ",
    Person.S2: "Az önce geldin. Biraz ",
    Person.S3: "Az önce geldi. Biraz ",
    Person.P1: "Az önce geldik. Biraz ",
    Person.P2: "Az önce geldiniz. Biraz ",
    Person.P3: "Az önce geldiler. Biraz ",
}


CAUSAL_DINLENMEK_PREFIX: dict[Person, str] = {
    Person.S1: "Çok yoruldum. Bu yüzden ",
    Person.S2: "Çok yoruldun. Bu yüzden ",
    Person.S3: "Çok yoruldu. Bu yüzden ",
    Person.P1: "Çok yorulduk. Bu yüzden ",
    Person.P2: "Çok yoruldunuz. Bu yüzden ",
    Person.P3: "Çok yoruldular. Bu yüzden ",
}


MATCHED_YAPMAK_PREFIX: dict[Person, str] = {
    Person.S1: "Çok acıktım. Bu yüzden kahvaltı ",
    Person.S2: "Çok acıktın. Bu yüzden kahvaltı ",
    Person.S3: "Çok acıktı. Bu yüzden kahvaltı ",
    Person.P1: "Çok acıktık. Bu yüzden kahvaltı ",
    Person.P2: "Çok acıktınız. Bu yüzden kahvaltı ",
    Person.P3: "Çok acıktılar. Bu yüzden kahvaltı ",
}


POSSESSIVE_PREFIX: dict[Person, str] = {
    Person.S1: "Çantam hazır. Yola ",
    Person.S2: "Çantan hazır. Yola ",
    Person.S3: "Çantası hazır. Yola ",
    Person.P1: "Çantamız hazır. Yola ",
    Person.P2: "Çantanız hazır. Yola ",
    Person.P3: "Çantaları hazır. Yola ",
}


def build_phase1_context_cases() -> list[Phase1ContextCase]:
    """Return the natural-context Phase 1 cases."""
    cases: list[Phase1ContextCase] = []
    gitmek_forms = VERB_FORMS[Verb.GITMEK][Tense.SIMDIKI]

    for person in PERSON_ORDER:
        cases.append(Phase1ContextCase(
            condition="ambiguous_prodrop",
            condition_label="Ambiguous pro-drop",
            cue_type="none",
            person=person,
            prefix_text="Her gün okula ",
            target_form=gitmek_forms[person],
            candidate_forms=gitmek_forms,
            notes="No overt person cue before the target verb.",
        ))

    for person in PERSON_ORDER:
        cases.append(Phase1ContextCase(
            condition="prior_predicate_person",
            condition_label="Prior predicate person",
            cue_type="finite_predicate",
            person=person,
            prefix_text=PRIOR_PREDICATE_PREFIX[person],
            target_form=DINLENMEK_SIMDIKI[person],
            candidate_forms=DINLENMEK_SIMDIKI,
            notes="Previous sentence has a finite predicate with matching person.",
        ))

    for person in PERSON_ORDER:
        cases.append(Phase1ContextCase(
            condition="possessive_person",
            condition_label="Possessive person",
            cue_type="possessive_suffix",
            person=person,
            prefix_text=POSSESSIVE_PREFIX[person],
            target_form=CIKMAK_SIMDIKI[person],
            candidate_forms=CIKMAK_SIMDIKI,
            notes="Possessive phrase supplies discourse-person evidence.",
        ))

    return cases


def build_phase1_control_cases() -> list[Phase1ContextCase]:
    """Return control cases used after the main Phase 1 context experiments."""
    cases: list[Phase1ContextCase] = []

    for person in PERSON_ORDER:
        cases.append(Phase1ContextCase(
            condition="control_token_matched_yapmak",
            condition_label="Control: token-matched yapmak",
            cue_type="finite_predicate_control",
            person=person,
            prefix_text=MATCHED_YAPMAK_PREFIX[person],
            target_form=YAPMAK_SIMDIKI[person],
            candidate_forms=YAPMAK_SIMDIKI,
            notes="All six target forms are expected to be single-token forms.",
        ))

    for person in PERSON_ORDER:
        cases.append(Phase1ContextCase(
            condition="control_mixed_token_dinlenmek",
            condition_label="Control: mixed-token dinlenmek",
            cue_type="finite_predicate_control",
            person=person,
            prefix_text=CAUSAL_DINLENMEK_PREFIX[person],
            target_form=DINLENMEK_SIMDIKI[person],
            candidate_forms=DINLENMEK_SIMDIKI,
            notes="Same causal discourse pattern, but mixed tokenization across candidate forms.",
        ))

    return cases


def _build_specs_for_cases(tokenizer, cases: list[Phase1ContextCase]) -> pd.DataFrame:
    """Build ``run_person_form_lens`` specs for concrete Phase 1 cases."""
    rows: list[dict] = []

    for case in cases:
        for candidate_person in PERSON_ORDER:
            candidate_form = case.candidate_forms[candidate_person]
            candidate_info = _form_encoding(tokenizer, candidate_form)
            candidate_text = f"{case.prefix_text}{candidate_form}"
            sentence_info = analyze_sentence(tokenizer, candidate_text, None, candidate_form)

            token_positions = []
            read_positions = []
            if sentence_info["verb_pos"] is not None and sentence_info["verb_end"] is not None:
                token_positions = list(range(sentence_info["verb_pos"], sentence_info["verb_end"] + 1))
                read_positions = [pos - 1 if pos > 0 else None for pos in token_positions]

            rows.append({
                "pair_id": case.pair_id,
                "condition": case.condition,
                "condition_label": case.condition_label,
                "cue_type": case.cue_type,
                "variant": case.condition,
                "actual_person": case.person.value,
                "candidate_person": candidate_person.value,
                "is_correct": candidate_person == case.person,
                "subject": "",
                "prefix_text": case.prefix_text,
                "full_text": case.full_text,
                "target_form": case.target_form,
                "candidate_form": candidate_form,
                "candidate_text": candidate_text,
                "candidate_type": candidate_info["type"],
                "candidate_n_tokens": candidate_info["n_tokens"],
                "candidate_tokens": _fmt_list(candidate_info["tokens"]),
                "candidate_token_ids": _fmt_list(candidate_info["token_ids"]),
                "verb_start": sentence_info["verb_pos"],
                "verb_end": sentence_info["verb_end"],
                "candidate_token_positions": _fmt_list(token_positions),
                "candidate_read_positions": _fmt_list(read_positions),
                "sentence_tokens": _fmt_list(sentence_info["token_strs"]),
                "sentence_token_ids": _fmt_list(sentence_info["token_ids"]),
                "notes": case.notes,
            })

    return pd.DataFrame(rows)


def build_phase1_context_person_specs(tokenizer) -> pd.DataFrame:
    """Build specs for the main Phase 1 context cases."""
    return _build_specs_for_cases(tokenizer, build_phase1_context_cases())


def build_phase1_control_person_specs(tokenizer) -> pd.DataFrame:
    """Build specs for the final Phase 1 control cases."""
    return _build_specs_for_cases(tokenizer, build_phase1_control_cases())


def summarize_context_conditions(margin_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final person-selection behavior by context condition."""
    rows: list[dict] = []

    for condition, group in margin_summary_df.groupby("condition", dropna=False):
        final_correct = group["final_selected_is_correct_mean"].astype(bool)
        positive = group["first_positive_margin_layer"].notna()
        first_positive = group.loc[positive, "first_positive_margin_layer"]

        rows.append({
            "condition": condition,
            "n_cases": len(group),
            "n_final_correct": int(final_correct.sum()),
            "final_correct_rate": float(final_correct.mean()),
            "n_ever_positive_margin": int(positive.sum()),
            "ever_positive_rate": float(positive.mean()),
            "mean_final_margin": float(group["final_margin"].mean()),
            "median_final_margin": float(group["final_margin"].median()),
            "median_first_positive_layer": (
                float(first_positive.median()) if not first_positive.empty else None
            ),
        })

    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def summarize_sum_vs_mean_sensitivity(margin_df: pd.DataFrame) -> pd.DataFrame:
    """Compare final-layer person selection under sum vs. mean scoring."""
    final = margin_df[margin_df["layer"] == margin_df["layer"].max()].copy()
    rows: list[dict] = []

    for _, row in final.iterrows():
        rows.append({
            "condition": row.get("condition"),
            "actual_person": row["actual_person"],
            "correct_candidate_form": row["correct_candidate_form"],
            "mean_margin": row["person_margin_mean"],
            "sum_margin": row["person_margin_sum"],
            "mean_rank": row["correct_form_rank_mean"],
            "sum_rank": row["correct_form_rank_sum"],
            "mean_selected_person": row["selected_person_mean"],
            "sum_selected_person": row["selected_person_sum"],
            "mean_selected_form": row["selected_form_mean"],
            "sum_selected_form": row["selected_form_sum"],
            "mean_selected_is_correct": row["selected_is_correct_mean"],
            "sum_selected_is_correct": row["selected_is_correct_sum"],
            "selection_agrees": (
                row["selected_person_mean"] == row["selected_person_sum"]
                and row["selected_form_mean"] == row["selected_form_sum"]
            ),
            "correctness_agrees": (
                bool(row["selected_is_correct_mean"]) == bool(row["selected_is_correct_sum"])
            ),
        })

    return pd.DataFrame(rows)
