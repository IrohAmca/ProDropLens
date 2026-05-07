"""
Run Phase 1 natural-context person-selection experiments.

Outputs are written under results/phase1/ so the original Phase 1 CSV files in
results/ remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset import get_pilot  # noqa: E402
from core.logit_lens import (  # noqa: E402
    MODEL_NAME,
    build_person_form_specs,
    load_hf_lens_model,
    load_lens_model,
    run_person_form_lens,
    summarize_person_margin_transitions,
    summarize_person_margins,
)
from core.phase1_experiments import (  # noqa: E402
    build_phase1_control_person_specs,
    build_phase1_context_person_specs,
    summarize_context_conditions,
    summarize_sum_vs_mean_sensitivity,
)


RESULTS_DIR = PROJECT_ROOT / "results"
PHASE0_DIR = RESULTS_DIR / "phase0"
PHASE1_DIR = RESULTS_DIR / "phase1"
FIGURES_DIR = PHASE1_DIR / "figures"
DEFAULT_MODEL_COMPARISON_MODELS = [
    "ytu-ce-cosmos/turkish-gpt2-medium",
    "ytu-ce-cosmos/turkish-gpt2-large",
]
MODEL_ALIASES = {
    "small": "ytu-ce-cosmos/turkish-gpt2",
    "base": "ytu-ce-cosmos/turkish-gpt2",
    "medium": "ytu-ce-cosmos/turkish-gpt2-medium",
    "large": "ytu-ce-cosmos/turkish-gpt2-large",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 1 context, control, and model-comparison experiments.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Baseline HF model name for the main Phase 1 run.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype for the baseline model, e.g. float32 or float16.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device passed to TransformerLens. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--comparison-models",
        nargs="*",
        default=DEFAULT_MODEL_COMPARISON_MODELS,
        help=(
            "Additional HF model names or aliases to compare against the baseline. "
            "Use 'none' to skip. Aliases: small, medium, large."
        ),
    )
    parser.add_argument(
        "--comparison-dtype",
        default="float16",
        help="Torch dtype for comparison models.",
    )
    parser.add_argument(
        "--skip-model-comparison",
        action="store_true",
        help="Run only the original Phase 1 context and control experiments.",
    )
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    """Resolve short model aliases to HF model names."""
    return MODEL_ALIASES.get(model_name, model_name)


def normalize_comparison_models(model_names: list[str], baseline_model_name: str) -> list[str]:
    """Return de-duplicated comparison models, excluding the baseline."""
    if any(name.lower() == "none" for name in model_names):
        return []

    normalized: list[str] = []
    seen = {baseline_model_name}
    for name in model_names:
        resolved = normalize_model_name(name)
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)
    return normalized


def model_label(model_name: str) -> str:
    """Return a compact model label for tables and plots."""
    return model_name.rsplit("/", maxsplit=1)[-1]


def model_metadata(model, model_name: str) -> dict:
    """Collect stable model metadata used in comparison tables."""
    if hasattr(model, "cfg"):
        n_layers = int(model.cfg.n_layers)
        d_model = int(model.cfg.d_model)
        n_heads = int(model.cfg.n_heads)
    else:
        n_layers = int(model.config.n_layer)
        d_model = int(model.config.n_embd)
        n_heads = int(model.config.n_head)
    return {
        "model_label": model_label(model_name),
        "model_name": model_name,
        "model_n_layers": n_layers,
        "model_d_model": d_model,
        "model_n_heads": n_heads,
        "model_n_params": int(sum(param.numel() for param in model.parameters())),
    }


def add_model_metadata(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Attach model metadata columns to a result table."""
    out = df.copy()
    for key, value in metadata.items():
        out[key] = value
    front_cols = list(metadata)
    other_cols = [col for col in out.columns if col not in front_cols]
    return out[front_cols + other_cols]


def release_model(model) -> None:
    """Drop a model reference and return cached GPU memory to the allocator."""
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"

    safe = df.copy()
    safe = safe.where(pd.notna(safe), "")
    columns = list(safe.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in safe.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_figure(fig, stem: str) -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    html_path = FIGURES_DIR / f"{stem}.html"
    fig.write_html(html_path)
    outputs = [html_path.relative_to(PHASE1_DIR).as_posix()]

    png_path = FIGURES_DIR / f"{stem}.png"
    try:
        fig.write_image(png_path, scale=2)
        outputs.append(png_path.relative_to(PHASE1_DIR).as_posix())
    except Exception as exc:  # pragma: no cover - depends on local kaleido setup
        outputs.append(f"PNG export skipped: {exc}")

    return outputs


def load_original_pilot_summary() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    transition_path = RESULTS_DIR / "phase1_person_margin_transition_summary.csv"
    margin_path = RESULTS_DIR / "phase1_person_margin_layers.csv"
    transition = pd.read_csv(transition_path) if transition_path.exists() else None
    margin = pd.read_csv(margin_path) if margin_path.exists() else None
    return transition, margin


def prepare_phase0_reference() -> dict[str, pd.DataFrame]:
    """Copy Phase 0 outputs into results/phase0 and add compact summaries."""
    PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    verb_path = RESULTS_DIR / "phase0_verb_classification.csv"
    position_path = RESULTS_DIR / "phase0_position_reference.csv"
    if not verb_path.exists():
        verb_path = PHASE0_DIR / "phase0_verb_classification.csv"
    if not position_path.exists():
        position_path = PHASE0_DIR / "phase0_position_reference.csv"

    if not verb_path.exists() or not position_path.exists():
        return {}

    verb_df = pd.read_csv(verb_path)
    position_df = pd.read_csv(position_path)

    write_table(verb_df, PHASE0_DIR / "phase0_verb_classification.csv")
    write_table(position_df, PHASE0_DIR / "phase0_position_reference.csv")

    type_summary = (
        verb_df["type"]
        .value_counts()
        .rename_axis("type")
        .reset_index(name="count")
    )
    type_summary["share"] = type_summary["count"] / type_summary["count"].sum()

    by_verb = pd.crosstab(verb_df["verb"], verb_df["type"]).reset_index()
    by_person = pd.crosstab(verb_df["person"], verb_df["type"]).reset_index()
    by_tense = pd.crosstab(verb_df["tense"], verb_df["type"]).reset_index()

    write_table(type_summary, PHASE0_DIR / "phase0_tokenization_type_summary.csv")
    write_table(by_verb, PHASE0_DIR / "phase0_tokenization_by_verb.csv")
    write_table(by_person, PHASE0_DIR / "phase0_tokenization_by_person.csv")
    write_table(by_tense, PHASE0_DIR / "phase0_tokenization_by_tense.csv")

    phase0_readme = f"""# Phase 0 Tokenizer Reference

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

{markdown_table(type_summary.assign(share=type_summary["share"].round(3)))}

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

{markdown_table(position_df[[
    "person", "target_form", "compare_mode", "target_tokens", "target_token_ids",
    "overt_primary_read_pos", "prodrop_primary_read_pos", "suffix_split",
]].head(12))}
"""
    (PHASE0_DIR / "README.md").write_text(phase0_readme, encoding="utf-8")

    return {
        "verb": verb_df,
        "position": position_df,
        "type_summary": type_summary,
        "by_verb": by_verb,
        "by_person": by_person,
        "by_tense": by_tense,
    }


def attach_case_metadata(df: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Attach one-row-per-case metadata to a score/margin summary table."""
    spec_context_cols = [
        "pair_id", "condition", "condition_label", "cue_type",
        "variant", "actual_person", "prefix_text", "full_text",
    ]
    return df.merge(
        specs[spec_context_cols].drop_duplicates("pair_id"),
        on=["pair_id", "variant", "actual_person"],
        how="left",
    )


def attach_condition_metadata(df: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Attach condition labels to a long-form table."""
    return df.merge(
        specs[["pair_id", "condition", "condition_label", "cue_type"]].drop_duplicates("pair_id"),
        on="pair_id",
        how="left",
        suffixes=("", "_case"),
    )


def final_selection_table(margins: pd.DataFrame) -> pd.DataFrame:
    """Return final-layer margin rows."""
    return margins[margins["layer"] == margins["layer"].max()].copy()


def condition_tokenization_summary(specs: pd.DataFrame) -> pd.DataFrame:
    """Summarize candidate token counts and split types by condition."""
    dedup = specs[[
        "condition", "candidate_person", "candidate_form",
        "candidate_type", "candidate_n_tokens", "candidate_tokens",
    ]].drop_duplicates()
    summary = (
        dedup.groupby(["condition", "candidate_type"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    mean_tokens = (
        dedup.groupby("condition", dropna=False)["candidate_n_tokens"]
        .mean()
        .rename("mean_candidate_n_tokens")
        .reset_index()
    )
    return summary.merge(mean_tokens, on="condition", how="left")


def summarize_sensitivity_by_condition(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sum-vs-mean sensitivity by condition."""
    rows: list[dict] = []
    for condition, group in sensitivity_df.groupby("condition", dropna=False):
        rows.append({
            "condition": condition,
            "n_cases": len(group),
            "mean_final_correct_rate": float(group["mean_selected_is_correct"].astype(bool).mean()),
            "sum_final_correct_rate": float(group["sum_selected_is_correct"].astype(bool).mean()),
            "selection_agreement_rate": float(group["selection_agrees"].astype(bool).mean()),
            "correctness_agreement_rate": float(group["correctness_agrees"].astype(bool).mean()),
            "n_selection_disagreements": int((~group["selection_agrees"].astype(bool)).sum()),
            "n_correctness_disagreements": int((~group["correctness_agrees"].astype(bool)).sum()),
        })
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def build_model_comparison_tables(
    metadata: dict,
    context_specs: pd.DataFrame,
    context_scores: pd.DataFrame,
    context_margins: pd.DataFrame,
    context_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    final_selection: pd.DataFrame,
    control_specs: pd.DataFrame,
    control_scores: pd.DataFrame,
    control_margins: pd.DataFrame,
    control_summary: pd.DataFrame,
    control_condition_summary: pd.DataFrame,
    control_final_selection: pd.DataFrame,
    all_sensitivity: pd.DataFrame,
    all_sensitivity_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the normalized tables used for cross-model comparison."""
    combined_specs = pd.concat([context_specs, control_specs], ignore_index=True)
    combined_scores = pd.concat([context_scores, control_scores], ignore_index=True)
    combined_margins = pd.concat([context_margins, control_margins], ignore_index=True)
    combined_transitions = pd.concat([context_summary, control_summary], ignore_index=True)
    combined_condition_summary = pd.concat(
        [condition_summary, control_condition_summary],
        ignore_index=True,
    )
    combined_final_selection = pd.concat(
        [final_selection, control_final_selection],
        ignore_index=True,
    )
    tokenization_summary = condition_tokenization_summary(combined_specs)

    return {
        "specs": add_model_metadata(combined_specs, metadata),
        "scores": add_model_metadata(combined_scores, metadata),
        "margins": add_model_metadata(combined_margins, metadata),
        "transitions": add_model_metadata(combined_transitions, metadata),
        "condition_summary": add_model_metadata(combined_condition_summary, metadata),
        "final_selection": add_model_metadata(combined_final_selection, metadata),
        "tokenization_summary": add_model_metadata(tokenization_summary, metadata),
        "sensitivity": add_model_metadata(all_sensitivity, metadata),
        "sensitivity_summary": add_model_metadata(all_sensitivity_summary, metadata),
    }


def run_context_control_bundle(model, tokenizer) -> dict[str, pd.DataFrame]:
    """Run Phase 1 context and control experiments for one loaded model."""
    context_specs = build_phase1_context_person_specs(tokenizer)
    context_scores = run_person_form_lens(model, tokenizer, context_specs)
    context_margins = summarize_person_margins(context_scores)
    context_summary = summarize_person_margin_transitions(context_margins)

    context_summary = attach_case_metadata(context_summary, context_specs)
    context_margins = attach_condition_metadata(context_margins, context_specs)
    context_scores = attach_condition_metadata(context_scores, context_specs)

    condition_summary = summarize_context_conditions(context_summary)
    final_selection = final_selection_table(context_margins)
    context_tokenization_summary = condition_tokenization_summary(context_specs)

    control_specs = build_phase1_control_person_specs(tokenizer)
    control_scores = run_person_form_lens(model, tokenizer, control_specs)
    control_margins = summarize_person_margins(control_scores)
    control_summary = summarize_person_margin_transitions(control_margins)
    control_summary = attach_case_metadata(control_summary, control_specs)
    control_margins = attach_condition_metadata(control_margins, control_specs)
    control_scores = attach_condition_metadata(control_scores, control_specs)
    control_condition_summary = summarize_context_conditions(control_summary)
    control_final_selection = final_selection_table(control_margins)
    control_tokenization_summary = condition_tokenization_summary(control_specs)
    control_sensitivity = summarize_sum_vs_mean_sensitivity(control_margins)
    control_sensitivity_summary = summarize_sensitivity_by_condition(control_sensitivity)

    all_margins_for_sensitivity = pd.concat([context_margins, control_margins], ignore_index=True)
    all_sensitivity = summarize_sum_vs_mean_sensitivity(all_margins_for_sensitivity)
    all_sensitivity_summary = summarize_sensitivity_by_condition(all_sensitivity)

    return {
        "context_specs": context_specs,
        "context_scores": context_scores,
        "context_margins": context_margins,
        "context_summary": context_summary,
        "condition_summary": condition_summary,
        "final_selection": final_selection,
        "context_tokenization_summary": context_tokenization_summary,
        "control_specs": control_specs,
        "control_scores": control_scores,
        "control_margins": control_margins,
        "control_summary": control_summary,
        "control_condition_summary": control_condition_summary,
        "control_final_selection": control_final_selection,
        "control_tokenization_summary": control_tokenization_summary,
        "control_sensitivity": control_sensitivity,
        "control_sensitivity_summary": control_sensitivity_summary,
        "all_sensitivity": all_sensitivity,
        "all_sensitivity_summary": all_sensitivity_summary,
    }


def concat_model_tables(model_tables: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Concatenate per-model comparison tables by table name."""
    if not model_tables:
        return {}
    return {
        key: pd.concat([tables[key] for tables in model_tables], ignore_index=True)
        for key in model_tables[0]
    }


def summarize_model_delta_vs_baseline(
    condition_summary: pd.DataFrame,
    baseline_label: str,
) -> pd.DataFrame:
    """Compare each non-baseline model's aggregate metrics against baseline."""
    if condition_summary.empty:
        return pd.DataFrame()
    baseline = condition_summary[
        condition_summary["model_label"] == baseline_label
    ].copy()
    others = condition_summary[
        condition_summary["model_label"] != baseline_label
    ].copy()
    if baseline.empty or others.empty:
        return pd.DataFrame()

    baseline = baseline[[
        "condition", "final_correct_rate", "mean_final_margin",
        "median_final_margin", "median_first_positive_layer",
    ]].rename(columns={
        "final_correct_rate": "baseline_final_correct_rate",
        "mean_final_margin": "baseline_mean_final_margin",
        "median_final_margin": "baseline_median_final_margin",
        "median_first_positive_layer": "baseline_median_first_positive_layer",
    })

    merged = others.merge(baseline, on="condition", how="left")
    merged["delta_final_correct_rate"] = (
        merged["final_correct_rate"] - merged["baseline_final_correct_rate"]
    )
    merged["delta_mean_final_margin"] = (
        merged["mean_final_margin"] - merged["baseline_mean_final_margin"]
    )
    merged["delta_median_final_margin"] = (
        merged["median_final_margin"] - merged["baseline_median_final_margin"]
    )
    merged["delta_median_first_positive_layer"] = (
        merged["median_first_positive_layer"] - merged["baseline_median_first_positive_layer"]
    )
    return merged[[
        "model_label", "model_name", "condition",
        "final_correct_rate", "baseline_final_correct_rate", "delta_final_correct_rate",
        "mean_final_margin", "baseline_mean_final_margin", "delta_mean_final_margin",
        "median_final_margin", "baseline_median_final_margin", "delta_median_final_margin",
        "median_first_positive_layer", "baseline_median_first_positive_layer",
        "delta_median_first_positive_layer",
    ]].sort_values(["model_label", "condition"]).reset_index(drop=True)


def _format_condition_list(values) -> str:
    """Format condition names for a compact markdown note."""
    return ", ".join(f"`{value}`" for value in values)


def build_model_comparison_notes(
    model_tables: dict[str, pd.DataFrame],
    run_status: pd.DataFrame,
    baseline_label: str,
) -> str:
    """Create a compact interpretation summary from model-comparison tables."""
    if not model_tables:
        return "_No completed model-comparison rows._"

    notes: list[str] = []
    tokenization = model_tables["tokenization_summary"]
    profile_cols = ["condition", "candidate_type", "count", "mean_candidate_n_tokens"]
    profiles = {
        label: group[profile_cols].sort_values(profile_cols).reset_index(drop=True)
        for label, group in tokenization.groupby("model_label")
    }
    baseline_profile = profiles.get(baseline_label)
    if baseline_profile is not None and len(profiles) > 1:
        same_profile = all(
            profile.equals(baseline_profile)
            for label, profile in profiles.items()
            if label != baseline_label
        )
        if same_profile:
            notes.append(
                "- Completed models have the same tokenizer split profile for these Phase 1 candidates; observed changes are model-scale effects, not tokenizer-boundary fixes."
            )

    delta = model_tables.get("delta_vs_baseline", pd.DataFrame())
    if not delta.empty:
        for label, group in delta.groupby("model_label"):
            improved = group[group["delta_final_correct_rate"] > 0]
            regressed = group[group["delta_final_correct_rate"] < 0]
            unchanged = group[group["delta_final_correct_rate"] == 0]

            if not improved.empty:
                notes.append(
                    f"- `{label}` improves final correctness in {_format_condition_list(improved['condition'])}."
                )
            if not regressed.empty:
                notes.append(
                    f"- `{label}` regresses final correctness in {_format_condition_list(regressed['condition'])}."
                )
            if not unchanged.empty:
                stronger = unchanged[unchanged["delta_mean_final_margin"] > 0.25]
                weaker = unchanged[unchanged["delta_mean_final_margin"] < -0.25]
                if not stronger.empty:
                    notes.append(
                        f"- `{label}` keeps final correctness unchanged but strengthens mean margin in {_format_condition_list(stronger['condition'])}."
                    )
                if not weaker.empty:
                    notes.append(
                        f"- `{label}` keeps final correctness unchanged but weakens mean margin in {_format_condition_list(weaker['condition'])}."
                    )

    failed = run_status[run_status["status"] == "failed"]
    for _, row in failed.iterrows():
        notes.append(
            f"- `{row['model_label']}` did not complete on this machine: {row['error']}."
        )

    return "\n".join(notes) if notes else "_No material model-scale differences were detected._"


def run_comparison_model(model_name: str, dtype: str, device: str | None) -> dict[str, pd.DataFrame]:
    """Load and run one additional model for the Phase 1 comparison."""
    print(f"Running model comparison for {model_name}")
    model, tokenizer = load_hf_lens_model(model_name=model_name, device=device, dtype=dtype)
    try:
        metadata = model_metadata(model, model_name)
        outputs = run_context_control_bundle(model, tokenizer)
        return build_model_comparison_tables(
            metadata,
            context_specs=outputs["context_specs"],
            context_scores=outputs["context_scores"],
            context_margins=outputs["context_margins"],
            context_summary=outputs["context_summary"],
            condition_summary=outputs["condition_summary"],
            final_selection=outputs["final_selection"],
            control_specs=outputs["control_specs"],
            control_scores=outputs["control_scores"],
            control_margins=outputs["control_margins"],
            control_summary=outputs["control_summary"],
            control_condition_summary=outputs["control_condition_summary"],
            control_final_selection=outputs["control_final_selection"],
            all_sensitivity=outputs["all_sensitivity"],
            all_sensitivity_summary=outputs["all_sensitivity_summary"],
        )
    finally:
        release_model(model)
        model = None


def write_model_comparison_tables(model_tables: dict[str, pd.DataFrame]) -> None:
    """Write cross-model comparison outputs."""
    if not model_tables:
        return
    write_table(model_tables["specs"], PHASE1_DIR / "phase1_model_comparison_person_form_specs.csv")
    write_table(model_tables["scores"], PHASE1_DIR / "phase1_model_comparison_person_form_scores.csv")
    write_table(model_tables["margins"], PHASE1_DIR / "phase1_model_comparison_person_margin_layers.csv")
    write_table(model_tables["transitions"], PHASE1_DIR / "phase1_model_comparison_transition_summary.csv")
    write_table(model_tables["condition_summary"], PHASE1_DIR / "phase1_model_comparison_condition_summary.csv")
    write_table(model_tables["final_selection"], PHASE1_DIR / "phase1_model_comparison_final_selection.csv")
    write_table(model_tables["tokenization_summary"], PHASE1_DIR / "phase1_model_comparison_tokenization_summary.csv")
    write_table(model_tables["sensitivity"], PHASE1_DIR / "phase1_model_comparison_sum_vs_mean_sensitivity.csv")
    write_table(
        model_tables["sensitivity_summary"],
        PHASE1_DIR / "phase1_model_comparison_sum_vs_mean_sensitivity_summary.csv",
    )
    if "delta_vs_baseline" in model_tables:
        write_table(
            model_tables["delta_vs_baseline"],
            PHASE1_DIR / "phase1_model_comparison_delta_vs_baseline.csv",
        )


def run(args: argparse.Namespace) -> None:
    PHASE1_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    phase0_refs = prepare_phase0_reference()

    baseline_model_name = normalize_model_name(args.model_name)
    comparison_model_names = []
    if not args.skip_model_comparison:
        comparison_model_names = normalize_comparison_models(
            args.comparison_models,
            baseline_model_name,
        )

    model, tokenizer = load_lens_model(
        model_name=baseline_model_name,
        device=args.device,
        dtype=args.dtype,
    )
    baseline_metadata = model_metadata(model, baseline_model_name)

    # Keep an archive copy of the existing original-pilot person-margin run.
    original_specs = build_person_form_specs(tokenizer, get_pilot())
    original_scores = run_person_form_lens(model, tokenizer, original_specs)
    original_margins = summarize_person_margins(original_scores)
    original_summary = summarize_person_margin_transitions(original_margins)
    original_summary["condition"] = "original_overt_vs_ambiguous"
    original_margins["condition"] = "original_overt_vs_ambiguous"
    original_scores["condition"] = "original_overt_vs_ambiguous"

    write_table(original_specs, PHASE1_DIR / "phase1_original_person_form_specs.csv")
    write_table(original_scores, PHASE1_DIR / "phase1_original_person_form_scores.csv")
    write_table(original_margins, PHASE1_DIR / "phase1_original_person_margin_layers.csv")
    write_table(original_summary, PHASE1_DIR / "phase1_original_person_margin_transition_summary.csv")

    context_specs = build_phase1_context_person_specs(tokenizer)
    context_scores = run_person_form_lens(model, tokenizer, context_specs)
    context_margins = summarize_person_margins(context_scores)
    context_summary = summarize_person_margin_transitions(context_margins)

    context_summary = attach_case_metadata(context_summary, context_specs)
    context_margins = attach_condition_metadata(context_margins, context_specs)
    context_scores = attach_condition_metadata(context_scores, context_specs)

    condition_summary = summarize_context_conditions(context_summary)
    final_selection = final_selection_table(context_margins)
    context_tokenization_summary = condition_tokenization_summary(context_specs)

    tokenization_cols = [
        "condition", "actual_person", "candidate_person", "candidate_form",
        "candidate_type", "candidate_tokens", "candidate_token_ids",
        "candidate_read_positions",
    ]
    tokenization = context_specs[tokenization_cols].copy()

    write_table(context_specs, PHASE1_DIR / "phase1_context_person_form_specs.csv")
    write_table(context_scores, PHASE1_DIR / "phase1_context_person_form_scores.csv")
    write_table(context_margins, PHASE1_DIR / "phase1_context_person_margin_layers.csv")
    write_table(context_summary, PHASE1_DIR / "phase1_context_person_margin_transition_summary.csv")
    write_table(condition_summary, PHASE1_DIR / "phase1_context_condition_summary.csv")
    write_table(final_selection, PHASE1_DIR / "phase1_context_final_selection.csv")
    write_table(tokenization, PHASE1_DIR / "phase1_context_tokenization.csv")
    write_table(context_tokenization_summary, PHASE1_DIR / "phase1_context_tokenization_summary.csv")

    control_specs = build_phase1_control_person_specs(tokenizer)
    control_scores = run_person_form_lens(model, tokenizer, control_specs)
    control_margins = summarize_person_margins(control_scores)
    control_summary = summarize_person_margin_transitions(control_margins)
    control_summary = attach_case_metadata(control_summary, control_specs)
    control_margins = attach_condition_metadata(control_margins, control_specs)
    control_scores = attach_condition_metadata(control_scores, control_specs)
    control_condition_summary = summarize_context_conditions(control_summary)
    control_final_selection = final_selection_table(control_margins)
    control_tokenization_summary = condition_tokenization_summary(control_specs)
    control_sensitivity = summarize_sum_vs_mean_sensitivity(control_margins)
    control_sensitivity_summary = summarize_sensitivity_by_condition(control_sensitivity)

    all_margins_for_sensitivity = pd.concat([context_margins, control_margins], ignore_index=True)
    all_sensitivity = summarize_sum_vs_mean_sensitivity(all_margins_for_sensitivity)
    all_sensitivity_summary = summarize_sensitivity_by_condition(all_sensitivity)

    write_table(control_specs, PHASE1_DIR / "phase1_control_person_form_specs.csv")
    write_table(control_scores, PHASE1_DIR / "phase1_control_person_form_scores.csv")
    write_table(control_margins, PHASE1_DIR / "phase1_control_person_margin_layers.csv")
    write_table(control_summary, PHASE1_DIR / "phase1_control_person_margin_transition_summary.csv")
    write_table(control_condition_summary, PHASE1_DIR / "phase1_control_condition_summary.csv")
    write_table(control_final_selection, PHASE1_DIR / "phase1_control_final_selection.csv")
    write_table(control_tokenization_summary, PHASE1_DIR / "phase1_control_tokenization_summary.csv")
    write_table(control_sensitivity, PHASE1_DIR / "phase1_control_sum_vs_mean_sensitivity.csv")
    write_table(control_sensitivity_summary, PHASE1_DIR / "phase1_control_sum_vs_mean_sensitivity_summary.csv")
    write_table(all_sensitivity, PHASE1_DIR / "phase1_all_sum_vs_mean_sensitivity.csv")
    write_table(all_sensitivity_summary, PHASE1_DIR / "phase1_all_sum_vs_mean_sensitivity_summary.csv")

    model_comparison_parts: list[dict[str, pd.DataFrame]] = []
    model_comparison_run_status = []
    model_comparison_tables: dict[str, pd.DataFrame] = {}
    if not args.skip_model_comparison:
        release_model(model)
        model = None

        for comparison_model_name in [baseline_model_name, *comparison_model_names]:
            try:
                comparison_tables = run_comparison_model(
                    comparison_model_name,
                    dtype=args.comparison_dtype,
                    device=args.device,
                )
                model_comparison_parts.append(comparison_tables)
                metadata = {
                    key: comparison_tables["condition_summary"].iloc[0][key]
                    for key in baseline_metadata
                }
                model_comparison_run_status.append({
                    **metadata,
                    "status": "completed",
                    "error": "",
                    "dtype": args.comparison_dtype,
                })
            except Exception as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                metadata = {
                    "model_label": model_label(comparison_model_name),
                    "model_name": comparison_model_name,
                    "model_n_layers": None,
                    "model_d_model": None,
                    "model_n_heads": None,
                    "model_n_params": None,
                }
                model_comparison_run_status.append({
                    **metadata,
                    "status": "failed",
                    "error": repr(exc),
                    "dtype": args.comparison_dtype,
                })

        model_comparison_tables = concat_model_tables(model_comparison_parts)
        if model_comparison_tables:
            model_comparison_tables["delta_vs_baseline"] = summarize_model_delta_vs_baseline(
                model_comparison_tables["condition_summary"],
                model_label(baseline_model_name),
            )
            write_model_comparison_tables(model_comparison_tables)
    else:
        model_comparison_run_status.append({
            **baseline_metadata,
            "status": "skipped",
            "error": "",
            "dtype": "",
        })
        release_model(model)
        model = None

    write_table(
        pd.DataFrame(model_comparison_run_status),
        PHASE1_DIR / "phase1_model_comparison_run_status.csv",
    )

    fig_paths: dict[str, list[str]] = {}
    margin_fig = px.line(
        context_margins,
        x="layer",
        y="person_margin_mean",
        color="actual_person",
        facet_col="condition",
        facet_col_wrap=1,
        markers=True,
        title="Phase 1 context experiments: correct-vs-best-wrong person margin",
        labels={
            "person_margin_mean": "person margin (mean sequence log-prob)",
            "actual_person": "person",
        },
    )
    margin_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    margin_fig.update_layout(height=900)
    fig_paths["person_margin_by_condition"] = save_figure(margin_fig, "person_margin_by_condition")

    final_fig = px.bar(
        final_selection,
        x="actual_person",
        y="person_margin_mean",
        color="selected_is_correct_mean",
        facet_col="condition",
        barmode="group",
        title="Final-layer person margin by condition",
        labels={
            "actual_person": "actual person",
            "person_margin_mean": "final person margin",
            "selected_is_correct_mean": "selected correct form",
        },
    )
    final_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    final_fig.update_layout(height=450)
    fig_paths["final_margin_by_condition"] = save_figure(final_fig, "final_margin_by_condition")

    selected_fig = px.scatter(
        final_selection,
        x="actual_person",
        y="selected_person_mean",
        color="condition",
        symbol="selected_is_correct_mean",
        size=final_selection["person_margin_mean"].abs() + 0.1,
        title="Final selected person form by condition",
        labels={
            "actual_person": "actual person",
            "selected_person_mean": "selected person",
        },
    )
    selected_fig.update_layout(height=500)
    fig_paths["final_selected_person"] = save_figure(selected_fig, "final_selected_person")

    control_margin_fig = px.line(
        control_margins,
        x="layer",
        y="person_margin_mean",
        color="actual_person",
        facet_col="condition",
        facet_col_wrap=1,
        markers=True,
        title="Phase 1 controls: correct-vs-best-wrong person margin",
        labels={
            "person_margin_mean": "person margin (mean sequence log-prob)",
            "actual_person": "person",
        },
    )
    control_margin_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    control_margin_fig.update_layout(height=650)
    fig_paths["control_person_margin"] = save_figure(control_margin_fig, "control_person_margin")

    control_final_fig = px.bar(
        control_final_selection,
        x="actual_person",
        y="person_margin_mean",
        color="selected_is_correct_mean",
        facet_col="condition",
        title="Final-layer control margins",
        labels={
            "actual_person": "actual person",
            "person_margin_mean": "final person margin",
            "selected_is_correct_mean": "selected correct form",
        },
    )
    control_final_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    control_final_fig.update_layout(height=450)
    fig_paths["control_final_margin"] = save_figure(control_final_fig, "control_final_margin")

    sensitivity_fig = px.bar(
        all_sensitivity_summary,
        x="condition",
        y=["mean_final_correct_rate", "sum_final_correct_rate"],
        barmode="group",
        title="Sum vs. mean scoring sensitivity by condition",
        labels={"value": "final correct rate", "condition": "condition"},
    )
    sensitivity_fig.update_layout(height=550, xaxis_tickangle=-25)
    fig_paths["sum_vs_mean_sensitivity"] = save_figure(sensitivity_fig, "sum_vs_mean_sensitivity")

    if model_comparison_tables:
        model_condition_fig = px.bar(
            model_comparison_tables["condition_summary"],
            x="condition",
            y="final_correct_rate",
            color="model_label",
            barmode="group",
            title="Model comparison: final correct rate by condition",
            labels={
                "condition": "condition",
                "final_correct_rate": "final correct rate",
                "model_label": "model",
            },
        )
        model_condition_fig.update_layout(height=550, xaxis_tickangle=-25)
        fig_paths["model_comparison_correct_rate"] = save_figure(
            model_condition_fig,
            "model_comparison_correct_rate",
        )

        model_final_fig = px.bar(
            model_comparison_tables["final_selection"],
            x="actual_person",
            y="person_margin_mean",
            color="model_label",
            facet_col="condition",
            facet_col_wrap=1,
            barmode="group",
            title="Model comparison: final-layer person margin",
            labels={
                "actual_person": "actual person",
                "person_margin_mean": "final person margin",
                "model_label": "model",
            },
        )
        model_final_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        model_final_fig.update_layout(height=1100)
        fig_paths["model_comparison_final_margin"] = save_figure(
            model_final_fig,
            "model_comparison_final_margin",
        )

    original_final = original_margins[original_margins["layer"] == original_margins["layer"].max()].copy()
    original_readme = original_final[[
        "variant", "actual_person", "correct_candidate_form", "person_margin_mean",
        "correct_form_rank_mean", "selected_person_mean", "selected_form_mean",
        "selected_is_correct_mean",
    ]]
    context_readme = final_selection[[
        "condition", "actual_person", "correct_candidate_form", "person_margin_mean",
        "correct_form_rank_mean", "selected_person_mean", "selected_form_mean",
        "selected_is_correct_mean",
    ]]
    context_transition_readme = context_summary[[
        "condition", "actual_person", "correct_candidate_form",
        "first_positive_margin_layer", "final_margin", "final_correct_form_rank_mean",
        "final_selected_person_mean", "final_selected_form_mean",
        "final_selected_is_correct_mean",
    ]]
    control_readme = control_final_selection[[
        "condition", "actual_person", "correct_candidate_form", "person_margin_mean",
        "correct_form_rank_mean", "selected_person_mean", "selected_form_mean",
        "selected_is_correct_mean",
    ]]
    control_transition_readme = control_summary[[
        "condition", "actual_person", "correct_candidate_form",
        "first_positive_margin_layer", "final_margin", "final_correct_form_rank_mean",
        "final_selected_person_mean", "final_selected_form_mean",
        "final_selected_is_correct_mean",
    ]]
    control_sensitivity_readme = all_sensitivity_summary.copy()
    model_status_readme = pd.DataFrame(model_comparison_run_status)
    model_comparison_section = ""
    if model_comparison_tables:
        model_condition_readme = model_comparison_tables["condition_summary"][[
            "model_label", "model_n_layers", "condition", "n_cases",
            "final_correct_rate", "ever_positive_rate", "mean_final_margin",
            "median_final_margin", "median_first_positive_layer",
        ]]
        model_delta_cols = [
            "model_label", "condition", "final_correct_rate",
            "baseline_final_correct_rate", "delta_final_correct_rate",
            "mean_final_margin", "baseline_mean_final_margin", "delta_mean_final_margin",
        ]
        model_delta_readme = (
            model_comparison_tables["delta_vs_baseline"][model_delta_cols]
            if not model_comparison_tables["delta_vs_baseline"].empty
            else pd.DataFrame(columns=model_delta_cols)
        )
        model_tokenization_readme = model_comparison_tables["tokenization_summary"][[
            "model_label", "condition", "candidate_type", "count",
            "mean_candidate_n_tokens",
        ]]
        model_sensitivity_readme = model_comparison_tables["sensitivity_summary"][[
            "model_label", "condition", "mean_final_correct_rate",
            "sum_final_correct_rate", "selection_agreement_rate",
            "correctness_agreement_rate",
        ]]
        model_notes = build_model_comparison_notes(
            model_comparison_tables,
            model_status_readme,
            model_label(baseline_model_name),
        )
        model_comparison_section = f"""
## Model Comparison

This section reruns the same Phase 1 context and control candidate-form
experiments across the baseline model and configured upper-model checkpoints.
It is meant to test whether the Phase 1 person-margin pattern is stable under
model scale, or whether conclusions are mostly tokenizer/scoring artifacts.

### Model Comparison Run Status

{markdown_table(model_status_readme)}

### Aggregate Results by Model

{markdown_table(model_condition_readme.round(3))}

### Delta vs. Baseline

Positive deltas mean the upper model improved over the baseline for that
condition; negative deltas mean it regressed.

{markdown_table(model_delta_readme.round(3))}

### Model Comparison Notes

{model_notes}

### Tokenization by Model

{markdown_table(model_tokenization_readme.round(3))}

### Sum vs. Mean Sensitivity by Model

{markdown_table(model_sensitivity_readme.round(3))}
"""
    else:
        model_comparison_section = f"""
## Model Comparison

Model comparison was skipped for this run.

### Model Comparison Run Status

{markdown_table(model_status_readme)}
"""

    phase0_section = ""
    if phase0_refs:
        phase0_section = f"""
## Phase 0 Reference Inputs

The Phase 0 tokenizer outputs are copied to `../phase0/` and are part of this
report package. They define which comparisons are clean single-token cases and
which ones are sensitive to suffix/token continuation.

### Phase 0 Tokenization Summary

{markdown_table(phase0_refs["type_summary"].assign(share=phase0_refs["type_summary"]["share"].round(3)))}

### Phase 0 Pilot Position Reference

{markdown_table(phase0_refs["position"][[
    "person", "target_form", "compare_mode", "target_tokens", "target_token_ids",
    "overt_primary_read_pos", "prodrop_primary_read_pos", "suffix_split",
]])}
"""

    readme = f"""# Phase 1 Context Experiments

This folder contains the Phase 1 follow-up experiments that separate target-token
rank from subject/person-conditioned form selection. Existing root-level
`results/phase1_*.csv` files are left unchanged.

{phase0_section}

## Analysis Flow

1. Keep Phase 0 tokenizer and position references visible.
2. Archive the original pilot person-margin run.
3. Run three natural context experiments.
4. Run three final controls:
   - tokenization-matched verb control
   - mixed-token verb control
   - sum-vs-mean scoring sensitivity
5. Rerun the same context and control margins on configured upper-model
   checkpoints for model-scale comparison.

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
- `phase1_model_comparison_*.csv`: cross-model context/control outputs,
  tokenization summaries, and deltas against the baseline model.

## Figures

"""
    for name, paths in fig_paths.items():
        readme += f"- `{name}`: " + ", ".join(f"`{path}`" for path in paths) + "\n"

    readme += f"""
## Aggregate Context Results

{markdown_table(condition_summary.round(3))}

## Original Pilot Final Layer

{markdown_table(original_readme.round(3))}

## New Context Final Layer

{markdown_table(context_readme.round(3))}

## New Context Transition Summary

{markdown_table(context_transition_readme.round(3))}

## Final Control Results

### Control Aggregate Results

{markdown_table(control_condition_summary.round(3))}

### Control Final Layer

{markdown_table(control_readme.round(3))}

### Control Transition Summary

{markdown_table(control_transition_readme.round(3))}

### Tokenization-Matched vs. Mixed-Token Summary

{markdown_table(control_tokenization_summary.round(3))}

### Sum vs. Mean Sensitivity

{markdown_table(control_sensitivity_readme.round(3))}

{model_comparison_section}

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
- Model-scale differences matter only when they change the condition-level
  correctness or margin pattern while leaving the tokenizer/scoring controls
  interpretable. If the tokenizer split profile stays the same, improved
  margins are model-capacity evidence, not a tokenizer fix.
"""

    (PHASE1_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote Phase 1 context outputs to {PHASE1_DIR}")
    print(condition_summary.to_string(index=False))


if __name__ == "__main__":
    run(parse_args())
