"""
Logit-lens helpers for ProDropLens.

Phase 1 asks where the model begins to rank the correct person-bearing
token highly. The functions here keep TransformerLens execution in one place
while preserving the Turkish GPT-2 tokenizer IDs from Hugging Face.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .dataset import Person, VERB_FORMS
from .utils import analyze_sentence, classify_verb_tokenization, compare_tokenizations


MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2"
GPT2_TL_BRIDGE_BY_SHAPE = {
    (12, 768, 12): "gpt2",
    (24, 1024, 16): "gpt2-medium",
    (36, 1280, 20): "gpt2-large",
    (48, 1600, 25): "gpt2-xl",
}


def get_default_device() -> str:
    """Return the preferred local device for model execution."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_tl_gpt2_bridge_name(model_name: str = MODEL_NAME) -> str:
    """
    Return the TransformerLens GPT-2 bridge matching a HF GPT-2 checkpoint.

    The Turkish GPT-2 family uses the same architecture shapes as GPT-2 small,
    medium, and large, but TransformerLens still needs the corresponding
    official bridge name when loading larger checkpoints.
    """
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type != "gpt2":
        raise ValueError(f"Only GPT-2 checkpoints are supported, got {config.model_type!r}")

    shape = (int(config.n_layer), int(config.n_embd), int(config.n_head))
    try:
        return GPT2_TL_BRIDGE_BY_SHAPE[shape]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported GPT-2 architecture shape {shape} for {model_name!r}"
        ) from exc


def _torch_dtype_from_name(dtype: str) -> torch.dtype:
    """Map a string dtype argument to a torch dtype."""
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}") from exc


def load_lens_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    dtype: str = "float32",
) -> tuple[HookedTransformer, object]:
    """
    Load Turkish GPT-2 as a TransformerLens model plus its HF tokenizer.

    TransformerLens does not list ``ytu-ce-cosmos/turkish-gpt2`` as an
    official model name, but the checkpoint uses GPT-2 architecture. We load
    the HF weights and pass them through the official GPT-2 bridge. Downstream
    code must tokenize with the returned HF tokenizer, not ``model.to_tokens``.
    """
    device = device or get_default_device()
    bridge_name = infer_tl_gpt2_bridge_name(model_name)
    torch_dtype = _torch_dtype_from_name(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).eval()

    model = HookedTransformer.from_pretrained(
        bridge_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        default_prepend_bos=False,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def load_hf_lens_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    dtype: str = "float32",
) -> tuple[AutoModelForCausalLM, object]:
    """
    Load a HF causal LM for hidden-state logit-lens scoring.

    This is lighter than converting larger GPT-2 checkpoints through
    TransformerLens and is sufficient for layer-wise sequence scoring.
    """
    device = device or get_default_device()
    torch_dtype = _torch_dtype_from_name(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    ).to(device).eval()
    return hf_model, tokenizer


def encode_with_hf_tokenizer(tokenizer, text: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Encode text with the HF tokenizer.

    This avoids accidental fallback to the official English GPT-2 tokenizer in
    TransformerLens helper methods.
    """
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids


def decode_token(tokenizer, token_id: Optional[int]) -> Optional[str]:
    """Decode one token ID with whitespace preserved."""
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False).strip()


def token_rank(logits_1d: torch.Tensor, token_id: int) -> int:
    """Return a token's 1-indexed rank without sorting the whole vocabulary."""
    target_logit = logits_1d[int(token_id)]
    return int((logits_1d > target_logit).sum().item()) + 1


def lens_logits_from_residual(model: HookedTransformer, residual: torch.Tensor) -> torch.Tensor:
    """
    Project a residual stream vector through final layer norm and unembed.

    For GPT-2 style models, final logits are computed from ``ln_final`` over
    the residual stream followed by unembedding. We reuse that path at each
    intermediate layer.
    """
    normalized = model.ln_final(residual)
    return model.unembed(normalized)


def hf_lens_logits_from_hidden(model, hidden: torch.Tensor, apply_final_ln: bool) -> torch.Tensor:
    """Project a HF GPT-2 hidden state through final LN and LM head."""
    if apply_final_ln:
        hidden = model.transformer.ln_f(hidden)
    return model.lm_head(hidden)


def choose_compare_mode(
    target_form: str,
    contrast_form: str,
    target_info: dict,
    contrast_info: dict,
) -> str:
    """Return the Phase 1 comparison strategy implied by tokenization."""
    if target_form == contrast_form:
        return "self_contrast_baseline"
    if target_info["n_tokens"] == 1 and contrast_info["n_tokens"] == 1:
        return "single_token_logit_diff"
    if target_info["type"] == "B_suffix_split":
        return "suffix_probe_plus_sequence_logprob"
    return "sequence_logprob_case_by_case"


def _fmt_list(values: Iterable) -> str:
    values = list(values)
    return " | ".join(str(v) for v in values) if values else "-"


def _form_encoding(tokenizer, form: str) -> dict:
    info = classify_verb_tokenization(tokenizer, form)
    return {
        "type": info["type"],
        "n_tokens": info["n_tokens"],
        "token_ids": info["token_ids"],
        "tokens": info["tokens"],
        "suffix_token": info["suffix_token"] or "-",
    }


def _token_at_span_pos(sentence_info: dict, pos: Optional[int]) -> tuple[Optional[int], Optional[str]]:
    span = sentence_info["verb_span"]
    if span is None or pos is None:
        return None, None
    offset = pos - span.start
    if offset < 0 or offset >= span.n_tokens:
        return None, None
    return span.token_ids[offset], span.token_strs[offset].strip()


def build_logit_lens_specs(tokenizer, pairs: Sequence) -> pd.DataFrame:
    """
    Build one analysis spec per overt/pro-drop sentence.

    The target token is the token carrying the strongest person signal under
    the Phase 0 rules: the suffix token when split, otherwise the final token in
    the verb span.
    """
    rows: list[dict] = []

    for pair in pairs:
        comparison = compare_tokenizations(
            tokenizer,
            overt_text=pair.overt_text,
            prodrop_text=pair.prodrop_text,
            verb_form=pair.target_form,
            subject=pair.subject_token,
        )
        target_info = _form_encoding(tokenizer, pair.target_form)
        contrast_info = _form_encoding(tokenizer, pair.contrast_form)
        compare_mode = choose_compare_mode(
            pair.target_form,
            pair.contrast_form,
            target_info,
            contrast_info,
        )

        contrast_token_id = None
        if compare_mode == "single_token_logit_diff":
            contrast_token_id = contrast_info["token_ids"][0]

        for variant, sentence_info in [
            ("overt", comparison["overt"]),
            ("prodrop", comparison["prodrop"]),
        ]:
            target_token_id, target_token = _token_at_span_pos(
                sentence_info,
                sentence_info["primary_probe_pos"],
            )

            rows.append({
                "pair_id": pair.id,
                "variant": variant,
                "person": pair.person.value,
                "subject": pair.subject_token if variant == "overt" else "",
                "text": pair.overt_text if variant == "overt" else pair.prodrop_text,
                "target_form": pair.target_form,
                "contrast_form": pair.contrast_form,
                "compare_mode": compare_mode,
                "target_type": target_info["type"],
                "target_tokens": _fmt_list(target_info["tokens"]),
                "target_token_ids": _fmt_list(target_info["token_ids"]),
                "target_token_id": target_token_id,
                "target_token": target_token,
                "contrast_token_id": contrast_token_id,
                "contrast_token": decode_token(tokenizer, contrast_token_id),
                "verb_start": sentence_info["verb_pos"],
                "verb_end": sentence_info["verb_end"],
                "primary_probe_pos": sentence_info["primary_probe_pos"],
                "read_pos": sentence_info["primary_read_pos"],
                "first_token_read_pos": sentence_info["first_token_read_pos"],
                "verb_token_read_positions": _fmt_list(sentence_info["verb_token_read_positions"]),
                "suffix_split": sentence_info["suffix_split"],
                "suffix_pos": sentence_info["suffix_pos"],
                "n_tokens": sentence_info["n_tokens"],
                "sentence_tokens": _fmt_list(sentence_info["token_strs"]),
                "sentence_token_ids": _fmt_list(sentence_info["token_ids"]),
            })

    return pd.DataFrame(rows)


def run_logit_lens_for_spec(
    model: HookedTransformer,
    tokenizer,
    spec: dict,
    layers: Optional[Sequence[int]] = None,
    top_k: int = 1,
) -> list[dict]:
    """Run logit lens for one sentence/spec row."""
    layers = list(layers) if layers is not None else list(range(model.cfg.n_layers))
    device = str(next(model.parameters()).device)
    input_ids = encode_with_hf_tokenizer(tokenizer, spec["text"], device=device)
    read_pos = int(spec["read_pos"])

    if read_pos < 0 or read_pos >= input_ids.shape[1]:
        raise ValueError(f"Invalid read_pos={read_pos} for text: {spec['text']!r}")

    names_filter = lambda name: name.endswith("hook_resid_post")
    rows: list[dict] = []

    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=names_filter,
            remove_batch_dim=False,
        )

        for layer in layers:
            residual = cache[f"blocks.{layer}.hook_resid_post"][:, read_pos, :]
            logits = lens_logits_from_residual(model, residual)[0]
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)

            target_id = int(spec["target_token_id"])
            target_logit = float(logits[target_id].item())
            target_prob = float(probs[target_id].item())
            target_log_prob = float(log_probs[target_id].item())
            target_rank = token_rank(logits, target_id)

            row = {
                "layer": layer,
                "target_logit": target_logit,
                "target_prob": target_prob,
                "target_log_prob": target_log_prob,
                "target_rank": target_rank,
            }

            contrast_id = spec.get("contrast_token_id")
            if contrast_id is not None and not pd.isna(contrast_id):
                contrast_id = int(contrast_id)
                row.update({
                    "contrast_logit": float(logits[contrast_id].item()),
                    "contrast_prob": float(probs[contrast_id].item()),
                    "contrast_log_prob": float(log_probs[contrast_id].item()),
                    "contrast_rank": token_rank(logits, contrast_id),
                    "logit_diff": float((logits[target_id] - logits[contrast_id]).item()),
                })
            else:
                row.update({
                    "contrast_logit": None,
                    "contrast_prob": None,
                    "contrast_log_prob": None,
                    "contrast_rank": None,
                    "logit_diff": None,
                })

            if top_k > 0:
                top_vals, top_ids = torch.topk(logits, k=top_k)
                row["top_tokens"] = _fmt_list(
                    decode_token(tokenizer, token_id.item()) for token_id in top_ids
                )
                row["top_token_ids"] = _fmt_list(token_id.item() for token_id in top_ids)
                row["top_logits"] = _fmt_list(f"{val.item():.4f}" for val in top_vals)

            rows.append(row)

    return rows


def run_logit_lens(
    model: HookedTransformer,
    tokenizer,
    specs_df: pd.DataFrame,
    layers: Optional[Sequence[int]] = None,
    top_k: int = 1,
) -> pd.DataFrame:
    """Run logit lens for every spec row and return a long-form table."""
    all_rows: list[dict] = []

    for _, spec_row in specs_df.iterrows():
        spec = spec_row.to_dict()
        lens_rows = run_logit_lens_for_spec(
            model,
            tokenizer,
            spec,
            layers=layers,
            top_k=top_k,
        )
        for lens_row in lens_rows:
            all_rows.append({**spec, **lens_row})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(all_rows)


def summarize_rank_transitions(
    lens_df: pd.DataFrame,
    thresholds: Sequence[int] = (100, 50, 10, 5, 1),
) -> pd.DataFrame:
    """
    Summarize when target-token rank improves for each sentence.

    ``critical_from_layer`` -> ``critical_to_layer`` marks the biggest drop in
    log rank between adjacent layers.
    """
    summary_rows: list[dict] = []
    group_cols = ["pair_id", "variant", "person", "target_form", "target_token", "compare_mode"]

    for group_key, group in lens_df.sort_values("layer").groupby(group_cols, dropna=False):
        group = group.sort_values("layer")
        ranks = group["target_rank"].astype(float).tolist()
        layers = group["layer"].astype(int).tolist()
        log_ranks = torch.log(torch.tensor(ranks) + 1.0).tolist()
        drops = [
            log_ranks[i - 1] - log_ranks[i]
            for i in range(1, len(log_ranks))
        ]

        if drops:
            best_i = max(range(len(drops)), key=lambda i: drops[i])
            critical_from = layers[best_i]
            critical_to = layers[best_i + 1]
            critical_drop = drops[best_i]
        else:
            critical_from = None
            critical_to = None
            critical_drop = None

        row = dict(zip(group_cols, group_key))
        row.update({
            "initial_rank": int(ranks[0]),
            "final_rank": int(ranks[-1]),
            "best_rank": int(min(ranks)),
            "best_rank_layer": int(group.loc[group["target_rank"].idxmin(), "layer"]),
            "critical_from_layer": critical_from,
            "critical_to_layer": critical_to,
            "critical_log_rank_drop": critical_drop,
        })

        for threshold in thresholds:
            hit = group[group["target_rank"] <= threshold]
            row[f"first_layer_rank_le_{threshold}"] = (
                int(hit.iloc[0]["layer"]) if not hit.empty else None
            )

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def compare_overt_prodrop(lens_df: pd.DataFrame) -> pd.DataFrame:
    """Create a layer-by-layer overt vs. pro-drop comparison table."""
    metric_cols = ["target_rank", "target_prob", "target_log_prob", "target_logit", "logit_diff"]
    id_cols = ["pair_id", "person", "target_form", "target_token", "compare_mode", "layer"]
    rows: list[dict] = []

    for keys, group in lens_df.groupby(["pair_id", "person", "target_form", "target_token", "compare_mode", "layer"], dropna=False):
        by_variant = {row["variant"]: row for _, row in group.iterrows()}
        if "overt" not in by_variant or "prodrop" not in by_variant:
            continue

        row = dict(zip(id_cols, keys))
        for metric in metric_cols:
            overt_val = by_variant["overt"].get(metric)
            prodrop_val = by_variant["prodrop"].get(metric)
            row[f"overt_{metric}"] = overt_val
            row[f"prodrop_{metric}"] = prodrop_val
            if pd.notna(overt_val) and pd.notna(prodrop_val):
                row[f"{metric}_diff_prodrop_minus_overt"] = prodrop_val - overt_val
            else:
                row[f"{metric}_diff_prodrop_minus_overt"] = None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["pair_id", "layer"]).reset_index(drop=True)


def _prefix_before_form(text: str, form: str) -> str:
    """Return the sentence prefix before a target form."""
    idx = text.rfind(form)
    if idx < 0:
        raise ValueError(f"Could not find form {form!r} in text {text!r}")
    return text[:idx]


def build_person_form_specs(tokenizer, pairs: Sequence) -> pd.DataFrame:
    """
    Build candidate-form specs for subject-conditioned person selection.

    Each context is scored against all six person forms from the same
    verb/tense family. This is stricter than tracking one target rank: it asks
    whether the correct person form separates from the competing person forms.
    """
    rows: list[dict] = []

    for pair in pairs:
        candidate_forms = VERB_FORMS[pair.verb][pair.tense]

        for variant, full_text in [
            ("overt", pair.overt_text),
            ("prodrop", pair.prodrop_text),
        ]:
            prefix = _prefix_before_form(full_text, pair.target_form)
            subject = pair.subject_token if variant == "overt" else ""

            for candidate_person in Person:
                candidate_form = candidate_forms[candidate_person]
                candidate_info = _form_encoding(tokenizer, candidate_form)
                candidate_text = f"{prefix}{candidate_form}"
                sentence_info = analyze_sentence(
                    tokenizer,
                    candidate_text,
                    subject if variant == "overt" else None,
                    candidate_form,
                )

                token_positions = []
                read_positions = []
                if sentence_info["verb_pos"] is not None and sentence_info["verb_end"] is not None:
                    token_positions = list(range(sentence_info["verb_pos"], sentence_info["verb_end"] + 1))
                    read_positions = [pos - 1 if pos > 0 else None for pos in token_positions]

                rows.append({
                    "pair_id": pair.id,
                    "variant": variant,
                    "actual_person": pair.person.value,
                    "candidate_person": candidate_person.value,
                    "is_correct": candidate_person == pair.person,
                    "subject": subject,
                    "prefix_text": prefix,
                    "target_form": pair.target_form,
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
                })

    return pd.DataFrame(rows)


def run_person_form_lens_for_spec_hf(
    model,
    tokenizer,
    spec: dict,
    layers: Optional[Sequence[int]] = None,
) -> list[dict]:
    """
    HF implementation of layer-wise candidate-form scoring.

    HF GPT-2 exposes hidden states for each block boundary. Intermediate layers
    are projected through the final layer norm and LM head; the final hidden
    state is already normalized by GPT-2's ``ln_f``.
    """
    n_layers = int(model.config.n_layer)
    layers = list(layers) if layers is not None else list(range(n_layers))
    device = str(next(model.parameters()).device)
    input_ids = encode_with_hf_tokenizer(tokenizer, spec["candidate_text"], device=device)
    sentence_info = analyze_sentence(tokenizer, spec["candidate_text"], None, spec["candidate_form"])

    if sentence_info["verb_pos"] is None or sentence_info["verb_end"] is None:
        raise ValueError(f"Could not locate candidate form in {spec['candidate_text']!r}")

    token_positions = list(range(sentence_info["verb_pos"], sentence_info["verb_end"] + 1))
    if any(pos <= 0 for pos in token_positions):
        raise ValueError(f"Cannot score candidate at position 0: {spec['candidate_text']!r}")

    rows: list[dict] = []

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states

        for layer in layers:
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"Invalid layer={layer} for n_layers={n_layers}")
            if layer == n_layers - 1:
                layer_hidden = hidden_states[-1]
                apply_final_ln = False
            else:
                layer_hidden = hidden_states[layer + 1]
                apply_final_ln = True

            token_log_probs = []
            token_logits = []
            token_vocab_ranks = []

            for pos in token_positions:
                read_pos = pos - 1
                token_id = int(input_ids[0, pos].item())
                hidden = layer_hidden[:, read_pos, :]
                logits = hf_lens_logits_from_hidden(model, hidden, apply_final_ln)[0]
                log_probs = torch.log_softmax(logits, dim=-1)

                token_log_probs.append(float(log_probs[token_id].item()))
                token_logits.append(float(logits[token_id].item()))
                token_vocab_ranks.append(token_rank(logits, token_id))

            first_pos = token_positions[0]
            first_read_pos = first_pos - 1
            first_token_id = int(input_ids[0, first_pos].item())
            first_hidden = layer_hidden[:, first_read_pos, :]
            first_logits = hf_lens_logits_from_hidden(model, first_hidden, apply_final_ln)[0]

            rows.append({
                "layer": layer,
                "sequence_log_prob_sum": sum(token_log_probs),
                "sequence_log_prob_mean": sum(token_log_probs) / len(token_log_probs),
                "sequence_logit_sum": sum(token_logits),
                "sequence_logit_mean": sum(token_logits) / len(token_logits),
                "token_log_probs": _fmt_list(f"{v:.6f}" for v in token_log_probs),
                "token_logits": _fmt_list(f"{v:.6f}" for v in token_logits),
                "token_vocab_ranks": _fmt_list(token_vocab_ranks),
                "first_token_id": first_token_id,
                "first_token": decode_token(tokenizer, first_token_id),
                "first_token_vocab_rank": token_rank(first_logits, first_token_id),
            })

    return rows


def run_person_form_lens_for_spec(
    model: HookedTransformer,
    tokenizer,
    spec: dict,
    layers: Optional[Sequence[int]] = None,
) -> list[dict]:
    """
    Score one candidate form at every layer using teacher-forced token scores.

    Multi-token forms are scored by summing and averaging the layer-wise
    log-probabilities of each token in the candidate form.
    """
    if not isinstance(model, HookedTransformer):
        return run_person_form_lens_for_spec_hf(
            model,
            tokenizer,
            spec,
            layers=layers,
        )

    layers = list(layers) if layers is not None else list(range(model.cfg.n_layers))
    device = str(next(model.parameters()).device)
    input_ids = encode_with_hf_tokenizer(tokenizer, spec["candidate_text"], device=device)
    sentence_info = analyze_sentence(tokenizer, spec["candidate_text"], None, spec["candidate_form"])

    if sentence_info["verb_pos"] is None or sentence_info["verb_end"] is None:
        raise ValueError(f"Could not locate candidate form in {spec['candidate_text']!r}")

    token_positions = list(range(sentence_info["verb_pos"], sentence_info["verb_end"] + 1))
    if any(pos <= 0 for pos in token_positions):
        raise ValueError(f"Cannot score candidate at position 0: {spec['candidate_text']!r}")

    names_filter = lambda name: name.endswith("hook_resid_post")
    rows: list[dict] = []

    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=names_filter,
            remove_batch_dim=False,
        )

        for layer in layers:
            token_log_probs = []
            token_logits = []
            token_vocab_ranks = []

            for pos in token_positions:
                read_pos = pos - 1
                token_id = int(input_ids[0, pos].item())
                residual = cache[f"blocks.{layer}.hook_resid_post"][:, read_pos, :]
                logits = lens_logits_from_residual(model, residual)[0]
                log_probs = torch.log_softmax(logits, dim=-1)

                token_log_probs.append(float(log_probs[token_id].item()))
                token_logits.append(float(logits[token_id].item()))
                token_vocab_ranks.append(token_rank(logits, token_id))

            first_pos = token_positions[0]
            first_read_pos = first_pos - 1
            first_token_id = int(input_ids[0, first_pos].item())
            first_residual = cache[f"blocks.{layer}.hook_resid_post"][:, first_read_pos, :]
            first_logits = lens_logits_from_residual(model, first_residual)[0]

            rows.append({
                "layer": layer,
                "sequence_log_prob_sum": sum(token_log_probs),
                "sequence_log_prob_mean": sum(token_log_probs) / len(token_log_probs),
                "sequence_logit_sum": sum(token_logits),
                "sequence_logit_mean": sum(token_logits) / len(token_logits),
                "token_log_probs": _fmt_list(f"{v:.6f}" for v in token_log_probs),
                "token_logits": _fmt_list(f"{v:.6f}" for v in token_logits),
                "token_vocab_ranks": _fmt_list(token_vocab_ranks),
                "first_token_id": first_token_id,
                "first_token": decode_token(tokenizer, first_token_id),
                "first_token_vocab_rank": token_rank(first_logits, first_token_id),
            })

    return rows


def run_person_form_lens(
    model: HookedTransformer,
    tokenizer,
    person_specs_df: pd.DataFrame,
    layers: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Run layer-wise candidate-form scoring for every person-form spec."""
    all_rows: list[dict] = []

    for _, spec_row in person_specs_df.iterrows():
        spec = spec_row.to_dict()
        score_rows = run_person_form_lens_for_spec(
            model,
            tokenizer,
            spec,
            layers=layers,
        )
        for score_row in score_rows:
            all_rows.append({**spec, **score_row})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores_df = pd.DataFrame(all_rows)
    return add_candidate_form_ranks(scores_df)


def add_candidate_form_ranks(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add within-context candidate ranks for sum and mean sequence scores."""
    scores_df = scores_df.copy()
    group_cols = ["pair_id", "variant", "actual_person", "layer"]

    for score_col, rank_col in [
        ("sequence_log_prob_sum", "candidate_rank_sum"),
        ("sequence_log_prob_mean", "candidate_rank_mean"),
    ]:
        scores_df[rank_col] = (
            scores_df.groupby(group_cols, dropna=False)[score_col]
            .rank(method="min", ascending=False)
            .astype(int)
        )

    return scores_df


def summarize_person_margins(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize correct-vs-other person separation at each layer.

    ``person_margin_*`` is positive when the correct person form scores above
    every competing person form in that context.
    """
    rows: list[dict] = []
    group_cols = ["pair_id", "variant", "actual_person", "subject", "prefix_text", "target_form", "layer"]

    for group_key, group in scores_df.groupby(group_cols, dropna=False):
        correct = group[group["is_correct"].astype(bool)]
        wrong = group[~group["is_correct"].astype(bool)]
        if correct.empty or wrong.empty:
            continue

        correct_row = correct.iloc[0]
        row = dict(zip(group_cols, group_key))
        row.update({
            "correct_candidate_form": correct_row["candidate_form"],
            "correct_candidate_type": correct_row["candidate_type"],
            "correct_candidate_tokens": correct_row["candidate_tokens"],
            "correct_candidate_token_ids": correct_row["candidate_token_ids"],
        })

        for score_col, rank_col, suffix in [
            ("sequence_log_prob_sum", "candidate_rank_sum", "sum"),
            ("sequence_log_prob_mean", "candidate_rank_mean", "mean"),
        ]:
            best_wrong_idx = wrong[score_col].idxmax()
            best_wrong = wrong.loc[best_wrong_idx]
            selected_idx = group[score_col].idxmax()
            selected = group.loc[selected_idx]

            row.update({
                f"correct_score_{suffix}": float(correct_row[score_col]),
                f"best_wrong_score_{suffix}": float(best_wrong[score_col]),
                f"person_margin_{suffix}": float(correct_row[score_col] - best_wrong[score_col]),
                f"correct_form_rank_{suffix}": int(correct_row[rank_col]),
                f"best_wrong_person_{suffix}": best_wrong["candidate_person"],
                f"best_wrong_form_{suffix}": best_wrong["candidate_form"],
                f"selected_person_{suffix}": selected["candidate_person"],
                f"selected_form_{suffix}": selected["candidate_form"],
                f"selected_is_correct_{suffix}": bool(selected["is_correct"]),
            })

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["pair_id", "variant", "layer"]).reset_index(drop=True)


def summarize_person_margin_transitions(
    margin_df: pd.DataFrame,
    margin_col: str = "person_margin_mean",
) -> pd.DataFrame:
    """
    Summarize when correct-vs-other person margin becomes strongest.

    Positive margin means the correct person form beats all five alternatives.
    """
    rows: list[dict] = []
    group_cols = ["pair_id", "variant", "actual_person", "target_form", "correct_candidate_form"]

    for group_key, group in margin_df.sort_values("layer").groupby(group_cols, dropna=False):
        group = group.sort_values("layer")
        layers = group["layer"].astype(int).tolist()
        margins = group[margin_col].astype(float).tolist()
        increases = [
            margins[i] - margins[i - 1]
            for i in range(1, len(margins))
        ]

        if increases:
            best_i = max(range(len(increases)), key=lambda i: increases[i])
            critical_from = layers[best_i]
            critical_to = layers[best_i + 1]
            critical_increase = increases[best_i]
        else:
            critical_from = None
            critical_to = None
            critical_increase = None

        positive = group[group[margin_col] > 0]
        best_idx = group[margin_col].idxmax()
        best = group.loc[best_idx]
        final = group.iloc[-1]

        row = dict(zip(group_cols, group_key))
        row.update({
            "margin_metric": margin_col,
            "initial_margin": margins[0],
            "final_margin": margins[-1],
            "best_margin": float(best[margin_col]),
            "best_margin_layer": int(best["layer"]),
            "first_positive_margin_layer": (
                int(positive.iloc[0]["layer"]) if not positive.empty else None
            ),
            "critical_from_layer": critical_from,
            "critical_to_layer": critical_to,
            "critical_margin_increase": critical_increase,
            "final_correct_form_rank_mean": int(final["correct_form_rank_mean"]),
            "final_selected_person_mean": final["selected_person_mean"],
            "final_selected_form_mean": final["selected_form_mean"],
            "final_selected_is_correct_mean": bool(final["selected_is_correct_mean"]),
        })
        rows.append(row)

    return pd.DataFrame(rows)
