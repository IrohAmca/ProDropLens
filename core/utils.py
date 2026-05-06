"""
Tokenizer and model helper functions for ProDropLens.

Phase 0 tokenizer-discovery utilities live here. Notebooks import these
functions instead of duplicating computation logic.

KEY INSIGHT FROM PHASE 0:
  turkish-gpt2 tokenizer splits verb forms into multiple tokens.
  For example:
    "gidiyorum"    -> [" gid", "iyorum"]       (2 tokens, suffix merged)
    "gidiyorsun"   -> [" gidiyor", "sun"]       (2 tokens, suffix separate)
    "gidiyorsunuz" -> [" gidiyor", "sunuz"]     (2 tokens, suffix separate)

  This creates two analysis strategies:
    TYPE A (suffix merged):   person info is inside the verb token
    TYPE B (suffix separate): person suffix is its own token -> easier to trace
"""

from dataclasses import dataclass
from typing import Optional

import torch


# ── Type aliases ──────────────────────────────────────────────────────────────

TokenId  = int
TokenStr = str
Position = int


# ── Verb span: multi-token verb representation ───────────────────────────────

@dataclass
class VerbSpan:
    """
    A verb form's location within a tokenized sentence.

    Handles both single-token and multi-token verb forms.

    Attributes:
        start:        first token index of the verb
        end:          last token index of the verb (inclusive)
        token_ids:    list of token IDs making up the verb
        token_strs:   list of token strings making up the verb
        n_tokens:     how many tokens the verb spans
        suffix_split: True if the person suffix is a separate token (TYPE B)
        suffix_pos:   position of the suffix token (only if suffix_split)
        stem_pos:     position of the verb stem token
    """
    start: int
    end: int
    token_ids: list[int]
    token_strs: list[str]
    n_tokens: int
    suffix_split: bool
    suffix_pos: Optional[int]
    stem_pos: int

    @property
    def last_pos(self) -> int:
        """Position of the last verb token (where logits should be read)."""
        return self.end


# ── Tokenizer analysis functions ─────────────────────────────────────────────

def tokenize(tokenizer, text: str) -> dict:
    """
    Tokenize one sentence and return an analysis-ready dictionary.

    Returns:
        {
          "text": original sentence,
          "token_ids": [int, ...],
          "token_strs": [str, ...],
          "n_tokens": int,
        }
    """
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    strs = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in ids
    ]
    return {
        "text": text,
        "token_ids": ids,
        "token_strs": strs,
        "n_tokens": len(ids),
    }


def find_token_position(token_strs: list[str], target: str) -> Optional[Position]:
    """
    Return the first token position matching a target string.

    Leading space markers such as Ġ, ▁, and plain spaces are ignored.
    """
    normalized_target = target.strip().lower()
    for i, tok in enumerate(token_strs):
        if tok.strip().lower() == normalized_target:
            return i

    for i, tok in enumerate(token_strs):
        if normalized_target in tok.strip().lower():
            return i
    return None


def find_verb_span(
    tokenizer,
    token_ids: list[int],
    token_strs: list[str],
    verb_form: str,
) -> Optional[VerbSpan]:
    """
    Locate a verb form within a tokenized sentence, even if it spans
    multiple tokens.

    Handles cases like:
      "gidiyorum"    in tokens [" gid", "iyorum"]     -> span [2, 3]
      "gidiyorsun"   in tokens [" gidiyor", "sun"]     -> span [4, 5]
      "gidiyor"      in tokens [" gidiyor"]            -> span [4, 4]

    Algorithm:
      1. Try exact single-token match first.
      2. Otherwise, slide a window across consecutive tokens and check
         if their concatenation (stripped) matches the verb form.

    Returns:
        VerbSpan or None if the form cannot be located.
    """
    normalized = verb_form.strip().lower()

    # 1. Single-token match
    for i, tok in enumerate(token_strs):
        if tok.strip().lower() == normalized:
            return VerbSpan(
                start=i, end=i,
                token_ids=[token_ids[i]],
                token_strs=[token_strs[i]],
                n_tokens=1,
                suffix_split=False,
                suffix_pos=None,
                stem_pos=i,
            )

    # 2. Multi-token sliding window (max 4 tokens)
    max_span = min(4, len(token_strs))
    for span_len in range(2, max_span + 1):
        for start in range(len(token_strs) - span_len + 1):
            window_strs = token_strs[start : start + span_len]
            concat = "".join(t.strip() for t in window_strs).lower()
            # Also try with the first token keeping its leading space stripped
            if concat == normalized:
                end = start + span_len - 1
                span_ids = token_ids[start : start + span_len]
                # Determine if suffix is split
                # Heuristic: if last token is short (<= 5 chars) and looks like
                # a Turkish person suffix, mark it as suffix_split
                last_tok = window_strs[-1].strip().lower()
                known_suffixes = {
                    "um", "ım", "im", "üm",
                    "sun", "sın", "sin", "sün",
                    "uz", "ız", "iz", "üz",
                    "sunuz", "sınız", "siniz", "sünüz",
                    "lar", "ler", "lardı", "lerdi",
                    "m", "n", "k", "niz", "nız", "nuz", "nüz",
                    "ım", "im", "um", "üm",
                    "sın", "sin", "sun", "sün",
                }
                suffix_split = last_tok in known_suffixes
                suffix_pos = end if suffix_split else None
                stem_pos = start

                return VerbSpan(
                    start=start, end=end,
                    token_ids=span_ids,
                    token_strs=window_strs,
                    n_tokens=span_len,
                    suffix_split=suffix_split,
                    suffix_pos=suffix_pos,
                    stem_pos=stem_pos,
                )

    return None


def find_all_token_positions(token_strs: list[str], target: str) -> list[Position]:
    """Return every token position containing the target string."""
    normalized_target = target.strip().lower()
    return [
        i for i, tok in enumerate(token_strs)
        if normalized_target in tok.strip().lower()
    ]


def get_token_id(tokenizer, word: str) -> Optional[TokenId]:
    """
    Return a word's token ID if it maps to exactly one token.

    Returns None for multi-token words.
    """
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def get_token_ids_for_forms(tokenizer, forms: list[str]) -> dict[str, Optional[TokenId]]:
    """
    Return token IDs for multiple verb forms.

    None means the form is split into multiple tokens.
    """
    result = {}
    for form in forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        ids_with_space = tokenizer.encode(" " + form, add_special_tokens=False)

        if len(ids) == 1:
            result[form] = ids[0]
        elif len(ids_with_space) == 1:
            result[form] = ids_with_space[0]
        else:
            result[form] = None
    return result


def classify_verb_tokenization(tokenizer, verb_form: str) -> dict:
    """
    Classify how the tokenizer handles a verb form.

    Returns:
        {
          "form":       the original verb form,
          "tokens":     list of token strings,
          "token_ids":  list of token IDs,
          "n_tokens":   number of tokens,
          "type":       "A_merged" | "B_suffix_split" | "C_multi_split",
          "suffix_token": the suffix string if split, else None,
        }
    """
    ids = tokenizer.encode(" " + verb_form, add_special_tokens=False)
    strs = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in ids
    ]

    known_suffixes = {
        "um", "ım", "im", "üm",
        "sun", "sın", "sin", "sün",
        "uz", "ız", "iz", "üz",
        "sunuz", "sınız", "siniz", "sünüz",
        "lar", "ler",
        "m", "n", "k",
        "niz", "nız", "nuz", "nüz",
    }

    if len(ids) == 1:
        tok_type = "A_merged"
        suffix_tok = None
    elif len(ids) == 2 and strs[-1].strip().lower() in known_suffixes:
        tok_type = "B_suffix_split"
        suffix_tok = strs[-1].strip()
    else:
        # Check if any token at the end is a known suffix
        last = strs[-1].strip().lower()
        if last in known_suffixes:
            tok_type = "B_suffix_split"
            suffix_tok = strs[-1].strip()
        else:
            tok_type = "C_multi_split"
            suffix_tok = None

    return {
        "form": verb_form,
        "tokens": [s.strip() for s in strs],
        "token_ids": ids,
        "n_tokens": len(ids),
        "type": tok_type,
        "suffix_token": suffix_tok,
    }


# ── Sentence analysis functions ──────────────────────────────────────────────

def analyze_sentence(
    tokenizer,
    text: str,
    subject: Optional[str] = None,
    verb_form: Optional[str] = None,
) -> dict:
    """
    Return a full tokenization analysis for one sentence.

    Now handles multi-token verb forms via VerbSpan.

    Returns:
        {
          "text", "token_ids", "token_strs", "n_tokens",
          "subject_pos": int or None,
          "verb_span":   VerbSpan or None,
          "verb_pos":    int (start of verb) or None,
          "verb_end":    int (end of verb) or None,
          "suffix_split": bool,
          "suffix_pos":  int or None,
          "token_table": [(pos, id, str), ...],
        }
    """
    tok_info = tokenize(tokenizer, text)
    token_ids = tok_info["token_ids"]
    token_strs = tok_info["token_strs"]

    subject_pos = find_token_position(token_strs, subject) if subject else None

    verb_span = None
    verb_pos = None
    verb_end = None
    suffix_split = False
    suffix_pos = None

    if verb_form:
        verb_span = find_verb_span(tokenizer, token_ids, token_strs, verb_form)
        if verb_span:
            verb_pos = verb_span.start
            verb_end = verb_span.end
            suffix_split = verb_span.suffix_split
            suffix_pos = verb_span.suffix_pos

    token_table = [
        (i, id_, s)
        for i, (id_, s) in enumerate(zip(token_ids, token_strs))
    ]

    return {
        **tok_info,
        "subject_pos":  subject_pos,
        "verb_span":    verb_span,
        "verb_pos":     verb_pos,
        "verb_end":     verb_end,
        "suffix_split": suffix_split,
        "suffix_pos":   suffix_pos,
        "token_table":  token_table,
    }


def compare_tokenizations(
    tokenizer,
    overt_text: str,
    prodrop_text: str,
    verb_form: Optional[str] = None,
    subject: Optional[str] = None,
) -> dict:
    """
    Compare tokenization for overt-subject and pro-drop versions.
    """
    overt_info = analyze_sentence(tokenizer, overt_text, subject, verb_form)
    prodrop_info = analyze_sentence(tokenizer, prodrop_text, None, verb_form)

    offset = None
    if overt_info["verb_pos"] is not None and prodrop_info["verb_pos"] is not None:
        offset = overt_info["verb_pos"] - prodrop_info["verb_pos"]

    return {
        "overt": overt_info,
        "prodrop": prodrop_info,
        "verb_offset": offset,
    }


# ── Model execution helpers ─────────────────────────────────────────────────

def tokens_to_tensor(tokenizer, text: str, device: str = "cpu") -> torch.Tensor:
    """Convert text to an input-id tensor on the requested device."""
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


def get_logit_diff(
    logits: torch.Tensor,
    target_id: TokenId,
    contrast_id: TokenId,
    pos: int = -1,
) -> float:
    """
    Compute the logit difference between two tokens.

    logit_diff = logit(target) - logit(contrast)

    Positive values mean the model prefers the target token.
    """
    last_logits = logits[0, pos, :]
    return (last_logits[target_id] - last_logits[contrast_id]).item()


def get_top_predictions(
    tokenizer,
    logits: torch.Tensor,
    k: int = 10,
    pos: int = -1,
) -> list[tuple[str, float]]:
    """Return the top-k predictions at a sequence position."""
    last_logits = logits[0, pos, :]
    top_logits, top_ids = torch.topk(last_logits, k)
    return [
        (tokenizer.decode([idx.item()], clean_up_tokenization_spaces=False), val.item())
        for idx, val in zip(top_ids, top_logits)
    ]


def get_token_rank(logits: torch.Tensor, token_id: TokenId, pos: int = -1) -> int:
    """
    Return a token's 1-indexed rank at a sequence position.

    Used in Logit Lens analysis.
    """
    last_logits = logits[0, pos, :]
    sorted_ids = torch.argsort(last_logits, descending=True)
    rank = (sorted_ids == token_id).nonzero(as_tuple=True)[0]
    if len(rank) == 0:
        return -1
    return rank[0].item() + 1
