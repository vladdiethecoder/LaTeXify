# scripts/metrics_text.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any
import math
import re

# Optional WER via jiwer if present
try:
    from jiwer import wer as _jiwer
except Exception:
    _jiwer = None


def _normalize(s: str) -> str:
    if s is None:
        return ""
    # basic normalization suitable for OCR-ish text
    s = s.replace("\u00A0", " ").replace("\t", " ")
    s = re.sub(r"[ \r\f\v]+", " ", s)
    return s.strip()


def _tokens(s: str) -> List[str]:
    return _normalize(s).split()


def _charseq(s: str) -> List[str]:
    return list(_normalize(s))


def levenshtein(a: str, b: str) -> int:
    """Plain DP Levenshtein distance (chars)."""
    A, B = _charseq(a), _charseq(b)
    n, m = len(A), len(B)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if A[i - 1] == B[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,       # deletion
                cur[j - 1] + 1,    # insertion
                prev[j - 1] + cost # substitution
            )
        prev = cur
    return prev[m]


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate = edit_distance(chars) / len(ref_chars)."""
    ref_chars = len(_charseq(ref))
    if ref_chars == 0:
        return 0.0 if len(_charseq(hyp)) == 0 else 1.0
    return levenshtein(hyp, ref) / ref_chars


def wer(hyp: str, ref: str) -> float:
    """Word Error Rate. Uses jiwer if available, else falls back to word-level Levenshtein/len(ref_words)."""
    ref_toks = _tokens(ref)
    hyp_toks = _tokens(hyp)
    if len(ref_toks) == 0:
        return 0.0 if len(hyp_toks) == 0 else 1.0
    if _jiwer is not None:
        try:
            return float(_jiwer(ref, hyp))
        except Exception:
            pass
    # fallback: Levenshtein on words
    n, m = len(ref_toks), len(hyp_toks)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_toks[i - 1] == hyp_toks[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(1, len(ref_toks))


def jaccard(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """Jaccard similarity on token sets."""
    A, B = set(tokens_a), set(tokens_b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def _n_grams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i:i+n]) for i in range(0, len(tokens)-n+1)]


def bleuish(hyp: str, ref: str, max_n: int = 4) -> float:
    """
    Lightweight BLEU-ish: uniform n=1..max_n modified precision, brevity penalty.
    NOT a drop-in for sacreBLEU; good enough as a stand-in signal.
    """
    hyp_t = _tokens(hyp)
    ref_t = _tokens(ref)
    if not hyp_t and not ref_t:
        return 1.0
    if not hyp_t or not ref_t:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hg = _n_grams(hyp_t, n)
        rg = _n_grams(ref_t, n)
        if not hg or not rg:
            precisions.append(0.0)
            continue
        hcount: Dict[Tuple[str, ...], int] = {}
        for g in hg:
            hcount[g] = hcount.get(g, 0) + 1
        rcount: Dict[Tuple[str, ...], int] = {}
        for g in rg:
            rcount[g] = rcount.get(g, 0) + 1
        match = 0
        for g, c in hcount.items():
            match += min(c, rcount.get(g, 0))
        precisions.append(match / max(1, len(hg)))

    # brevity penalty
    hyp_len = len(hyp_t)
    ref_len = len(ref_t)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    # geometric mean (avoid log(0))
    log_p = 0.0
    for p in precisions:
        log_p += math.log(max(p, 1e-12))
    geo_mean = math.exp(log_p / max_n)
    return float(bp * geo_mean)


def meteorish(hyp: str, ref: str) -> float:
    """Approximate METEOR-style score using unigram precision/recall + fragmentation penalty."""
    hyp_t = _tokens(hyp)
    ref_t = _tokens(ref)
    if not hyp_t and not ref_t:
        return 1.0
    if not hyp_t or not ref_t:
        return 0.0

    # greedy alignment left-to-right
    ref_positions: Dict[str, List[int]] = {}
    for idx, tok in enumerate(ref_t):
        ref_positions.setdefault(tok, []).append(idx)

    matches: List[Tuple[int, int]] = []
    used_positions: Dict[str, int] = {}
    last_ref_idx = -1
    for h_idx, tok in enumerate(hyp_t):
        positions = ref_positions.get(tok)
        if not positions:
            continue
        start = used_positions.get(tok, 0)
        chosen = None
        for pos_idx in range(start, len(positions)):
            ref_idx = positions[pos_idx]
            if ref_idx > last_ref_idx:
                chosen = ref_idx
                used_positions[tok] = pos_idx + 1
                break
        if chosen is None:
            continue
        matches.append((h_idx, chosen))
        last_ref_idx = chosen

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(hyp_t)
    recall = match_count / len(ref_t)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    f_mean = (10 * precision * recall) / (recall + 9 * precision) if (recall + 9 * precision) else 0.0

    chunks = 1
    for i in range(1, match_count):
        prev_h, prev_r = matches[i - 1]
        cur_h, cur_r = matches[i]
        if cur_h != prev_h + 1 or cur_r != prev_r + 1:
            chunks += 1
    frag = chunks / match_count
    penalty = 0.5 * (frag ** 3)
    return max(0.0, (1 - penalty) * f_mean)


@dataclass
class TextScores:
    wer: float
    cer: float
    lev_dist: int
    jaccard_unigram: float
    bleuish: float
    meteorish: float
    ref_len_chars: int
    hyp_len_chars: int
    ref_len_words: int
    hyp_len_words: int

    @classmethod
    def compute(cls, hyp: str, ref: str) -> "TextScores":
        w = wer(hyp, ref)
        c = cer(hyp, ref)
        d = levenshtein(hyp, ref)
        j = jaccard(_tokens(hyp), _tokens(ref))
        b = bleuish(hyp, ref)
        m = meteorish(hyp, ref)
        return cls(
            wer=w, cer=c, lev_dist=d, jaccard_unigram=j, bleuish=b, meteorish=m,
            ref_len_chars=len(_charseq(ref)),
            hyp_len_chars=len(_charseq(hyp)),
            ref_len_words=len(_tokens(ref)),
            hyp_len_words=len(_tokens(hyp)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "wer": self.wer,
            "cer": self.cer,
            "lev_dist": self.lev_dist,
            "jaccard_unigram": self.jaccard_unigram,
            "bleuish": self.bleuish,
            "meteorish": self.meteorish,
            "ref_len_chars": self.ref_len_chars,
            "hyp_len_chars": self.hyp_len_chars,
            "ref_len_words": self.ref_len_words,
            "hyp_len_words": self.hyp_len_words,
        }
