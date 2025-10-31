# dev/eval/metrics.py
from __future__ import annotations
from typing import Tuple

def _levenshtein(a: str, b: str) -> int:
    """Char-level Levenshtein distance."""
    n, m = len(a), len(b)
    if n < m: a, b, n, m = b, a, m, n
    prev = list(range(m+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[m]

def cer(hyp: str, ref: str) -> float:
    """Character Error Rate (lower is better)."""  # CER = edit_distance / |ref|
    if not ref:
        return 0.0 if not hyp else 1.0
    return _levenshtein(hyp, ref) / max(1, len(ref))

def _tokenize_words(s: str):
    return s.strip().split()

def wer(hyp: str, ref: str) -> float:
    """Word Error Rate (lower is better)."""  # WER = (S+D+I) / #ref_words
    r = _tokenize_words(ref); h = _tokenize_words(hyp)
    # Build word-level Levenshtein
    n, m = len(r), len(h)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / max(1, len(r))

def bertscore_f1(hyp: str, ref: str) -> float | None:
    """Optional semantic similarity via BERTScore F1 (0..1)."""
    try:
        from bert_score import score as bert_score
    except Exception:
        return None
    P, R, F1 = bert_score([hyp], [ref], lang="en", rescale_with_baseline=True)
    return float(F1[0])
