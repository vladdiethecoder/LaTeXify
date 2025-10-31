# dev/eval/consensus.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class AlignedToken:
    tokens: List[str]  # one per system (or "<eps>" for gap)

GAP = "<eps>"

def _tokenize(s: str) -> List[str]:
    # very light tokenizer; swap with regex/icu if needed
    return s.strip().split()

def _align_pair(a: List[str], b: List[str]) -> List[Tuple[str, str]]:
    """Simple dynamic-programming alignment (Levenshtein backtrace at token level)."""
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[(0,0)]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = (i-1, 0)
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = (0, j-1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            cand = [(dp[i-1][j]+1, (i-1,j)), (dp[i][j-1]+1, (i,j-1)), (dp[i-1][j-1]+cost, (i-1,j-1))]
            dp[i][j], bt[i][j] = min(cand, key=lambda x:x[0])
    i, j = n, m
    out = []
    while i>0 or j>0:
        pi, pj = bt[i][j]
        if pi==i-1 and pj==j-1:
            out.append((a[i-1], b[j-1]))
        elif pi==i-1 and pj==j:
            out.append((a[i-1], GAP))
        else:
            out.append((GAP, b[j-1]))
        i, j = pi, pj
    return list(reversed(out))

def _merge_alignment(cur: List[AlignedToken], other: List[str]) -> List[AlignedToken]:
    """Merge a new hypothesis into the running alignment matrix."""
    a = [row.tokens[0] for row in cur] if cur else []
    # build pair alignment between current 'consensus timeline' (a) and other
    pair = _align_pair(a, other)
    merged: List[AlignedToken] = []
    ai = 0
    for tok_a, tok_b in pair:
        if tok_a != GAP:
            row = cur[ai]
            merged.append(AlignedToken(tokens=row.tokens + [tok_b]))
            ai += 1
        else:
            # insertion in 'other' relative to current timeline
            merged.append(AlignedToken(tokens=[GAP]*(len(cur[0].tokens) if cur else 1) + [tok_b]))
    return merged

def rover_consensus(hypotheses: List[str], min_vote: int = 1) -> str:
    """
    ROVER-style: align all systems at token level, then majority vote per column.
    min_vote lets you drop weakly-supported tokens.
    """
    if not hypotheses:
        return ""
    token_lists = [_tokenize(h) for h in hypotheses]
    # bootstrap alignment with the longest hypothesis (stabilizes columns)
    token_lists.sort(key=len, reverse=True)
    aligned: List[AlignedToken] = [AlignedToken(tokens=[t]) for t in token_lists[0]]
    for h in token_lists[1:]:
        aligned = _merge_alignment(aligned, h)
    # vote per column
    out = []
    for col in aligned:
        counts = {}
        for t in col.tokens:
            if t == GAP:
                continue
            counts[t] = counts.get(t, 0) + 1
        if not counts:
            continue
        winner, votes = max(counts.items(), key=lambda kv: (kv[1], len(kv[0])))
        if votes >= min_vote:
            out.append(winner)
    return " ".join(out)
