Interpreting the metrics (and what to add next)

CER / WER (core fidelity): For text preservation, add/track Character Error Rate (CER) and Word Error Rate (WER). They’re both Levenshtein-based error rates (insertions + deletions + substitutions over length). You can compute WER with the Python jiwer library; it wraps the standard definition and common transforms.

Levenshtein distance: Useful as a raw edit distance or normalized (NER = distance / |reference|) for page-level severity ranking.

n-gram coverage (BLEU-style): Use a BLEU-like n-gram precision signal to catch paraphrasing/omissions that don’t show as many edits but still drop content. You can pull a lightweight BLEU implementation from Hugging Face’s evaluate space or any standard reference.

Layout/structure retention: A cheap proxy is Jaccard similarity over detected structural cues (e.g., bullet tokens, “Theorem/Proof”, “Figure/Table”, equation markers like \(, \[), or over block labels if your backends emit them. Jaccard = |A∩B| / |A∪B|.
