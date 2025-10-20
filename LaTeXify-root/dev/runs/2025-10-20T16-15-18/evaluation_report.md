# OCR Evaluation Report

Run: `2025-10-20T16-15-18`

## Per-page Summary

### page-0001.png

- **nanonets-ocr-s**: len=86, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0001.md`
- **nanonets-ocr2-3b**: len=86, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0001.md`
- **qwen2-vl-ocr-2b-instruct**: len=67, jaccard_vs_ref=0.474, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0001.md`

**Issues:**
- qwen2-vl-ocr-2b-instruct: low_jaccard(0.47)

### page-0002.png

- **nanonets-ocr-s**: len=878, jaccard_vs_ref=0.863, struct_delta={'bullets': 1, 'code_fences': 0, 'eq_block': 6, 'eq_inline': 12, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0002.md`
- **nanonets-ocr2-3b**: len=886, jaccard_vs_ref=0.863, struct_delta={'bullets': 4, 'code_fences': 0, 'eq_block': 5, 'eq_inline': 12, 'headings': 2}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0002.md`
- **qwen2-vl-ocr-2b-instruct**: len=1020, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0002.md`

**Issues:**
- nanonets-ocr-s: struct_mismatch({'bullets': 1, 'code_fences': 0, 'eq_block': 6, 'eq_inline': 12, 'headings': 0})
- nanonets-ocr2-3b: struct_mismatch({'bullets': 4, 'code_fences': 0, 'eq_block': 5, 'eq_inline': 12, 'headings': 2})

### page-0003.png

- **nanonets-ocr-s**: len=1011, jaccard_vs_ref=0.548, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 24, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0003.md`
- **nanonets-ocr2-3b**: len=1043, jaccard_vs_ref=0.548, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 26, 'eq_inline': 26, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0003.md`
- **qwen2-vl-ocr-2b-instruct**: len=2272, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0003.md`

**Issues:**
- nanonets-ocr-s: struct_mismatch({'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 24, 'headings': 0})
- nanonets-ocr2-3b: struct_mismatch({'bullets': 0, 'code_fences': 0, 'eq_block': 26, 'eq_inline': 26, 'headings': 0})

### page-0004.png

- **nanonets-ocr-s**: len=925, jaccard_vs_ref=0.685, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0004.md`
- **nanonets-ocr2-3b**: len=1029, jaccard_vs_ref=0.670, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 3}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0004.md`
- **qwen2-vl-ocr-2b-instruct**: len=1089, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0004.md`

### page-0005.png

- **nanonets-ocr-s**: len=537, jaccard_vs_ref=0.185, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0005.md`
- **nanonets-ocr2-3b**: len=583, jaccard_vs_ref=0.175, struct_delta={'bullets': 2, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0005.md`
- **qwen2-vl-ocr-2b-instruct**: len=2085, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0005.md`

**Issues:**
- nanonets-ocr-s: low_jaccard(0.18)
- nanonets-ocr2-3b: low_jaccard(0.18)

### page-0006.png

- **nanonets-ocr-s**: len=1051, jaccard_vs_ref=0.649, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0006.md`
- **nanonets-ocr2-3b**: len=872, jaccard_vs_ref=0.394, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0006.md`
- **qwen2-vl-ocr-2b-instruct**: len=2762, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0006.md`

**Issues:**
- nanonets-ocr2-3b: low_jaccard(0.39)

### page-0007.png

- **nanonets-ocr-s**: len=981, jaccard_vs_ref=0.928, struct_delta={'bullets': -9, 'code_fences': 0, 'eq_block': 9, 'eq_inline': 29, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0007.md`
- **nanonets-ocr2-3b**: len=1135, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0007.md`
- **qwen2-vl-ocr-2b-instruct**: len=59, jaccard_vs_ref=0.151, struct_delta={'bullets': -9, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0007.md`

**Issues:**
- nanonets-ocr-s: struct_mismatch({'bullets': -9, 'code_fences': 0, 'eq_block': 9, 'eq_inline': 29, 'headings': 0})
- qwen2-vl-ocr-2b-instruct: low_jaccard(0.15), struct_mismatch({'bullets': -9, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0})

### page-0008.png

- **nanonets-ocr-s**: len=682, jaccard_vs_ref=0.731, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 12, 'eq_inline': 23, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0008.md`
- **nanonets-ocr2-3b**: len=729, jaccard_vs_ref=0.776, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0008.md`
- **qwen2-vl-ocr-2b-instruct**: len=989, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0008.md`

**Issues:**
- nanonets-ocr-s: struct_mismatch({'bullets': 0, 'code_fences': 0, 'eq_block': 12, 'eq_inline': 23, 'headings': 0})

### page-0009.png

- **nanonets-ocr-s**: len=474, jaccard_vs_ref=0.721, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 6, 'eq_inline': 15, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr-s/page-0009.md`
- **nanonets-ocr2-3b**: len=511, jaccard_vs_ref=0.797, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/nanonets-ocr2-3b/page-0009.md`
- **qwen2-vl-ocr-2b-instruct**: len=605, jaccard_vs_ref=1.000, struct_delta={'bullets': 0, 'code_fences': 0, 'eq_block': 0, 'eq_inline': 0, 'headings': 0}  
  file: `dev/runs/2025-10-20T16-15-18/outputs/qwen2-vl-ocr-2b-instruct/page-0009.md`

**Issues:**
- nanonets-ocr-s: struct_mismatch({'bullets': 0, 'code_fences': 0, 'eq_block': 6, 'eq_inline': 15, 'headings': 0})

## Consensus (proxy)

- **page-0001.png**: consensus_len=86, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0002.png**: consensus_len=1020, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0003.png**: consensus_len=2272, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0004.png**: consensus_len=1089, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0005.png**: consensus_len=2085, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0006.png**: consensus_len=2762, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0007.png**: consensus_len=1135, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0008.png**: consensus_len=989, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
- **page-0009.png**: consensus_len=605, jaccard_vs_ref=1.000, struct_delta={'headings': 0, 'bullets': 0, 'code_fences': 0, 'eq_inline': 0, 'eq_block': 0}
