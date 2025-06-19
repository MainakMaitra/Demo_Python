START
  |
  v
[Define `negation_cues` dictionary]
  - Types: simple, contracted, weak, emphatic
  - Fields: patterns, weight, complaint_relevance
  |
  v
[Loop over Input DataFrame rows]
  |
  v
[Run `extract_negation_patterns()` on:]
  - Customer Transcript
  - Agent Transcript
  |
  v
[For each token in transcript]
  |
  ├─> If token matches `negation_cue`:
  |     |
  |     v
  |   [Create `negation_info` object]
  |     - Cue, Type, Position
  |     - Scope (via dependency parsing)
  |     - Context window (±5 tokens)
  |     - Complaint & Info Relevance (weighted scope)
  |
  v
[Aggregate all negation_info into `negation_df`]
  - Each row = one negation cue instance
  - Multiple rows per transcript possible
  |
  v
[Create Visualizations via `create_negation_visualizations()`]
  |
  ├─> Bar plots:
  |     - Type distribution by TP vs FP
  |     - Avg complaint/info relevance by TP vs FP
  |
  ├─> Trend plots:
  |     - Relevance over time (monthly)
  |     - Volume over time (monthly)
  |
  ├─> Word clouds:
  |     - Contexts of high complaint relevance
  |     - Contexts of high information relevance
  |
  v
[Print Summary Stats]
  - Total cues detected
  - Avg scores by TP vs FP
  - Cues by speaker role
  |
  v
END
