# Original structure
negation_structure = {
    'simple_negation': ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'],
    'contracted_negation': ["don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't", 
                            "doesn't", "didn't", "haven't", "hasn't", "hadn't"],
    'weak_negation': ['hardly', 'barely', 'scarcely', 'rarely', 'seldom'],
    'emphatic_negation': ['absolutely not', 'definitely not', 'certainly not', 'never ever']
}

# Your original weights
negation_weights = {
    'simple_negation': 1.0,
    'contracted_negation': 1.2,
    'weak_negation': 0.6,
    'emphatic_negation': 2.0
}

# Generate the updated dictionary
learned_negation_cues = {}

for neg_type, patterns in negation_structure.items():
    cue_entry = {
        'patterns': patterns,
        'weight': negation_weights[neg_type],
        'complaint_relevance': 0.0  # Will be overridden below
    }

    # Calculate average SHAP-based complaint relevance for this type
    learned_scores = []
    for cue in patterns:
        if cue in normalized_scores:
            learned_scores.append(normalized_scores[cue])
    
    # If we found learned scores, average them
    if learned_scores:
        cue_entry['complaint_relevance'] = round(sum(learned_scores) / len(learned_scores), 3)
    else:
        cue_entry['complaint_relevance'] = 0.0  # Fallback for missing cues

    learned_negation_cues[neg_type] = cue_entry
