import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import shap

def learn_negation_cue_scores(negation_df):
    """
    Learn complaint_relevance and weight for each negation cue category
    using SHAP values and cue frequency.
    Returns a dictionary with updated scores.
    """
    # ------------------------
    # Configuration
    # ------------------------
    MIN_RELEVANCE = 0.05
    MIN_WEIGHT = 0.05

    legacy_defaults = {
        'simple_negation': 0.7,
        'contracted_negation': 0.8,
        'weak_negation': 0.5,
        'emphatic_negation': 0.9
    }

    negation_structure = {
        'simple_negation': ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'],
        'contracted_negation': ["don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't", 
                                "doesn't", "didn't", "haven't", "hasn't", "hadn't"],
        'weak_negation': ['hardly', 'barely', 'scarcely', 'rarely', 'seldom'],
        'emphatic_negation': ['absolutely not', 'definitely not', 'certainly not', 'never ever']
    }

    # ------------------------
    # Step 1: SHAP scoring
    # ------------------------
    df = negation_df.copy()
    df['label'] = df['Primary_Marker'].apply(lambda x: 1 if x == 'TP' else 0)
    X = pd.get_dummies(df['Negation_Cue'])
    y = df['label']
    
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, random_state=42)
    
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    shap_mean = shap_df.abs().mean().sort_values(ascending=False)
    normalized_shap = (shap_mean - shap_mean.min()) / (shap_mean.max() - shap_mean.min())

    # ------------------------
    # Step 2: Frequency scoring
    # ------------------------
    cue_counts = Counter(df['Negation_Cue'])
    max_freq = max(cue_counts.values())

    # ------------------------
    # Step 3: Learn scores per type
    # ------------------------
    learned_negation_cues = {}

    for neg_type, patterns in negation_structure.items():
        cue_entry = {
            'patterns': patterns,
            'complaint_relevance': 0.0,
            'weight': 0.0
        }

        complaint_scores = []
        weight_scores = []

        for cue in patterns:
            shap_score = normalized_shap.get(cue, 0.0)
            raw_freq = cue_counts.get(cue, 0)
            log_freq_score = np.log1p(raw_freq) / np.log1p(max_freq) if max_freq > 0 else 0

            complaint_scores.append(shap_score)
            weight_scores.append(shap_score * log_freq_score)

        # Modulated assignment with smoothing fallback
        avg_complaint = np.mean(complaint_scores) if complaint_scores else 0
        avg_weight = np.mean(weight_scores) if weight_scores else 0

        cue_entry['complaint_relevance'] = float(round(max(avg_complaint, MIN_RELEVANCE, legacy_defaults[neg_type]), 3))
        cue_entry['weight'] = float(round(max(avg_weight, MIN_WEIGHT), 3))

        learned_negation_cues[neg_type] = cue_entry

    return learned_negation_cues
