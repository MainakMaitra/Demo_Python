import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
import shap

# ---------------------------------------
# Step 1: Load and prepare the data
# ---------------------------------------

# Assume `negation_df` is already defined and has:
# Columns: 'Negation_Cue', 'Primary_Marker' (TP/FP)

# Binary label
negation_df['label'] = negation_df['Primary_Marker'].apply(lambda x: 1 if x == 'TP' else 0)

# One-hot encode cues
X = pd.get_dummies(negation_df['Negation_Cue'])
y = negation_df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# ---------------------------------------
# Step 2: Train model + compute SHAP values
# ---------------------------------------

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# SHAP explain
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# SHAP values → mean absolute contribution per cue
shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
shap_mean = shap_df.abs().mean().sort_values(ascending=False)

# Normalize SHAP scores (0–1) → complaint relevance
normalized_shap = (shap_mean - shap_mean.min()) / (shap_mean.max() - shap_mean.min())

# ---------------------------------------
# Step 3: Compute cue frequencies
# ---------------------------------------

cue_counts = Counter(negation_df['Negation_Cue'])
max_freq = max(cue_counts.values())
normalized_freqs = {cue: count / max_freq for cue, count in cue_counts.items()}

# ---------------------------------------
# Step 4: Define negation types
# ---------------------------------------

negation_structure = {
    'simple_negation': ['not', 'no', 'never', 'none', 'nothing', 'nobody', 'nowhere'],
    'contracted_negation': ["don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't", 
                            "doesn't", "didn't", "haven't", "hasn't", "hadn't"],
    'weak_negation': ['hardly', 'barely', 'scarcely', 'rarely', 'seldom'],
    'emphatic_negation': ['absolutely not', 'definitely not', 'certainly not', 'never ever']
}

# ---------------------------------------
# Step 5: Learn complaint_relevance and weight
# ---------------------------------------

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
        freq_score = normalized_freqs.get(cue, 0.0)
        
        complaint_scores.append(shap_score)
        weight_scores.append(shap_score * freq_score)
    
    if complaint_scores:
        cue_entry['complaint_relevance'] = round(sum(complaint_scores) / len(complaint_scores), 3)
    if weight_scores:
        cue_entry['weight'] = round(sum(weight_scores) / len(weight_scores), 3)

    learned_negation_cues[neg_type] = cue_entry

# ---------------------------------------
# Step 6: Display Results
# ---------------------------------------

from pprint import pprint
print("\nLearned Negation Cues with Complaint Relevance and Weight:")
pprint(learned_negation_cues)
