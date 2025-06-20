# ============================================================================
# PART 1: DYNAMIC PATTERN DISCOVERY FROM DATA
# Data-driven approach to discover negation patterns
# ============================================================================

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DYNAMIC NEGATION PATTERN DISCOVERY")
print("Learning patterns from actual data instead of hardcoding")
print("=" * 80)

def load_and_prepare_data():
    """Load the data for pattern discovery"""
    try:
        df = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        df.columns = df.columns.str.rstrip()
        df = df[df['Prosodica L1'].str.lower() != 'dissatisfaction']
        
        # Enhanced data preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
        
        # Period classification
        pre_months = ['2024-10', '2024-11', '2024-12']
        post_months = ['2025-01', '2025-02', '2025-03']
        
        df['Period'] = df['Year_Month'].apply(
            lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
        )
        
        # Text processing
        df['Customer Transcript'] = df['Customer Transcript'].fillna('')
        df['Agent Transcript'] = df['Agent Transcript'].fillna('')
        df['Full_Transcript'] = df['Customer Transcript'] + ' ' + df['Agent Transcript']
        
        print(f"Data loaded successfully: {df.shape}")
        return df
        
    except FileNotFoundError:
        print("Error: Main dataset file not found.")
        return None

def discover_negation_patterns(df):
    """
    Dynamically discover negation patterns from the data
    instead of using hardcoded patterns
    """
    
    print("\n" + "=" * 60)
    print("DISCOVERING NEGATION PATTERNS FROM DATA")
    print("=" * 60)
    
    # Load spaCy model for linguistic analysis
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model not found. Using regex-based approach.")
        nlp = None
    
    # Step 1: Extract all negation contexts from transcripts
    print("1. Extracting negation contexts from transcripts...")
    
    negation_contexts = []
    
    # Base negation words to start pattern discovery
    base_negation_words = [
        'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'none',
        "don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't",
        "doesn't", "didn't", "haven't", "hasn't", "hadn't", "couldn't",
        "wouldn't", "shouldn't", "mustn't"
    ]
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing transcript {idx+1}/{len(df)}...")
        
        # Process both customer and agent transcripts
        for transcript_type in ['Customer Transcript', 'Agent Transcript']:
            text = str(row[transcript_type]).lower()
            
            # Find negation contexts
            for neg_word in base_negation_words:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(neg_word) + r'\b'
                
                for match in re.finditer(pattern, text):
                    start_pos = max(0, match.start() - 100)  # 100 chars before
                    end_pos = min(len(text), match.end() + 100)  # 100 chars after
                    
                    context = text[start_pos:end_pos]
                    
                    negation_contexts.append({
                        'UUID': row['UUID'],
                        'Primary_Marker': row['Primary Marker'],
                        'Period': row['Period'],
                        'Speaker': transcript_type.split()[0].lower(),
                        'Negation_Word': neg_word,
                        'Context': context,
                        'Position_In_Text': match.start(),
                        'Full_Text_Length': len(text)
                    })
    
    print(f"Extracted {len(negation_contexts)} negation contexts")
    
    # Step 2: Cluster negation contexts to discover patterns
    print("\n2. Clustering negation contexts to discover patterns...")
    
    negation_df = pd.DataFrame(negation_contexts)
    
    if len(negation_df) == 0:
        print("No negation contexts found!")
        return None, None
    
    # Use TF-IDF to vectorize contexts
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=5
    )
    
    try:
        context_vectors = vectorizer.fit_transform(negation_df['Context'])
        
        # Cluster contexts to discover patterns
        n_clusters = min(8, max(3, len(negation_df) // 100))  # Dynamic cluster number
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        negation_df['Pattern_Cluster'] = kmeans.fit_predict(context_vectors)
        
        print(f"Discovered {n_clusters} negation pattern clusters")
        
    except Exception as e:
        print(f"Clustering failed: {e}")
        negation_df['Pattern_Cluster'] = 0
    
    # Step 3: Analyze patterns by TP/FP and Pre/Post
    print("\n3. Analyzing discovered patterns...")
    
    pattern_analysis = []
    
    for cluster_id in negation_df['Pattern_Cluster'].unique():
        cluster_data = negation_df[negation_df['Pattern_Cluster'] == cluster_id]
        
        # Get top terms for this cluster
        cluster_indices = negation_df[negation_df['Pattern_Cluster'] == cluster_id].index
        cluster_vectors = context_vectors[cluster_indices]
        
        if cluster_vectors.shape[0] > 0:
            # Get feature importance for this cluster
            feature_names = vectorizer.get_feature_names_out()
            cluster_importance = np.mean(cluster_vectors.toarray(), axis=0)
            top_features_idx = np.argsort(cluster_importance)[-10:]
            top_features = [feature_names[i] for i in top_features_idx[::-1]]
        else:
            top_features = []
        
        # Analyze TP vs FP distribution
        tp_count = len(cluster_data[cluster_data['Primary_Marker'] == 'TP'])
        fp_count = len(cluster_data[cluster_data['Primary_Marker'] == 'FP'])
        
        # Analyze Pre vs Post distribution
        pre_count = len(cluster_data[cluster_data['Period'] == 'Pre'])
        post_count = len(cluster_data[cluster_data['Period'] == 'Post'])
        
        # Speaker analysis
        customer_count = len(cluster_data[cluster_data['Speaker'] == 'customer'])
        agent_count = len(cluster_data[cluster_data['Speaker'] == 'agent'])
        
        pattern_analysis.append({
            'Cluster_ID': cluster_id,
            'Total_Count': len(cluster_data),
            'TP_Count': tp_count,
            'FP_Count': fp_count,
            'TP_Rate': tp_count / len(cluster_data) if len(cluster_data) > 0 else 0,
            'FP_Rate': fp_count / len(cluster_data) if len(cluster_data) > 0 else 0,
            'Pre_Count': pre_count,
            'Post_Count': post_count,
            'Customer_Count': customer_count,
            'Agent_Count': agent_count,
            'Top_Features': ', '.join(top_features[:5]),
            'Sample_Contexts': list(cluster_data['Context'].head(3))
        })
    
    pattern_analysis_df = pd.DataFrame(pattern_analysis)
    pattern_analysis_df = pattern_analysis_df.sort_values('Total_Count', ascending=False)
    
    print("\nDiscovered Negation Patterns:")
    print("=" * 60)
    
    for _, pattern in pattern_analysis_df.iterrows():
        print(f"\nCluster {pattern['Cluster_ID']} ({pattern['Total_Count']} instances):")
        print(f"  TP Rate: {pattern['TP_Rate']:.3f} | FP Rate: {pattern['FP_Rate']:.3f}")
        print(f"  Pre: {pattern['Pre_Count']} | Post: {pattern['Post_Count']}")
        print(f"  Customer: {pattern['Customer_Count']} | Agent: {pattern['Agent_Count']}")
        print(f"  Key Terms: {pattern['Top_Features']}")
        print(f"  Sample: {pattern['Sample_Contexts'][0][:100]}...")
    
    return negation_df, pattern_analysis_df

def extract_dynamic_negation_types(negation_df, pattern_analysis_df):
    """
    Extract dynamic negation types based on discovered patterns
    instead of hardcoded categories
    """
    
    print("\n" + "=" * 60)
    print("EXTRACTING DYNAMIC NEGATION TYPES")
    print("=" * 60)
    
    # Analyze patterns to create dynamic types
    dynamic_types = {}
    
    for _, pattern in pattern_analysis_df.iterrows():
        cluster_id = pattern['Cluster_ID']
        tp_rate = pattern['TP_Rate']
        fp_rate = pattern['FP_Rate']
        features = pattern['Top_Features']
        
        # Classify pattern based on characteristics
        if tp_rate > 0.8:
            pattern_type = "Strong_Complaint_Negation"
        elif fp_rate > 0.7:
            pattern_type = "Information_Seeking_Negation"
        elif 'agent' in features.lower() or 'explain' in features.lower():
            pattern_type = "Agent_Explanation_Negation"
        elif any(word in features.lower() for word in ['understand', 'know', 'sure', 'clear']):
            pattern_type = "Uncertainty_Negation"
        elif any(word in features.lower() for word in ['received', 'got', 'working', 'fixed']):
            pattern_type = "Service_Issue_Negation"
        else:
            pattern_type = f"Mixed_Pattern_{cluster_id}"
        
        dynamic_types[cluster_id] = {
            'type': pattern_type,
            'tp_rate': tp_rate,
            'fp_rate': fp_rate,
            'complaint_likelihood': tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0,
            'features': features
        }
    
    print("Dynamic Negation Types Discovered:")
    print("=" * 40)
    
    for cluster_id, type_info in dynamic_types.items():
        print(f"\nCluster {cluster_id}: {type_info['type']}")
        print(f"  Complaint Likelihood: {type_info['complaint_likelihood']:.3f}")
        print(f"  TP Rate: {type_info['tp_rate']:.3f} | FP Rate: {type_info['fp_rate']:.3f}")
        print(f"  Key Features: {type_info['features']}")
    
    return dynamic_types

def analyze_temporal_pattern_evolution(negation_df):
    """
    Analyze how negation patterns evolved from Pre to Post period
    """
    
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERN EVOLUTION ANALYSIS")
    print("=" * 60)
    
    # Pre vs Post analysis by cluster
    temporal_analysis = []
    
    for cluster_id in negation_df['Pattern_Cluster'].unique():
        cluster_data = negation_df[negation_df['Pattern_Cluster'] == cluster_id]
        
        pre_data = cluster_data[cluster_data['Period'] == 'Pre']
        post_data = cluster_data[cluster_data['Period'] == 'Post']
        
        if len(pre_data) > 0 and len(post_data) > 0:
            pre_tp_rate = len(pre_data[pre_data['Primary_Marker'] == 'TP']) / len(pre_data)
            post_tp_rate = len(post_data[post_data['Primary_Marker'] == 'TP']) / len(post_data)
            
            pre_fp_rate = len(pre_data[pre_data['Primary_Marker'] == 'FP']) / len(pre_data)
            post_fp_rate = len(post_data[post_data['Primary_Marker'] == 'FP']) / len(post_data)
            
            temporal_analysis.append({
                'Cluster_ID': cluster_id,
                'Pre_Count': len(pre_data),
                'Post_Count': len(post_data),
                'Pre_TP_Rate': pre_tp_rate,
                'Post_TP_Rate': post_tp_rate,
                'Pre_FP_Rate': pre_fp_rate,
                'Post_FP_Rate': post_fp_rate,
                'TP_Rate_Change': post_tp_rate - pre_tp_rate,
                'FP_Rate_Change': post_fp_rate - pre_fp_rate,
                'Volume_Change': len(post_data) - len(pre_data),
                'Volume_Change_Pct': ((len(post_data) - len(pre_data)) / len(pre_data)) * 100 if len(pre_data) > 0 else 0
            })
    
    temporal_df = pd.DataFrame(temporal_analysis)
    temporal_df = temporal_df.sort_values('FP_Rate_Change', ascending=False)
    
    print("Temporal Evolution of Negation Patterns:")
    print("=" * 50)
    
    for _, row in temporal_df.iterrows():
        print(f"\nCluster {row['Cluster_ID']}:")
        print(f"  Volume Change: {row['Volume_Change']:+d} ({row['Volume_Change_Pct']:+.1f}%)")
        print(f"  TP Rate: {row['Pre_TP_Rate']:.3f} -> {row['Post_TP_Rate']:.3f} ({row['TP_Rate_Change']:+.3f})")
        print(f"  FP Rate: {row['Pre_FP_Rate']:.3f} -> {row['Post_FP_Rate']:.3f} ({row['FP_Rate_Change']:+.3f})")
        
        if row['FP_Rate_Change'] > 0.1:
            print(f"  WARNING: Significant FP rate increase!")
        elif row['TP_Rate_Change'] < -0.1:
            print(f"  WARNING: Significant TP rate decrease!")
    
    return temporal_df

# Main execution
if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    if df is not None:
        # Discover patterns dynamically
        negation_df, pattern_analysis_df = discover_negation_patterns(df)
        
        if negation_df is not None:
            # Extract dynamic types
            dynamic_types = extract_dynamic_negation_types(negation_df, pattern_analysis_df)
            
            # Analyze temporal evolution
            temporal_df = analyze_temporal_pattern_evolution(negation_df)
            
            print("\n" + "=" * 80)
            print("PART 1 COMPLETED: Dynamic Pattern Discovery")
            print("Patterns discovered from data instead of hardcoded rules")
            print("=" * 80)
        else:
            print("Pattern discovery failed!")
    else:
        print("Data loading failed!")
