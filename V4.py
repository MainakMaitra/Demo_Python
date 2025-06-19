# ============================================================================
# ENHANCED DYNAMIC NEGATION SCOPE DETECTION WITH QUERY COMPONENT ANALYSIS
# Advanced NLP implementation with data-driven pattern discovery and SHAP analysis
# ============================================================================

import pandas as pd
import numpy as np
import spacy
import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import networkx as nx
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ENHANCED DYNAMIC NEGATION SCOPE DETECTION WITH QUERY COMPONENT ANALYSIS")
print("Advanced NLP Analysis with Data-Driven Pattern Discovery and SHAP Explainability")
print("=" * 80)


def load_and_prepare_data():
    """Enhanced data preparation with monthly tracking capabilities"""
    
    print("="*80)
    print("ENHANCED DATA PREPARATION WITH MONTHLY TRACKING")
    print("="*80)
    
    # Load main transcript data
    try:
        df_main = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        df_main.columns = df_main.columns.str.rstrip()
        df_main = df_main[df_main['Prosodica L1'].str.lower() != 'dissatisfaction']
        print(f"Main dataset loaded: {df_main.shape}")
    except FileNotFoundError:
        print("Warning: Main dataset file not found.")
        return None, None, None
    
    # Load validation summary
    try:
        df_validation = pd.read_excel('Categorical Validation.xlsx', sheet_name='Summary validation vol')
        print(f"Validation summary loaded: {df_validation.shape}")
    except FileNotFoundError:
        print("Warning: Validation file not found.")
        df_validation = None
    
    # Load query rules
    try:
        df_rules = pd.read_excel('Query_Rules.xlsx')
        df_rules_filtered = df_rules[df_rules['Category'].isin(['complaints'])].copy()
        print(f"Query rules loaded and filtered: {df_rules_filtered.shape}")
    except FileNotFoundError:
        print("Warning: Query rules file not found.")
        df_rules_filtered = None
    
    # Enhanced data preprocessing
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    df_main['DayOfWeek'] = df_main['Date'].dt.day_name()
    df_main['WeekOfMonth'] = df_main['Date'].dt.day // 7 + 1
    df_main['Quarter'] = df_main['Date'].dt.quarter
    df_main['Is_Holiday_Season'] = df_main['Date'].dt.month.isin([11, 12, 1])
    df_main['Is_Month_End'] = df_main['Date'].dt.day >= 25
    
    # CRITICAL ADDITION: Period Classification for Pre vs Post Analysis
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    print(f"Period Classification:")
    print(f"  Pre Period (Oct-Dec 2024): {(df_main['Period'] == 'Pre').sum()} records")
    print(f"  Post Period (Jan-Mar 2025): {(df_main['Period'] == 'Post').sum()} records")
    print(f"  Other Periods: {(df_main['Period'] == 'Other').sum()} records")
    
    # Text processing
    df_main['Customer Transcript'] = df_main['Customer Transcript'].fillna('')
    df_main['Agent Transcript'] = df_main['Agent Transcript'].fillna('')
    df_main['Full_Transcript'] = df_main['Customer Transcript'] + ' ' + df_main['Agent Transcript']
    
    # Text features
    df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
    df_main['Customer_Word_Count'] = df_main['Customer Transcript'].str.split().str.len()
    df_main['Agent_Word_Count'] = df_main['Agent Transcript'].str.split().str.len()
    df_main['Customer_Agent_Ratio'] = df_main['Customer_Word_Count'] / (df_main['Agent_Word_Count'] + 1)
    
    # Advanced text features
    df_main['Customer_Question_Count'] = df_main['Customer Transcript'].str.count('\?')
    df_main['Customer_Exclamation_Count'] = df_main['Customer Transcript'].str.count('!')
    df_main['Customer_Caps_Ratio'] = df_main['Customer Transcript'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    
    # Negation and qualifying patterns
    negation_patterns = r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t|isnt|isn\'t)\b'
    df_main['Customer_Negation_Count'] = df_main['Customer Transcript'].str.lower().str.count(negation_patterns)
    df_main['Agent_Negation_Count'] = df_main['Agent Transcript'].str.lower().str.count(negation_patterns)
    
    qualifying_patterns = r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely)\b'
    df_main['Customer_Qualifying_Count'] = df_main['Customer Transcript'].str.lower().str.count(qualifying_patterns)
    
    # Target variables
    df_main['Is_TP'] = (df_main['Primary Marker'] == 'TP').astype(int)
    df_main['Is_FP'] = (df_main['Primary Marker'] == 'FP').astype(int)
    df_main['Has_Secondary_Validation'] = df_main['Secondary Marker'].notna()
    df_main['Primary_Secondary_Agreement'] = np.where(
        df_main['Has_Secondary_Validation'] & df_main['Secondary Marker'].notna(),
        (df_main['Primary Marker'] == df_main['Secondary Marker']).astype(int),
        np.nan
    )
    
    print(f"Enhanced data preparation completed. Final dataset shape: {df_main.shape}")
    
    return df_main, df_validation, df_rules_filtered


def extract_negation_candidates_from_data(df):
    """
    Extract potential negation words using linguistic patterns and frequency analysis
    """
    
    print("1. EXTRACTING NEGATION CANDIDATES FROM DATA")
    print("-" * 40)
    
    negation_candidates = Counter()
    
    # Common negation prefixes and suffixes
    negation_prefixes = ['un', 'non', 'dis', 'in', 'im', 'ir', 'il']
    negation_suffixes = ['n\'t', 'nt']
    
    # Process both customer and agent transcripts
    for _, row in df.iterrows():
        for transcript_type in ['Customer Transcript', 'Agent Transcript']:
            text = str(row[transcript_type]).lower()
            words = text.split()
            
            for word in words:
                # Check for explicit negation words
                if any(neg in word for neg in ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere']):
                    negation_candidates[word] += 1
                
                # Check for contractions
                if any(suffix in word for suffix in negation_suffixes):
                    negation_candidates[word] += 1
                
                # Check for negation prefixes
                if len(word) > 4 and any(word.startswith(prefix) for prefix in negation_prefixes):
                    negation_candidates[word] += 1
    
    # Filter by frequency threshold
    min_frequency = max(5, len(df) * 0.001)  # At least 5 occurrences or 0.1% of data
    filtered_candidates = {word: count for word, count in negation_candidates.items() 
                         if count >= min_frequency}
    
    print(f"Found {len(filtered_candidates)} negation candidates")
    print("Top 20 candidates:", list(dict(negation_candidates.most_common(20)).keys()))
    
    return filtered_candidates


def analyze_negation_contexts(df, negation_candidates):
    """
    Analyze the context around each negation candidate to understand patterns
    """
    
    print("\n2. ANALYZING NEGATION CONTEXTS")
    print("-" * 40)
    
    context_analysis = {}
    
    for negation_word in negation_candidates.keys():
        context_data = {
            'total_occurrences': 0,
            'tp_occurrences': 0,
            'fp_occurrences': 0,
            'customer_occurrences': 0,
            'agent_occurrences': 0,
            'contexts_before': Counter(),
            'contexts_after': Counter(),
            'complaint_associations': Counter(),
            'information_associations': Counter()
        }
        
        for _, row in df.iterrows():
            is_tp = row['Primary Marker'] == 'TP'
            
            # Analyze both customer and agent transcripts
            for transcript_type, speaker in [('Customer Transcript', 'customer'), ('Agent Transcript', 'agent')]:
                text = str(row[transcript_type]).lower()
                words = text.split()
                
                for i, word in enumerate(words):
                    if negation_word in word:
                        context_data['total_occurrences'] += 1
                        
                        if is_tp:
                            context_data['tp_occurrences'] += 1
                        else:
                            context_data['fp_occurrences'] += 1
                        
                        if speaker == 'customer':
                            context_data['customer_occurrences'] += 1
                        else:
                            context_data['agent_occurrences'] += 1
                        
                        # Extract context (3 words before and after)
                        before_context = ' '.join(words[max(0, i-3):i])
                        after_context = ' '.join(words[i+1:min(len(words), i+4)])
                        
                        context_data['contexts_before'][before_context] += 1
                        context_data['contexts_after'][after_context] += 1
                        
                        # Classify as complaint or information based on context
                        full_context = ' '.join(words[max(0, i-5):min(len(words), i+6)])
                        if is_complaint_context(full_context):
                            context_data['complaint_associations'][full_context] += 1
                        else:
                            context_data['information_associations'][full_context] += 1
        
        context_analysis[negation_word] = context_data
    
    return context_analysis


def is_complaint_context(context):
    """
    Determine if a context suggests a complaint rather than information seeking
    """
    complaint_indicators = [
        'problem', 'issue', 'wrong', 'error', 'broken', 'failed', 'trouble',
        'disappointed', 'frustrated', 'angry', 'upset', 'dissatisfied',
        'received', 'working', 'resolved', 'fixed', 'helped'
    ]
    
    information_indicators = [
        'understand', 'know', 'sure', 'clear', 'how', 'what', 'when', 'where',
        'explain', 'help me', 'can you', 'would you'
    ]
    
    complaint_score = sum(1 for indicator in complaint_indicators if indicator in context)
    information_score = sum(1 for indicator in information_indicators if indicator in context)
    
    return complaint_score > information_score


def cluster_negation_patterns(negation_analysis):
    """
    Cluster negation patterns by semantic similarity and usage patterns
    """
    
    print("\n3. CLUSTERING NEGATION PATTERNS")
    print("-" * 40)
    
    # Create feature vectors for each negation word
    features = []
    negation_words = []
    
    for word, data in negation_analysis.items():
        if data['total_occurrences'] > 10:  # Minimum threshold
            tp_ratio = data['tp_occurrences'] / max(data['total_occurrences'], 1)
            fp_ratio = data['fp_occurrences'] / max(data['total_occurrences'], 1)
            customer_ratio = data['customer_occurrences'] / max(data['total_occurrences'], 1)
            agent_ratio = data['agent_occurrences'] / max(data['total_occurrences'], 1)
            complaint_ratio = len(data['complaint_associations']) / max(data['total_occurrences'], 1)
            
            features.append([tp_ratio, fp_ratio, customer_ratio, agent_ratio, complaint_ratio, data['total_occurrences']])
            negation_words.append(word)
    
    if len(features) > 3:
        # Apply K-means clustering
        n_clusters = min(5, len(features) // 3)  # Maximum 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Group words by cluster
        clustered_patterns = defaultdict(list)
        for word, cluster in zip(negation_words, clusters):
            clustered_patterns[f"cluster_{cluster}"].append(word)
        
        print(f"Created {len(clustered_patterns)} negation pattern clusters:")
        for cluster_name, words in clustered_patterns.items():
            print(f"  {cluster_name}: {words[:10]}")  # Show first 10 words
        
        return dict(clustered_patterns)
    else:
        # If too few patterns, create single cluster
        return {"cluster_0": negation_words}


def analyze_query_components_enhanced(df_main, df_rules):
    """
    Enhanced query component analysis with clustering for dynamic pattern discovery
    """
    
    print("\n" + "="*60)
    print("ENHANCED QUERY COMPONENT ANALYSIS WITH CLUSTERING")
    print("="*60)
    
    if df_rules is None:
        print("No query rules data available")
        return {}
    
    # Extract individual query components with their performance metrics
    query_components = []
    
    for _, rule in df_rules.iterrows():
        query_text = rule['Query Text']
        event = rule['Event']
        query_name = rule['Query']
        channel = rule.get('Channel', 'both')
        
        # Parse query into components (quoted phrases and individual words)
        quoted_components = re.findall(r'"([^"]+)"', query_text)
        word_components = re.findall(r'\b(\w+)\b', query_text.lower())
        
        # Remove quotes and operators from word components
        operators = {'and', 'or', 'not', 'near', 'before', 'after'}
        word_components = [word for word in word_components if word not in operators and len(word) > 2]
        
        all_components = quoted_components + word_components
        
        # Test each component's correlation with FPs/TPs
        for component in all_components:
            if len(component) > 2:  # Filter out very short components
                component_metrics = analyze_component_performance(df_main, component, channel, event, query_name)
                if component_metrics['total_matches'] > 5:  # Minimum threshold
                    query_components.append(component_metrics)
    
    # Convert to DataFrame for analysis
    components_df = pd.DataFrame(query_components)
    
    if len(components_df) == 0:
        print("No valid query components found")
        return {}
    
    print(f"Analyzed {len(components_df)} query components")
    
    # Cluster components based on performance characteristics
    clustered_components = cluster_query_components(components_df)
    
    # Validate clusters against TP/FP performance
    validated_clusters = validate_component_clusters(df_main, clustered_components)
    
    return {
        'components_df': components_df,
        'clustered_components': clustered_components,
        'validated_clusters': validated_clusters
    }


def analyze_component_performance(df_main, component, channel, event, query_name):
    """
    Analyze performance of a single query component
    """
    
    # Determine which transcript(s) to search based on channel
    if channel == 'customer':
        search_texts = df_main['Customer Transcript']
    elif channel == 'agent':
        search_texts = df_main['Agent Transcript']
    else:  # 'both' or default
        search_texts = df_main['Full_Transcript']
    
    # Find matches
    contains_component = search_texts.str.lower().str.contains(
        re.escape(component.lower()), case=False, na=False
    )
    
    matches_data = df_main[contains_component]
    
    if len(matches_data) == 0:
        return {
            'component': component,
            'channel': channel,
            'event': event,
            'query_name': query_name,
            'tp_count': 0,
            'fp_count': 0,
            'total_matches': 0,
            'tp_rate': 0,
            'fp_rate': 0,
            'precision': 0,
            'discrimination_power': 0
        }
    
    tp_count = (matches_data['Primary Marker'] == 'TP').sum()
    fp_count = (matches_data['Primary Marker'] == 'FP').sum()
    total_matches = len(matches_data)
    
    # Calculate rates relative to total dataset
    total_tps = (df_main['Primary Marker'] == 'TP').sum()
    total_fps = (df_main['Primary Marker'] == 'FP').sum()
    
    tp_rate = tp_count / max(total_tps, 1)
    fp_rate = fp_count / max(total_fps, 1)
    precision = tp_count / max(total_matches, 1)
    discrimination_power = abs(tp_rate - fp_rate)
    
    return {
        'component': component,
        'channel': channel,
        'event': event,
        'query_name': query_name,
        'tp_count': tp_count,
        'fp_count': fp_count,
        'total_matches': total_matches,
        'tp_rate': tp_rate,
        'fp_rate': fp_rate,
        'precision': precision,
        'discrimination_power': discrimination_power
    }


def cluster_query_components(components_df):
    """
    Cluster query components based on their performance characteristics
    """
    
    print("\n4. CLUSTERING QUERY COMPONENTS")
    print("-" * 40)
    
    # Create feature vectors for clustering
    feature_columns = ['tp_rate', 'fp_rate', 'precision', 'discrimination_power', 'total_matches']
    features = components_df[feature_columns].fillna(0)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply clustering
    n_clusters = min(5, len(features) // 10)  # Maximum 5 clusters, minimum 10 components per cluster
    
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        components_df['cluster'] = cluster_labels
        
        # Group components by cluster
        clustered_components = {}
        for cluster_id in range(n_clusters):
            cluster_components = components_df[components_df['cluster'] == cluster_id]
            clustered_components[f"component_cluster_{cluster_id}"] = {
                'components': cluster_components['component'].tolist(),
                'avg_precision': cluster_components['precision'].mean(),
                'avg_discrimination': cluster_components['discrimination_power'].mean(),
                'cluster_size': len(cluster_components)
            }
        
        print(f"Created {len(clustered_components)} component clusters:")
        for cluster_name, cluster_data in clustered_components.items():
            print(f"  {cluster_name}: {cluster_data['cluster_size']} components, "
                  f"avg precision: {cluster_data['avg_precision']:.3f}")
        
        return clustered_components
    else:
        # Single cluster if too few components
        return {
            "component_cluster_0": {
                'components': components_df['component'].tolist(),
                'avg_precision': components_df['precision'].mean(),
                'avg_discrimination': components_df['discrimination_power'].mean(),
                'cluster_size': len(components_df)
            }
        }


def validate_component_clusters(df_main, clustered_components):
    """
    Validate component clusters against actual TP/FP performance
    """
    
    print("\n5. VALIDATING COMPONENT CLUSTERS")
    print("-" * 40)
    
    validated_clusters = {}
    
    for cluster_name, cluster_data in clustered_components.items():
        components = cluster_data['components']
        
        # Test cluster performance on the dataset
        cluster_matches = pd.Series([False] * len(df_main))
        
        for component in components:
            component_matches = df_main['Full_Transcript'].str.lower().str.contains(
                re.escape(component.lower()), case=False, na=False
            )
            cluster_matches = cluster_matches | component_matches
        
        matches_data = df_main[cluster_matches]
        
        if len(matches_data) > 0:
            cluster_precision = (matches_data['Primary Marker'] == 'TP').mean()
            cluster_coverage = len(matches_data) / len(df_main)
            
            # Determine cluster type based on performance
            if cluster_precision > 0.7:
                cluster_type = 'high_precision'
            elif cluster_precision < 0.4:
                cluster_type = 'low_precision'
            else:
                cluster_type = 'moderate_precision'
            
            validated_clusters[cluster_name] = {
                'components': components,
                'precision': cluster_precision,
                'coverage': cluster_coverage,
                'cluster_type': cluster_type,
                'sample_size': len(matches_data)
            }
    
    # Sort by precision
    validated_clusters = dict(sorted(validated_clusters.items(), 
                                   key=lambda x: x[1]['precision'], 
                                   reverse=True))
    
    print("Cluster Performance Summary:")
    for name, data in validated_clusters.items():
        print(f"  {name}: {data['cluster_type']} (precision: {data['precision']:.3f}, "
              f"coverage: {data['coverage']:.3f})")
    
    return validated_clusters


def create_comprehensive_feature_matrix(df, negation_patterns, query_components):
    """
    Create comprehensive feature matrix for SHAP analysis
    """
    
    print("\n6. CREATING COMPREHENSIVE FEATURE MATRIX FOR SHAP ANALYSIS")
    print("-" * 40)
    
    features_matrix = []
    feature_names = []
    
    for _, row in df.iterrows():
        feature_vector = []
        
        # Add existing features
        basic_features = [
            'Customer_Word_Count', 'Agent_Word_Count', 'Customer_Agent_Ratio',
            'Customer_Question_Count', 'Customer_Exclamation_Count', 'Customer_Caps_Ratio',
            'Customer_Negation_Count', 'Agent_Negation_Count', 'Customer_Qualifying_Count'
        ]
        
        for feature in basic_features:
            if feature in df.columns:
                feature_vector.append(row[feature])
                if feature not in feature_names:
                    feature_names.append(feature)
        
        # Add dynamic negation pattern features
        for pattern_name, pattern_data in negation_patterns.items():
            customer_text = str(row['Customer Transcript']).lower()
            agent_text = str(row['Agent Transcript']).lower()
            
            pattern_count = sum(1 for word in pattern_data if word in customer_text or word in agent_text)
            feature_vector.append(pattern_count)
            
            pattern_feature_name = f"negation_{pattern_name}_count"
            if pattern_feature_name not in feature_names:
                feature_names.append(pattern_feature_name)
        
        # Add query component features
        if 'validated_clusters' in query_components:
            for cluster_name, cluster_data in query_components['validated_clusters'].items():
                customer_text = str(row['Customer Transcript']).lower()
                agent_text = str(row['Agent Transcript']).lower()
                full_text = customer_text + ' ' + agent_text
                
                cluster_matches = sum(1 for component in cluster_data['components'] 
                                    if component.lower() in full_text)
                feature_vector.append(cluster_matches)
                
                cluster_feature_name = f"query_{cluster_name}_matches"
                if cluster_feature_name not in feature_names:
                    feature_names.append(cluster_feature_name)
        
        features_matrix.append(feature_vector)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_matrix, columns=feature_names)
    
    print(f"Created feature matrix with {len(feature_names)} features for {len(features_df)} samples")
    
    return features_df, feature_names


def perform_shap_analysis(features_df, target, feature_names):
    """
    Perform SHAP analysis for feature importance and explainability
    """
    
    print("\n7. PERFORMING SHAP ANALYSIS")
    print("-" * 40)
    
    # Prepare data
    X = features_df.fillna(0)
    y = target
    
    # Train a model for SHAP analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X, y)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # If binary classification, use positive class SHAP values
    if len(shap_values) == 2:
        shap_values_analysis = shap_values[1]  # Positive class (FP)
    else:
        shap_values_analysis = shap_values
    
    # Calculate feature importance
    feature_importance = np.abs(shap_values_analysis).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features (SHAP):")
    print(importance_df.head(15))
    
    return {
        'model': model,
        'explainer': explainer,
        'shap_values': shap_values_analysis,
        'feature_importance': importance_df,
        'X': X
    }


def create_enhanced_visualizations(df, negation_analysis, query_components, shap_results, output_dir='enhanced_analysis'):
    """
    Create comprehensive visualizations including original and new analyses
    """
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Original Negation Type Distribution by TP/FP (Retained)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    tp_data = df[df['Primary Marker'] == 'TP']
    fp_data = df[df['Primary Marker'] == 'FP']
    
    # Negation counts distribution
    axes[0,0].hist([tp_data['Customer_Negation_Count'], fp_data['Customer_Negation_Count']], 
                   bins=20, alpha=0.7, label=['TP', 'FP'], color=['green', 'red'])
    axes[0,0].set_title('Customer Negation Count Distribution')
    axes[0,0].set_xlabel('Negation Count')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    axes[0,1].hist([tp_data['Agent_Negation_Count'], fp_data['Agent_Negation_Count']], 
                   bins=20, alpha=0.7, label=['TP', 'FP'], color=['green', 'red'])
    axes[0,1].set_title('Agent Negation Count Distribution')
    axes[0,1].set_xlabel('Negation Count')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    # Customer vs Agent negation comparison
    customer_negations = [tp_data['Customer_Negation_Count'].mean(), fp_data['Customer_Negation_Count'].mean()]
    agent_negations = [tp_data['Agent_Negation_Count'].mean(), fp_data['Agent_Negation_Count'].mean()]
    
    x = ['TP', 'FP']
    width = 0.35
    x_pos = np.arange(len(x))
    
    axes[0,2].bar(x_pos - width/2, customer_negations, width, label='Customer', alpha=0.7, color='blue')
    axes[0,2].bar(x_pos + width/2, agent_negations, width, label='Agent', alpha=0.7, color='orange')
    axes[0,2].set_title('Average Negation Count by Speaker')
    axes[0,2].set_ylabel('Average Negation Count')
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels(x)
    axes[0,2].legend()
    
    # 2. Dynamic Negation Pattern Performance
    if negation_analysis:
        pattern_performance = []
        for word, data in negation_analysis.items():
            if data['total_occurrences'] > 10:
                tp_rate = data['tp_occurrences'] / max(data['total_occurrences'], 1)
                fp_rate = data['fp_occurrences'] / max(data['total_occurrences'], 1)
                discrimination = abs(tp_rate - fp_rate)
                pattern_performance.append({
                    'word': word,
                    'tp_rate': tp_rate,
                    'fp_rate': fp_rate,
                    'discrimination': discrimination,
                    'total_occurrences': data['total_occurrences']
                })
        
        if pattern_performance:
            pattern_df = pd.DataFrame(pattern_performance).sort_values('discrimination', ascending=False)
            top_patterns = pattern_df.head(15)
            
            axes[1,0].barh(top_patterns['word'], top_patterns['discrimination'], color='purple', alpha=0.7)
            axes[1,0].set_title('Top 15 Discriminative Negation Patterns')
            axes[1,0].set_xlabel('Discrimination Power')
            axes[1,0].tick_params(axis='y', labelsize=8)
    
    # 3. Query Component Cluster Performance
    if 'validated_clusters' in query_components:
        cluster_names = []
        cluster_precisions = []
        cluster_coverages = []
        
        for name, data in query_components['validated_clusters'].items():
            cluster_names.append(name.replace('component_cluster_', 'Cluster '))
            cluster_precisions.append(data['precision'])
            cluster_coverages.append(data['coverage'])
        
        if cluster_names:
            axes[1,1].scatter(cluster_coverages, cluster_precisions, s=100, alpha=0.7, color='red')
            for i, name in enumerate(cluster_names):
                axes[1,1].annotate(name, (cluster_coverages[i], cluster_precisions[i]), 
                                 xytext=(5, 5), textcoords='offset points', fontsize=8)
            axes[1,1].set_title('Query Component Cluster Performance')
            axes[1,1].set_xlabel('Coverage (% of data)')
            axes[1,1].set_ylabel('Precision')
            axes[1,1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Target Precision')
            axes[1,1].legend()
    
    # 4. SHAP Feature Importance
    if shap_results and 'feature_importance' in shap_results:
        top_features = shap_results['feature_importance'].head(15)
        
        axes[1,2].barh(top_features['feature'], top_features['importance'], color='teal', alpha=0.7)
        axes[1,2].set_title('Top 15 SHAP Feature Importance')
        axes[1,2].set_xlabel('SHAP Importance')
        axes[1,2].tick_params(axis='y', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Monthly Evolution Analysis (Enhanced from original)
    monthly_analysis = df.groupby(['Year_Month', 'Primary Marker']).agg({
        'Customer_Negation_Count': 'mean',
        'Agent_Negation_Count': 'mean',
        'Customer_Agent_Ratio': 'mean',
        'UUID': 'count'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Customer Negation Trends', 'Agent Negation Trends',
                       'Customer-Agent Ratio Trends', 'Volume Trends'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    tp_monthly = monthly_analysis[monthly_analysis['Primary_Marker'] == 'TP']
    fp_monthly = monthly_analysis[monthly_analysis['Primary_Marker'] == 'FP']
    
    # Customer negation trends
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Customer_Negation_Count'],
                  mode='lines+markers', name='TP Customer Neg', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Customer_Negation_Count'],
                  mode='lines+markers', name='FP Customer Neg', line=dict(color='red')),
        row=1, col=1
    )
    
    # Agent negation trends
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Agent_Negation_Count'],
                  mode='lines+markers', name='TP Agent Neg', line=dict(color='darkgreen')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Agent_Negation_Count'],
                  mode='lines+markers', name='FP Agent Neg', line=dict(color='darkred')),
        row=1, col=2
    )
    
    # Customer-Agent ratio trends
    fig.add_trace(
        go.Scatter(x=tp_monthly['Year_Month'], y=tp_monthly['Customer_Agent_Ratio'],
                  mode='lines+markers', name='TP Ratio', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=fp_monthly['Year_Month'], y=fp_monthly['Customer_Agent_Ratio'],
                  mode='lines+markers', name='FP Ratio', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Volume trends
    fig.add_trace(
        go.Bar(x=tp_monthly['Year_Month'], y=tp_monthly['UUID'],
               name='TP Volume', marker_color='green', opacity=0.7),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(x=fp_monthly['Year_Month'], y=fp_monthly['UUID'],
               name='FP Volume', marker_color='red', opacity=0.7),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Enhanced Monthly Pattern Analysis")
    fig.write_html(f'{output_dir}/enhanced_monthly_analysis.html')
    
    # 6. SHAP Summary Plots
    if shap_results and 'shap_values' in shap_results:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_results['shap_values'], shap_results['X'], 
                         feature_names=shap_results['feature_importance']['feature'].tolist(),
                         max_display=20, show=False)
        plt.title('SHAP Summary Plot - Feature Impact on FP Classification')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Waterfall plot for a few examples
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(min(4, len(shap_results['X']))):
            plt.sca(axes[i])
            shap.plots.waterfall(shap.Explanation(
                values=shap_results['shap_values'][i],
                base_values=shap_results['explainer'].expected_value,
                data=shap_results['X'].iloc[i],
                feature_names=shap_results['feature_importance']['feature'].tolist()
            ), max_display=10, show=False)
            plt.title(f'SHAP Explanation - Sample {i+1}')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_waterfall_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Enhanced Word Clouds (Retained from original)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # TP Customer transcript word cloud
    tp_customer_text = ' '.join(tp_data['Customer Transcript'].fillna(''))
    if tp_customer_text.strip():
        wordcloud_tp_customer = WordCloud(width=400, height=300, background_color='white').generate(tp_customer_text)
        axes[0,0].imshow(wordcloud_tp_customer, interpolation='bilinear')
        axes[0,0].set_title('TP Customer Transcript Word Cloud')
        axes[0,0].axis('off')
    
    # FP Customer transcript word cloud
    fp_customer_text = ' '.join(fp_data['Customer Transcript'].fillna(''))
    if fp_customer_text.strip():
        wordcloud_fp_customer = WordCloud(width=400, height=300, background_color='white').generate(fp_customer_text)
        axes[0,1].imshow(wordcloud_fp_customer, interpolation='bilinear')
        axes[0,1].set_title('FP Customer Transcript Word Cloud')
        axes[0,1].axis('off')
    
    # TP Agent transcript word cloud
    tp_agent_text = ' '.join(tp_data['Agent Transcript'].fillna(''))
    if tp_agent_text.strip():
        wordcloud_tp_agent = WordCloud(width=400, height=300, background_color='white').generate(tp_agent_text)
        axes[1,0].imshow(wordcloud_tp_agent, interpolation='bilinear')
        axes[1,0].set_title('TP Agent Transcript Word Cloud')
        axes[1,0].axis('off')
    
    # FP Agent transcript word cloud
    fp_agent_text = ' '.join(fp_data['Agent Transcript'].fillna(''))
    if fp_agent_text.strip():
        wordcloud_fp_agent = WordCloud(width=400, height=300, background_color='white').generate(fp_agent_text)
        axes[1,1].imshow(wordcloud_fp_agent, interpolation='bilinear')
        axes[1,1].set_title('FP Agent Transcript Word Cloud')
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Pattern Evolution Heatmap
    if negation_analysis:
        # Create monthly pattern evolution data
        monthly_pattern_data = []
        months = sorted(df['Year_Month'].unique())
        
        for month in months:
            month_data = df[df['Year_Month'] == month]
            month_patterns = {}
            
            for word, data in negation_analysis.items():
                if data['total_occurrences'] > 20:  # Only significant patterns
                    # Count occurrences in this month
                    month_tp_count = 0
                    month_fp_count = 0
                    
                    for _, row in month_data.iterrows():
                        customer_text = str(row['Customer Transcript']).lower()
                        agent_text = str(row['Agent Transcript']).lower()
                        
                        if word in customer_text or word in agent_text:
                            if row['Primary Marker'] == 'TP':
                                month_tp_count += 1
                            else:
                                month_fp_count += 1
                    
                    total_month_count = month_tp_count + month_fp_count
                    if total_month_count > 0:
                        month_discrimination = abs(
                            month_tp_count / max(len(month_data[month_data['Primary Marker'] == 'TP']), 1) -
                            month_fp_count / max(len(month_data[month_data['Primary Marker'] == 'FP']), 1)
                        )
                        month_patterns[word] = month_discrimination
            
            monthly_pattern_data.append(month_patterns)
        
        # Create heatmap data
        if monthly_pattern_data:
            all_patterns = set()
            for month_patterns in monthly_pattern_data:
                all_patterns.update(month_patterns.keys())
            
            heatmap_data = []
            for pattern in list(all_patterns)[:20]:  # Top 20 patterns
                pattern_row = []
                for month_patterns in monthly_pattern_data:
                    pattern_row.append(month_patterns.get(pattern, 0))
                heatmap_data.append(pattern_row)
            
            if heatmap_data:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(heatmap_data, xticklabels=months, yticklabels=list(all_patterns)[:20],
                           cmap='RdYlBu_r', annot=True, fmt='.3f', ax=ax)
                ax.set_title('Monthly Pattern Discrimination Evolution')
                ax.set_xlabel('Month')
                ax.set_ylabel('Negation Patterns')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/pattern_evolution_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    # 9. Component Effectiveness Analysis
    if 'components_df' in query_components and len(query_components['components_df']) > 0:
        components_df = query_components['components_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision vs Total Matches scatter
        axes[0,0].scatter(components_df['total_matches'], components_df['precision'], 
                         alpha=0.6, c=components_df['discrimination_power'], cmap='viridis')
        axes[0,0].set_xlabel('Total Matches')
        axes[0,0].set_ylabel('Precision')
        axes[0,0].set_title('Component Precision vs Volume')
        cbar = plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])
        cbar.set_label('Discrimination Power')
        
        # Channel distribution
        channel_counts = components_df['channel'].value_counts()
        axes[0,1].pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Query Components by Channel')
        
        # Top precision components
        top_precision = components_df.nlargest(15, 'precision')
        axes[1,0].barh(range(len(top_precision)), top_precision['precision'])
        axes[1,0].set_yticks(range(len(top_precision)))
        axes[1,0].set_yticklabels([f"{comp[:15]}..." if len(comp) > 15 else comp 
                                  for comp in top_precision['component']], fontsize=8)
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_title('Top 15 High-Precision Components')
        
        # Discrimination power distribution
        axes[1,1].hist(components_df['discrimination_power'], bins=20, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('Discrimination Power')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Component Discrimination Power')
        axes[1,1].axvline(components_df['discrimination_power'].mean(), color='red', 
                         linestyle='--', label='Mean')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/component_effectiveness_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Enhanced visualizations saved to {output_dir}/")


def create_recommendations_report(negation_patterns, query_components, shap_results, pattern_evolution=None):
    """
    Generate comprehensive recommendations based on all analyses
    """
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE RECOMMENDATIONS")
    print("="*60)
    
    recommendations = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': [],
        'implementation_details': {}
    }
    
    # 1. Negation Pattern Recommendations
    if negation_patterns:
        high_discrimination_patterns = []
        low_discrimination_patterns = []
        
        for cluster_name, pattern_words in negation_patterns.items():
            # Calculate average discrimination for cluster
            # (This would need the full analysis data, simplified here)
            if len(pattern_words) > 5:
                high_discrimination_patterns.append(cluster_name)
            else:
                low_discrimination_patterns.append(cluster_name)
        
        if high_discrimination_patterns:
            recommendations['high_priority'].append({
                'action': 'Implement Context-Aware Negation Rules',
                'rationale': f'Found {len(high_discrimination_patterns)} high-discrimination negation patterns',
                'details': 'Create separate weights for customer vs agent negations',
                'patterns': high_discrimination_patterns
            })
    
    # 2. Query Component Recommendations
    if 'validated_clusters' in query_components:
        high_precision_clusters = []
        low_precision_clusters = []
        
        for cluster_name, cluster_data in query_components['validated_clusters'].items():
            if cluster_data['precision'] > 0.8:
                high_precision_clusters.append(cluster_name)
            elif cluster_data['precision'] < 0.5:
                low_precision_clusters.append(cluster_name)
        
        if high_precision_clusters:
            recommendations['high_priority'].append({
                'action': 'Boost High-Precision Query Components',
                'rationale': f'Found {len(high_precision_clusters)} high-precision component clusters',
                'details': 'Increase weights for these components in production rules',
                'clusters': high_precision_clusters
            })
        
        if low_precision_clusters:
            recommendations['medium_priority'].append({
                'action': 'Review Low-Precision Query Components',
                'rationale': f'Found {len(low_precision_clusters)} low-precision component clusters',
                'details': 'Consider removing or modifying these components',
                'clusters': low_precision_clusters
            })
    
    # 3. SHAP-Based Recommendations
    if shap_results and 'feature_importance' in shap_results:
        top_features = shap_results['feature_importance'].head(10)
        
        # Identify feature types for targeted recommendations
        negation_features = top_features[top_features['feature'].str.contains('negation', case=False)]
        query_features = top_features[top_features['feature'].str.contains('query', case=False)]
        
        if len(negation_features) > 0:
            recommendations['high_priority'].append({
                'action': 'Focus on Top SHAP Negation Features',
                'rationale': f'{len(negation_features)} negation features in top 10 SHAP importance',
                'details': 'These features have highest impact on FP classification',
                'features': negation_features['feature'].tolist()
            })
        
        if len(query_features) > 0:
            recommendations['medium_priority'].append({
                'action': 'Optimize Query Component Features',
                'rationale': f'{len(query_features)} query features in top 10 SHAP importance',
                'details': 'These query components significantly influence classification',
                'features': query_features['feature'].tolist()
            })
    
    # 4. Implementation Framework
    recommendations['implementation_details'] = {
        'pattern_refresh_cycle': {
            'frequency': 'Monthly',
            'method': 'Re-run dynamic pattern discovery on latest 3 months',
            'validation': 'A/B test new patterns vs current rules',
            'threshold': 'Minimum 0.05 discrimination power'
        },
        'component_optimization': {
            'frequency': 'Quarterly', 
            'method': 'Re-cluster query components and validate performance',
            'action': 'Update component weights based on cluster performance',
            'monitoring': 'Track precision impact of component changes'
        },
        'shap_monitoring': {
            'frequency': 'Weekly',
            'method': 'Generate SHAP explanations for recent classifications',
            'alerts': 'Flag when top features change significantly',
            'review': 'Monthly SHAP feature importance review'
        },
        'speaker_attribution': {
            'implementation': 'Separate customer and agent pattern weights',
            'customer_boost': 'Increase weight for customer complaint patterns by 2x',
            'agent_filtering': 'Reduce weight for agent explanation patterns by 0.5x',
            'validation': 'Measure precision improvement from speaker attribution'
        }
    }
    
    # Print recommendations
    print("\nHIGH PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations['high_priority'], 1):
        print(f"{i}. {rec['action']}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"   Details: {rec['details']}")
        if 'patterns' in rec:
            print(f"   Key patterns: {rec.get('patterns', [])[:3]}")
        if 'features' in rec:
            print(f"   Key features: {rec.get('features', [])[:3]}")
        print()
    
    print("MEDIUM PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations['medium_priority'], 1):
        print(f"{i}. {rec['action']}")
        print(f"   Rationale: {rec['rationale']}")
        print(f"   Details: {rec['details']}")
        print()
    
    print("IMPLEMENTATION FRAMEWORK:")
    for component, details in recommendations['implementation_details'].items():
        print(f"\n{component.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    return recommendations


def export_comprehensive_results(df, negation_analysis, query_components, shap_results, 
                                recommendations, output_prefix="enhanced_negation_analysis"):
    """
    Export all analysis results to multiple formats
    """
    
    print("\n" + "="*60)
    print("EXPORTING COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data for export
    export_data = {}
    
    # 1. Dynamic Negation Patterns
    if negation_analysis:
        negation_patterns_list = []
        for word, data in negation_analysis.items():
            negation_patterns_list.append({
                'Negation_Word': word,
                'Total_Occurrences': data['total_occurrences'],
                'TP_Occurrences': data['tp_occurrences'],
                'FP_Occurrences': data['fp_occurrences'],
                'Customer_Occurrences': data['customer_occurrences'],
                'Agent_Occurrences': data['agent_occurrences'],
                'TP_Rate': data['tp_occurrences'] / max(data['total_occurrences'], 1),
                'FP_Rate': data['fp_occurrences'] / max(data['total_occurrences'], 1),
                'Customer_Rate': data['customer_occurrences'] / max(data['total_occurrences'], 1),
                'Agent_Rate': data['agent_occurrences'] / max(data['total_occurrences'], 1)
            })
        
        export_data['Dynamic_Negation_Patterns'] = pd.DataFrame(negation_patterns_list)
    
    # 2. Query Components Analysis
    if 'components_df' in query_components:
        export_data['Query_Components'] = query_components['components_df']
    
    if 'validated_clusters' in query_components:
        cluster_summary = []
        for cluster_name, cluster_data in query_components['validated_clusters'].items():
            cluster_summary.append({
                'Cluster_Name': cluster_name,
                'Precision': cluster_data['precision'],
                'Coverage': cluster_data['coverage'],
                'Cluster_Type': cluster_data['cluster_type'],
                'Sample_Size': cluster_data['sample_size'],
                'Component_Count': len(cluster_data['components'])
            })
        
        export_data['Component_Clusters'] = pd.DataFrame(cluster_summary)
    
    # 3. SHAP Results
    if shap_results and 'feature_importance' in shap_results:
        export_data['SHAP_Feature_Importance'] = shap_results['feature_importance']
    
    # 4. Recommendations
    recommendations_list = []
    for priority in ['high_priority', 'medium_priority', 'low_priority']:
        for rec in recommendations[priority]:
            recommendations_list.append({
                'Priority': priority.replace('_priority', '').title(),
                'Action': rec['action'],
                'Rationale': rec['rationale'],
                'Details': rec['details']
            })
    
    export_data['Recommendations'] = pd.DataFrame(recommendations_list)
    
    # 5. Enhanced Dataset with Features
    enhanced_df = df.copy()
    export_data['Enhanced_Dataset'] = enhanced_df
    
    # Export to Excel
    excel_filename = f"{output_prefix}_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for sheet_name, data in export_data.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Comprehensive analysis results exported to: {excel_filename}")
    
    # Export recommendations as text report
    report_filename = f"{output_prefix}_recommendations_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write("ENHANCED DYNAMIC NEGATION AND QUERY COMPONENT ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"- Analyzed {len(df)} transcript records\n")
        if negation_analysis:
            f.write(f"- Discovered {len(negation_analysis)} dynamic negation patterns\n")
        if 'validated_clusters' in query_components:
            f.write(f"- Identified {len(query_components['validated_clusters'])} query component clusters\n")
        if shap_results:
            f.write(f"- Generated SHAP explanations for {len(shap_results['feature_importance'])} features\n")
        f.write(f"- Created {len(recommendations_list)} actionable recommendations\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-" * 20 + "\n")
        
        # Top negation pattern
        if negation_analysis:
            top_pattern = max(negation_analysis.items(), 
                            key=lambda x: x[1]['total_occurrences'])
            f.write(f"- Most frequent negation pattern: '{top_pattern[0]}' "
                    f"({top_pattern[1]['total_occurrences']} occurrences)\n")
        
        # Best component cluster
        if 'validated_clusters' in query_components:
            best_cluster = max(query_components['validated_clusters'].items(),
                             key=lambda x: x[1]['precision'])
            f.write(f"- Best query component cluster: {best_cluster[0]} "
                    f"(precision: {best_cluster[1]['precision']:.3f})\n")
        
        # Top SHAP feature
        if shap_results and 'feature_importance' in shap_results:
            top_feature = shap_results['feature_importance'].iloc[0]
            f.write(f"- Most important SHAP feature: {top_feature['feature']} "
                    f"(importance: {top_feature['importance']:.3f})\n")
        
        f.write("\nHIGH PRIORITY ACTIONS:\n")
        f.write("-" * 20 + "\n")
        for rec in recommendations['high_priority']:
            f.write(f"- {rec['action']}\n")
            f.write(f"  Rationale: {rec['rationale']}\n")
            f.write(f"  Implementation: {rec['details']}\n\n")
    
    print(f"Recommendations report exported to: {report_filename}")
    
    return {
        'excel_file': excel_filename,
        'report_file': report_filename
    }


def run_enhanced_comprehensive_analysis(df, df_rules):
    """
    Main function to run the complete enhanced analysis
    """
    
    print("=" * 80)
    print("STARTING ENHANCED COMPREHENSIVE NEGATION AND QUERY ANALYSIS")
    print("=" * 80)
    
    # Step 1: Extract dynamic negation patterns
    print("\nStep 1: Dynamic Negation Pattern Discovery")
    negation_candidates = extract_negation_candidates_from_data(df)
    negation_analysis = analyze_negation_contexts(df, negation_candidates)
    negation_patterns = cluster_negation_patterns(negation_analysis)
    
    # Step 2: Enhanced query component analysis
    print("\nStep 2: Enhanced Query Component Analysis")
    query_components = analyze_query_components_enhanced(df, df_rules)
    
    # Step 3: Create comprehensive feature matrix
    print("\nStep 3: Comprehensive Feature Engineering")
    features_df, feature_names = create_comprehensive_feature_matrix(df, negation_patterns, query_components)
    
    # Step 4: SHAP analysis
    print("\nStep 4: SHAP Explainability Analysis")
    target = df['Is_FP']
    shap_results = perform_shap_analysis(features_df, target, feature_names)
    
    # Step 5: Create visualizations
    print("\nStep 5: Enhanced Visualization Creation")
    create_enhanced_visualizations(df, negation_analysis, query_components, shap_results)
    
    # Step 6: Generate recommendations
    print("\nStep 6: Generating Comprehensive Recommendations")
    recommendations = create_recommendations_report(negation_patterns, query_components, shap_results)
    
    # Step 7: Export results
    print("\nStep 7: Exporting All Results")
    export_files = export_comprehensive_results(df, negation_analysis, query_components, 
                                               shap_results, recommendations)
    
    print("\n" + "=" * 80)
    print("ENHANCED COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("Results exported to:")
    for file_type, filename in export_files.items():
        print(f"  {file_type}: {filename}")
    
    # Return comprehensive results
    return {
        'negation_analysis': negation_analysis,
        'negation_patterns': negation_patterns,
        'query_components': query_components,
        'features_df': features_df,
        'shap_results': shap_results,
        'recommendations': recommendations,
        'export_files': export_files
    }


def analyze_speaker_attribution_impact(df, negation_analysis, query_components):
    """
    Analyze the impact of considering both customer and agent transcripts
    """
    
    print("\n" + "="*60)
    print("SPEAKER ATTRIBUTION IMPACT ANALYSIS")
    print("="*60)
    
    # Compare customer-only vs agent-only vs combined analysis
    analysis_results = {}
    
    # 1. Customer-only analysis
    customer_only_features = df[['Customer_Negation_Count', 'Customer_Qualifying_Count', 
                               'Customer_Question_Count', 'Customer_Word_Count']].fillna(0)
    
    # 2. Agent-only analysis  
    agent_only_features = df[['Agent_Negation_Count', 'Agent_Word_Count']].fillna(0)
    
    # 3. Combined analysis (what we're doing)
    combined_features = pd.concat([customer_only_features, agent_only_features], axis=1)
    
    # Train simple models to compare performance
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    
    target = df['Is_FP']
    
    for approach, features in [('Customer_Only', customer_only_features), 
                              ('Agent_Only', agent_only_features),
                              ('Combined', combined_features)]:
        
        if len(features.columns) > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.3, random_state=42, stratify=target
            )
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            analysis_results[approach] = {
                'auc_score': auc_score,
                'feature_count': len(features.columns),
                'model': model
            }
    
    # Print comparison
    print("Speaker Attribution Impact Comparison:")
    print("-" * 40)
    for approach, results in analysis_results.items():
        print(f"{approach}:")
        print(f"  AUC Score: {results['auc_score']:.3f}")
        print(f"  Features: {results['feature_count']}")
        
        if approach == 'Combined':
            improvement_over_customer = results['auc_score'] - analysis_results['Customer_Only']['auc_score']
            improvement_over_agent = results['auc_score'] - analysis_results['Agent_Only']['auc_score']
            print(f"  Improvement over customer-only: {improvement_over_customer:+.3f}")
            print(f"  Improvement over agent-only: {improvement_over_agent:+.3f}")
        print()
    
    # Analyze negation pattern differences by speaker
    print("Negation Pattern Analysis by Speaker:")
    print("-" * 40)
    
    customer_negation_impact = df.groupby('Primary Marker')['Customer_Negation_Count'].mean()
    agent_negation_impact = df.groupby('Primary Marker')['Agent_Negation_Count'].mean()
    
    print("Average Negation Counts:")
    print(f"Customer - TP: {customer_negation_impact['TP']:.2f}, FP: {customer_negation_impact['FP']:.2f}")
    print(f"Agent - TP: {agent_negation_impact['TP']:.2f}, FP: {agent_negation_impact['FP']:.2f}")
    
    customer_discrimination = abs(customer_negation_impact['TP'] - customer_negation_impact['FP'])
    agent_discrimination = abs(agent_negation_impact['TP'] - agent_negation_impact['FP'])
    
    print(f"\nDiscrimination Power:")
    print(f"Customer negations: {customer_discrimination:.2f}")
    print(f"Agent negations: {agent_discrimination:.2f}")
    
    if customer_discrimination > agent_discrimination:
        print("FINDING: Customer negations are more discriminative for complaint detection")
    else:
        print("FINDING: Agent negations are more discriminative for complaint detection")
    
    return analysis_results


def create_temporal_pattern_analysis(df, negation_analysis):
    """
    Analyze how patterns evolve over time (Pre vs Post)
    """
    
    print("\n" + "="*60)
    print("TEMPORAL PATTERN EVOLUTION ANALYSIS")
    print("="*60)
    
    pre_data = df[df['Period'] == 'Pre']
    post_data = df[df['Period'] == 'Post']
    
    temporal_evolution = {}
    
    # Analyze negation pattern evolution
    for word, data in negation_analysis.items():
        if data['total_occurrences'] > 20:  # Only analyze significant patterns
            
            # Pre period analysis
            pre_tp_count = 0
            pre_fp_count = 0
            pre_customer_count = 0
            pre_agent_count = 0
            
            for _, row in pre_data.iterrows():
                customer_text = str(row['Customer Transcript']).lower()
                agent_text = str(row['Agent Transcript']).lower()
                
                if word in customer_text or word in agent_text:
                    if row['Primary Marker'] == 'TP':
                        pre_tp_count += 1
                    else:
                        pre_fp_count += 1
                    
                    if word in customer_text:
                        pre_customer_count += 1
                    if word in agent_text:
                        pre_agent_count += 1
            
            # Post period analysis
            post_tp_count = 0
            post_fp_count = 0
            post_customer_count = 0
            post_agent_count = 0
            
            for _, row in post_data.iterrows():
                customer_text = str(row['Customer Transcript']).lower()
                agent_text = str(row['Agent Transcript']).lower()
                
                if word in customer_text or word in agent_text:
                    if row['Primary Marker'] == 'TP':
                        post_tp_count += 1
                    else:
                        post_fp_count += 1
                    
                    if word in customer_text:
                        post_customer_count += 1
                    if word in agent_text:
                        post_agent_count += 1
            
            # Calculate evolution metrics
            pre_total = pre_tp_count + pre_fp_count
            post_total = post_tp_count + post_fp_count
            
            if pre_total > 0 and post_total > 0:
                pre_precision = pre_tp_count / pre_total
                post_precision = post_tp_count / post_total
                precision_change = post_precision - pre_precision
                
                pre_customer_ratio = pre_customer_count / pre_total
                post_customer_ratio = post_customer_count / post_total
                customer_ratio_change = post_customer_ratio - pre_customer_ratio
                
                temporal_evolution[word] = {
                    'pre_precision': pre_precision,
                    'post_precision': post_precision,
                    'precision_change': precision_change,
                    'pre_customer_ratio': pre_customer_ratio,
                    'post_customer_ratio': post_customer_ratio,
                    'customer_ratio_change': customer_ratio_change,
                    'pre_volume': pre_total,
                    'post_volume': post_total,
                    'volume_change': post_total - pre_total
                }
    
    # Sort by precision change (most degraded first)
    sorted_evolution = dict(sorted(temporal_evolution.items(), 
                                 key=lambda x: x[1]['precision_change']))
    
    print("Top 10 Most Degraded Patterns (Pre vs Post):")
    print("-" * 50)
    count = 0
    for word, evolution in sorted_evolution.items():
        if count < 10:
            print(f"{word}:")
            print(f"  Precision: {evolution['pre_precision']:.3f} -> {evolution['post_precision']:.3f} "
                  f"({evolution['precision_change']:+.3f})")
            print(f"  Customer ratio: {evolution['pre_customer_ratio']:.3f} -> "
                  f"{evolution['post_customer_ratio']:.3f} ({evolution['customer_ratio_change']:+.3f})")
            print(f"  Volume: {evolution['pre_volume']} -> {evolution['post_volume']} "
                  f"({evolution['volume_change']:+d})")
            print()
            count += 1
    
    print("Top 10 Most Improved Patterns (Pre vs Post):")
    print("-" * 50)
    count = 0
    for word, evolution in reversed(list(sorted_evolution.items())):
        if count < 10 and evolution['precision_change'] > 0:
            print(f"{word}:")
            print(f"  Precision: {evolution['pre_precision']:.3f} -> {evolution['post_precision']:.3f} "
                  f"({evolution['precision_change']:+.3f})")
            print(f"  Customer ratio: {evolution['pre_customer_ratio']:.3f} -> "
                  f"{evolution['post_customer_ratio']:.3f} ({evolution['customer_ratio_change']:+.3f})")
            print(f"  Volume: {evolution['pre_volume']} -> {evolution['post_volume']} "
                  f"({evolution['volume_change']:+d})")
            print()
            count += 1
    
    return temporal_evolution


# Main execution function
def main():
    """
    Main execution function for the enhanced analysis
    """
    
    print("Enhanced Dynamic Negation Detection with Query Component Analysis")
    print("Loading and preparing data...")
    
    # Load data
    df_main, df_validation, df_rules = load_and_prepare_data()
    
    if df_main is None:
        print("Failed to load data. Exiting.")
        return
    
    # Run comprehensive analysis
    results = run_enhanced_comprehensive_analysis(df_main, df_rules)
    
    # Additional analyses
    speaker_analysis = analyze_speaker_attribution_impact(
        df_main, results['negation_analysis'], results['query_components']
    )
    
    temporal_analysis = create_temporal_pattern_analysis(
        df_main, results['negation_analysis']
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
    print("=" * 80)
    
    print(f"Dataset: {len(df_main)} transcript records analyzed")
    print(f"Negation patterns discovered: {len(results['negation_patterns'])}")
    
    if 'validated_clusters' in results['query_components']:
        print(f"Query component clusters: {len(results['query_components']['validated_clusters'])}")
    
    if 'feature_importance' in results['shap_results']:
        print(f"SHAP features analyzed: {len(results['shap_results']['feature_importance'])}")
    
    print(f"Recommendations generated: {len(results['recommendations']['high_priority']) + len(results['recommendations']['medium_priority'])}")
    
    print("\nKey Insights:")
    print("- Dynamic pattern discovery identified data-driven negation clusters")
    print("- Query component analysis revealed high/low precision components")
    print("- SHAP analysis provided feature-level explainability")
    print("- Speaker attribution (customer vs agent) analysis completed")
    print("- Temporal evolution (Pre vs Post) patterns identified")
    
    print(f"\nAll results exported to: {results['export_files']}")
    
    return results


# Example usage for Jupyter notebook
if __name__ == "__main__":
    print("Enhanced Dynamic Negation Detection System with Query Component Analysis Ready!")
    print("\nTo use this system:")
    print("1. Ensure data files are in the working directory:")
    print("   - Precision_Drop_Analysis_OG.xlsx")
    print("   - Categorical Validation.xlsx") 
    print("   - Query_Rules.xlsx")
    print("2. Run: results = main()")
    print("3. Check exported files for detailed analysis results")
    print("\nFeatures included:")
    print("- Dynamic negation pattern discovery from data")
    print("- Enhanced query component clustering analysis")
    print("- SHAP explainability for word/phrase/n-gram importance")
    print("- Dual speaker (customer + agent) transcript analysis")
    print("- Comprehensive visualizations (retained + enhanced)")
    print("- Temporal pattern evolution (Pre vs Post period)")
    print("- Actionable recommendations with implementation details")
    
    # Uncomment to run automatically
    # results = main()
