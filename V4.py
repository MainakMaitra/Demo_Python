## 1. **Deeper Query Rule Analysis**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import re
from collections import Counter
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)

print("=== STRUCTURED COMPLAINTS PRECISION DROP ANALYSIS ===")
print("Investigation Framework: Systematic Root Cause Analysis")
print("Target: Maintain 70% precision for complaints, 30% for non-complaints")
print("Approach: Macro → Deep Dive → Root Cause Analysis\n")

# =============================================================================
# DATA PREPARATION AND LOADING
# =============================================================================

# Enhanced Data Preparation with Monthly Tracking Framework
# Modification 1: Add Period Classification and Monthly Analysis Framework

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
    
    # Category metadata using Query Rules begin_date
    if df_rules_filtered is not None and 'begin_date' in df_rules_filtered.columns:
        # Process begin_date from rules
        df_rules_filtered['begin_date'] = pd.to_datetime(df_rules_filtered['begin_date'], errors='coerce')
        
        # Create mapping of (Event, Query) -> begin_date
        category_date_mapping = df_rules_filtered.groupby(['Event', 'Query'])['begin_date'].min().to_dict()
        
        # Apply mapping to main dataframe
        df_main['Category_Added_Date'] = df_main.apply(
            lambda row: category_date_mapping.get((row['Prosodica L1'], row['Prosodica L2']), pd.NaT), 
            axis=1
        )
        
        # Convert to datetime and handle NaT values
        df_main['Category_Added_Date'] = pd.to_datetime(df_main['Category_Added_Date'])
        
        # For categories without begin_date, use a default early date
        default_date = pd.to_datetime('2024-01-01')
        df_main['Category_Added_Date'] = df_main['Category_Added_Date'].fillna(default_date)
        
        # Calculate category age and new category flag
        df_main['Category_Age_Days'] = (df_main['Date'] - df_main['Category_Added_Date']).dt.days
        df_main['Is_New_Category'] = df_main['Category_Age_Days'] <= 30
        
        print(f"Category date mapping applied successfully.")
        print(f"Categories with begin_date: {len(category_date_mapping)}")
        print(f"Records flagged as new categories: {df_main['Is_New_Category'].sum()}")
        
    else:
        print("Warning: begin_date column not found in Query Rules. Using default category dating.")
        # Fallback: use default early date for all categories
        default_date = pd.to_datetime('2024-01-01')
        df_main['Category_Added_Date'] = default_date
        df_main['Category_Age_Days'] = (df_main['Date'] - df_main['Category_Added_Date']).dt.days
        df_main['Is_New_Category'] = False  # All categories considered old
    
    print(f"Enhanced data preparation completed. Final dataset shape: {df_main.shape}")
    
    return df_main, df_validation, df_rules_filtered

# Data Preparation
df_main, df_validation, df_rules = load_and_prepare_data()

### Query Evolution Timeline

def analyze_query_rule_evolution(df_main, df_rules):
    """Analyze how query modifications correlate with precision changes"""
    
    # Track when each query was last modified
    query_modifications = df_rules.groupby(['Event', 'Query']).agg({
        'begin_date': ['min', 'max', 'count']
    }).reset_index()
    
    # Correlate query changes with precision drops
    for query in query_modifications.itertuples():
        # Find precision before and after query modification
        before_data = df_main[
            (df_main['Prosodica L1'] == query.Event) & 
            (df_main['Prosodica L2'] == query.Query) &
            (df_main['Date'] < query.begin_date_max)
        ]
        after_data = df_main[
            (df_main['Prosodica L1'] == query.Event) & 
            (df_main['Prosodica L2'] == query.Query) &
            (df_main['Date'] >= query.begin_date_max)
        ]
        
        if len(before_data) > 10 and len(after_data) > 10:
            before_precision = before_data['Is_TP'].mean()
            after_precision = after_data['Is_TP'].mean()
            precision_change = after_precision - before_precision
            
            print(f"{query.Query}: {precision_change:+.3f} after modification")


### Query Component Effectiveness

def analyze_query_components(df_main, df_rules):
    """Break down which parts of queries are causing FPs"""
    
    # Extract individual query components
    for _, rule in df_rules.iterrows():
        query_text = rule['Query Text']
        
        # Parse query into components
        components = re.findall(r'"([^"]+)"|\b(\w+)\b', query_text)
        
        # Test each component's correlation with FPs
        for component in components:
            if component[0]:  # Quoted phrase
                phrase = component[0]
                # Check FP rate when this phrase appears
                contains_phrase = df_main[
                    df_main['Full_Transcript'].str.contains(phrase, case=False, na=False)
                ]
                fp_rate = contains_phrase['Is_FP'].mean()
                print(f"Phrase '{phrase}': FP rate = {fp_rate:.3f}")


## 2. **Context Window Analysis**

### Surrounding Context Extraction

def analyze_context_windows(df_main):
    """Analyze text surrounding complaint keywords"""
    
    complaint_keywords = ['problem', 'issue', 'wrong', 'error', 'frustrated']
    context_window = 50  # characters before/after
    
    # Helper function to extract common patterns
    def extract_common_patterns(contexts):
        """Extract common words/phrases from context list"""
        if not contexts:
            return []
        
        # Combine all contexts
        all_text = ' '.join(contexts).lower()
        
        # Remove the keyword itself and common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'this', 'that'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Also look for common bigrams
        bigrams = []
        for i in range(len(words)-1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                bigrams.append(f"{words[i]} {words[i+1]}")
        
        bigram_freq = {}
        for bigram in bigrams:
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        common_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'common_words': [word for word, count in common_words],
            'common_phrases': [phrase for phrase, count in common_bigrams]
        }
    
    context_analysis_results = []
    
    for keyword in complaint_keywords:
        tp_contexts = []
        fp_contexts = []
        
        # Analyze each row
        for _, row in df_main.iterrows():
            text = str(row['Customer Transcript']).lower()
            
            if keyword in text:
                # Find all occurrences of the keyword
                positions = [m.start() for m in re.finditer(r'\b' + keyword + r'\b', text)]
                
                for pos in positions:
                    # Extract context
                    start = max(0, pos - context_window)
                    end = min(len(text), pos + len(keyword) + context_window)
                    context = text[start:end]
                    
                    # Store context based on classification
                    if row['Primary Marker'] == 'TP':
                        tp_contexts.append(context)
                    else:
                        fp_contexts.append(context)
        
        # Analyze patterns
        tp_patterns = extract_common_patterns(tp_contexts)
        fp_patterns = extract_common_patterns(fp_contexts)
        
        print(f"\nKeyword: '{keyword}'")
        print(f"Found in {len(tp_contexts)} TP contexts and {len(fp_contexts)} FP contexts")
        
        if tp_patterns:
            print(f"TP contexts often include words: {tp_patterns['common_words']}")
            print(f"TP contexts common phrases: {tp_patterns['common_phrases']}")
        
        if fp_patterns:
            print(f"FP contexts often include words: {fp_patterns['common_words']}")
            print(f"FP contexts common phrases: {fp_patterns['common_phrases']}")
        
        # Store results
        context_analysis_results.append({
            'keyword': keyword,
            'tp_count': len(tp_contexts),
            'fp_count': len(fp_contexts),
            'tp_patterns': tp_patterns,
            'fp_patterns': fp_patterns
        })
        
        # Show example contexts
        if len(tp_contexts) > 0 and len(fp_contexts) > 0:
            print("\nExample TP context:")
            print(f"...{tp_contexts[0]}...")
            print("\nExample FP context:")
            print(f"...{fp_contexts[0]}...")
    
    # Identify distinguishing patterns
    print("\n" + "="*50)
    print("DISTINGUISHING PATTERNS SUMMARY")
    print("="*50)
    
    for result in context_analysis_results:
        keyword = result['keyword']
        tp_words = set(result['tp_patterns'].get('common_words', []))
        fp_words = set(result['fp_patterns'].get('common_words', []))
        
        # Words unique to TP contexts
        tp_unique = tp_words - fp_words
        # Words unique to FP contexts
        fp_unique = fp_words - tp_words
        
        if tp_unique or fp_unique:
            print(f"\n'{keyword}' distinguishing patterns:")
            if tp_unique:
                print(f"  TP indicators: {list(tp_unique)}")
            if fp_unique:
                print(f"  FP indicators: {list(fp_unique)}")
    
    return context_analysis_results

# Call the function
context_results = analyze_context_windows(df_main)


## 3. **Conversation Flow Analysis**

### Turn-Taking Patterns

def analyze_conversation_dynamics(df_main):
    """Analyze conversation flow patterns"""
    
    # Split conversations into turns
    for _, row in df_main.iterrows():
        customer_turns = len(re.findall(r'[.!?]+', row['Customer Transcript']))
        agent_turns = len(re.findall(r'[.!?]+', row['Agent Transcript']))
        
        # Calculate conversation metrics
        turn_ratio = customer_turns / (agent_turns + 1)
        avg_customer_turn_length = row['Customer_Word_Count'] / (customer_turns + 1)
        avg_agent_turn_length = row['Agent_Word_Count'] / (agent_turns + 1)
        
        # Add to dataframe
        df_main.loc[_, 'Turn_Ratio'] = turn_ratio
        df_main.loc[_, 'Avg_Customer_Turn_Length'] = avg_customer_turn_length
        df_main.loc[_, 'Avg_Agent_Turn_Length'] = avg_agent_turn_length
    
    # Compare TP vs FP conversation dynamics
    tp_dynamics = df_main[df_main['Primary Marker'] == 'TP'][
        ['Turn_Ratio', 'Avg_Customer_Turn_Length', 'Avg_Agent_Turn_Length']
    ].mean()
    
    fp_dynamics = df_main[df_main['Primary Marker'] == 'FP'][
        ['Turn_Ratio', 'Avg_Customer_Turn_Length', 'Avg_Agent_Turn_Length']
    ].mean()
    
    print("Conversation Dynamics Comparison:")
    print(f"TP conversations: {tp_dynamics}")
    print(f"FP conversations: {fp_dynamics}")


## 4. **Semantic Similarity Analysis**

### Category Confusion Matrix

def analyze_category_confusion(df_main):
    """Identify which categories are often confused"""
    
    # For transcripts with secondary validation where there's disagreement
    confusion_data = df_main[
        df_main['Has_Secondary_Validation'] & 
        df_main['Secondary L1'].notna() &
        df_main['Secondary L2'].notna()
    ].copy()
    
    if len(confusion_data) == 0:
        print("No secondary validation data available for confusion analysis")
        return pd.DataFrame()
    
    # Create confusion matrix for L1 categories
    print("L1 Category Confusion Matrix:")
    l1_confusion = pd.crosstab(
        confusion_data['Primary L1'].fillna('None'), 
        confusion_data['Secondary L1'].fillna('None')
    )
    print(l1_confusion)
    
    # Create confusion matrix for L2 categories
    print("\nL2 Category Confusion Matrix:")
    l2_confusion = pd.crosstab(
        confusion_data['Primary L2'].fillna('None'), 
        confusion_data['Secondary L2'].fillna('None')
    )
    
    # Find most confused L1 pairs
    confused_l1_pairs = []
    for primary in l1_confusion.index:
        for secondary in l1_confusion.columns:
            if primary != secondary and l1_confusion.loc[primary, secondary] > 0:
                confused_l1_pairs.append({
                    'Primary_L1': primary,
                    'Secondary_L1': secondary,
                    'Count': l1_confusion.loc[primary, secondary],
                    'Type': 'L1_Confusion'
                })
    
    # Find most confused L2 pairs
    confused_l2_pairs = []
    for primary in l2_confusion.index:
        for secondary in l2_confusion.columns:
            if primary != secondary and l2_confusion.loc[primary, secondary] > 0:
                confused_l2_pairs.append({
                    'Primary_L2': primary,
                    'Secondary_L2': secondary,
                    'Count': l2_confusion.loc[primary, secondary],
                    'Type': 'L2_Confusion'
                })
    
    # Combine and sort
    all_confused_pairs = pd.DataFrame(confused_l1_pairs + confused_l2_pairs)
    
    if len(all_confused_pairs) > 0:
        all_confused_pairs = all_confused_pairs.sort_values('Count', ascending=False)
        
        print("\nMost Confused Category Pairs:")
        print(all_confused_pairs.head(10))
        
        # Analyze disagreement patterns
        disagreement_analysis = confusion_data[
            (confusion_data['Primary L1'] != confusion_data['Secondary L1']) |
            (confusion_data['Primary L2'] != confusion_data['Secondary L2'])
        ]
        
        print(f"\nDisagreement Statistics:")
        print(f"Total records with secondary validation: {len(confusion_data)}")
        print(f"Records with disagreement: {len(disagreement_analysis)} ({len(disagreement_analysis)/len(confusion_data)*100:.1f}%)")
        
        # Show example disagreements
        if len(disagreement_analysis) > 0:
            print("\nExample Disagreements:")
            sample_disagreements = disagreement_analysis[
                ['variable5', 'Primary L1', 'Primary L2', 'Secondary L1', 'Secondary L2', 'Customer Transcript']
            ].head(5)
            
            for idx, row in sample_disagreements.iterrows():
                print(f"\nVariable5: {row['variable5']}")
                print(f"Primary: {row['Primary L1']} - {row['Primary L2']}")
                print(f"Secondary: {row['Secondary L1']} - {row['Secondary L2']}")
                print(f"Transcript snippet: {row['Customer Transcript'][:100]}...")
    
    return all_confused_pairs

# Call the corrected function
confused_categories = analyze_category_confusion(df_main)


## 5. **Call Metadata Analysis**

### Call Duration and Precision

def analyze_call_metadata(df_main):
    """Analyze if call characteristics affect precision"""
    
    # Estimate call duration from transcript length
    df_main['Estimated_Duration'] = df_main['Transcript_Length'] / 150  # chars per second
    
    # Bin calls by duration
    df_main['Duration_Bin'] = pd.qcut(
        df_main['Estimated_Duration'], 
        q=5, 
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Analyze precision by duration
    duration_precision = df_main.groupby('Duration_Bin').agg({
        'Is_TP': 'mean',
        'Is_FP': 'mean',
        'Customer_Negation_Count': 'mean',
        'Customer_Agent_Ratio': 'mean'
    })
    
    print("Precision by Call Duration:")
    print(duration_precision)


## 6. **Error Propagation Analysis**

### Track How Errors Cascade

def analyze_error_propagation(df_main):
    """Analyze how one misclassification leads to others"""
    
    # Group by conversation (variable5)
    conversation_errors = df_main.groupby('variable5').agg({
        'Is_FP': ['sum', 'count'],
        'Prosodica L1': 'nunique',
        'UUID': 'count'
    })
    
    conversation_errors.columns = ['FP_Count', 'Total_Classifications', 'Unique_Categories', 'UUID_Count']
    conversation_errors['FP_Rate'] = conversation_errors['FP_Count'] / conversation_errors['Total_Classifications']
    
    # Identify conversations with cascading errors
    cascading_errors = conversation_errors[
        (conversation_errors['FP_Count'] > 1) & 
        (conversation_errors['FP_Rate'] > 0.5)
    ]
    
    print(f"Conversations with cascading errors: {len(cascading_errors)}")
    
    # Analyze patterns in cascading errors
    for conv_id in cascading_errors.index[:5]:  # Top 5
        conv_data = df_main[df_main['variable5'] == conv_id]
        print(f"\nConversation {conv_id}:")
        print(f"Categories flagged: {conv_data['Prosodica L2'].unique()}")
        print(f"FP rate: {conv_data['Is_FP'].mean():.3f}")


## 7. **Real-Time Performance Monitoring**

### Daily Precision Tracking

def create_daily_monitoring_dashboard(df_main):
    """Create daily precision monitoring metrics"""
    
    daily_metrics = df_main.groupby('Date').agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'variable5': 'nunique',
        'Customer_Negation_Count': 'mean',
        'Customer_Agent_Ratio': 'mean'
    })
    
    daily_metrics.columns = ['TP_Count', 'Total_Flagged', 'FP_Count', 
                           'Unique_Calls', 'Avg_Negations', 'Avg_Ratio']
    daily_metrics['Precision'] = daily_metrics['TP_Count'] / daily_metrics['Total_Flagged']
    daily_metrics['Precision_7day_MA'] = daily_metrics['Precision'].rolling(7).mean()
    
    # Detect anomalies
    daily_metrics['Z_Score'] = (
        (daily_metrics['Precision'] - daily_metrics['Precision_7day_MA']) / 
        daily_metrics['Precision'].rolling(7).std()
    )
    
    anomaly_days = daily_metrics[abs(daily_metrics['Z_Score']) > 2]
    
    print(f"Days with anomalous precision: {len(anomaly_days)}")
    return daily_metrics


## 8. **Machine Learning Insights**

### Feature Importance for FP Prediction

def analyze_fp_drivers_ml(df_main):
    """Use ML to identify key FP drivers"""
    
    # Prepare features
    feature_cols = [
        'Transcript_Length', 'Customer_Word_Count', 'Agent_Word_Count',
        'Customer_Agent_Ratio', 'Customer_Question_Count', 'Customer_Exclamation_Count',
        'Customer_Caps_Ratio', 'Customer_Negation_Count', 'Agent_Negation_Count',
        'Customer_Qualifying_Count', 'WeekOfMonth', 'Is_Holiday_Season'
    ]
    
    X = df_main[feature_cols].fillna(0)
    y = df_main['Is_FP']
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top FP Predictors:")
    print(importance.head(10))
    
    return rf, importance
