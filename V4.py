# Dynamic Negation and Lexicon Analysis - Step by Step Notebook
# Save this as: dynamic_negation_analysis_notebook.ipynb

# =============================================================================
# CELL 1: Setup and Environment Check
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("="*60)
print("DYNAMIC NEGATION ANALYSIS - STEP BY STEP")
print("="*60)

def check_environment():
    """Check if all required packages are available"""
    
    required_packages = [
        ('pandas', 'pd'), ('numpy', 'np'), ('matplotlib.pyplot', 'plt'), 
        ('seaborn', 'sns'), ('sklearn', 'sklearn'), ('plotly', 'plotly'),
        ('wordcloud', 'WordCloud'), ('re', 're')
    ]
    
    print("Checking environment...")
    missing = []
    
    for package, alias in required_packages:
        try:
            if '.' in package:
                __import__(package.split('.')[0])
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    # Check spaCy
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ spacy with en_core_web_sm model")
        except OSError:
            print("⚠ spacy installed but missing en_core_web_sm model")
            print("  Install with: python -m spacy download en_core_web_sm")
    except ImportError:
        print("✗ spacy - MISSING")
        missing.append('spacy')
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All packages available!")
        return True

# Run environment check
check_result = check_environment()

# =============================================================================
# CELL 2: Data Loading and Initial Validation
# =============================================================================

def load_and_validate_data():
    """Load and validate the input data"""
    
    print("\n" + "="*50)
    print("DATA LOADING AND VALIDATION")
    print("="*50)
    
    try:
        # Load the main dataset
        print("Loading Precision_Drop_Analysis_OG.xlsx...")
        df = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        df.columns = df.columns.str.rstrip()
        
        print(f"✓ Data loaded: {df.shape}")
        
        # Basic validation
        required_columns = ['UUID', 'Primary Marker', 'Customer Transcript', 'Agent Transcript', 'Date']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print("✓ All required columns present")
        
        # Show data overview
        print(f"\nData Overview:")
        print(f"Records: {len(df):,}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Primary Marker distribution:")
        print(df['Primary Marker'].value_counts())
        
        # Filter dissatisfaction if needed
        original_size = len(df)
        if 'Prosodica L1' in df.columns:
            df = df[df['Prosodica L1'].str.lower() != 'dissatisfaction']
            if len(df) < original_size:
                print(f"Filtered out {original_size - len(df)} dissatisfaction records")
        
        print(f"Final dataset: {len(df):,} records")
        
        return df
        
    except FileNotFoundError:
        print("❌ File 'Precision_Drop_Analysis_OG.xlsx' not found")
        print("Please ensure the file is in the same directory as this notebook")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Load data
df_raw = load_and_validate_data()

# Display first few rows if successful
if df_raw is not None:
    print("\nFirst 3 rows:")
    display(df_raw.head(3))
else:
    print("⚠️ Data loading failed. Please check your file and re-run this cell.")

# =============================================================================
# CELL 3: Data Preparation and Feature Engineering
# =============================================================================

def prepare_data_for_analysis(df):
    """Prepare data with all necessary features for analysis"""
    
    print("\n" + "="*50)
    print("DATA PREPARATION AND FEATURE ENGINEERING")
    print("="*50)
    
    df = df.copy()
    
    # 1. Date processing
    print("1. Processing dates...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    df['DayOfWeek'] = df['Date'].dt.day_name()
    
    # 2. CRITICAL: Period classification for Pre vs Post analysis
    print("2. Classifying periods (Pre vs Post)...")
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df['Period'] = df['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    period_summary = df['Period'].value_counts()
    print("Period distribution:")
    for period, count in period_summary.items():
        print(f"  {period}: {count:,} records")
    
    if period_summary.get('Pre', 0) == 0 or period_summary.get('Post', 0) == 0:
        print("⚠️ WARNING: Missing Pre or Post period data!")
        print("You may need to adjust the month ranges above.")
    
    # 3. Text preprocessing
    print("3. Processing transcripts...")
    df['Customer Transcript'] = df['Customer Transcript'].fillna('')
    df['Agent Transcript'] = df['Agent Transcript'].fillna('')
    df['Full_Transcript'] = df['Customer Transcript'] + ' ' + df['Agent Transcript']
    
    # 4. Basic text features
    print("4. Creating text features...")
    df['Transcript_Length'] = df['Full_Transcript'].str.len()
    df['Customer_Word_Count'] = df['Customer Transcript'].str.split().str.len()
    df['Agent_Word_Count'] = df['Agent Transcript'].str.split().str.len()
    df['Customer_Agent_Ratio'] = df['Customer_Word_Count'] / (df['Agent_Word_Count'] + 1)
    
    # 5. Quick quality checks
    print("5. Quality checks...")
    empty_transcripts = (df['Customer Transcript'].str.len() == 0) | (df['Agent Transcript'].str.len() == 0)
    print(f"Empty transcripts: {empty_transcripts.sum():,}")
    
    print(f"Average transcript length: {df['Transcript_Length'].mean():.0f} characters")
    print(f"Average customer words: {df['Customer_Word_Count'].mean():.0f}")
    print(f"Average agent words: {df['Agent_Word_Count'].mean():.0f}")
    
    print("✅ Data preparation completed!")
    
    return df

# Prepare the data
if df_raw is not None:
    df_prepared = prepare_data_for_analysis(df_raw)
    
    # Show preparation results
    print(f"\nPrepared dataset shape: {df_prepared.shape}")
    print("\nNew columns added:")
    new_cols = ['Period', 'Year_Month', 'Full_Transcript', 'Transcript_Length', 
                'Customer_Word_Count', 'Agent_Word_Count', 'Customer_Agent_Ratio']
    for col in new_cols:
        if col in df_prepared.columns:
            print(f"  ✓ {col}")
    
    # Show sample of prepared data
    display(df_prepared[['UUID', 'Primary Marker', 'Period', 'Year_Month', 'Transcript_Length']].head())
else:
    print("⚠️ Cannot prepare data - please fix data loading first.")

# =============================================================================
# CELL 4: Step 1 - Dynamic Negation Context Discovery
# =============================================================================

def discover_negation_contexts(df):
    """
    Discover negation contexts from transcripts
    KEY IMPROVEMENT: No hardcoded patterns - learns from data
    """
    
    print("\n" + "="*60)
    print("STEP 1: DYNAMIC NEGATION CONTEXT DISCOVERY")
    print("="*60)
    
    # Start with minimal base negation words
    base_negation_words = [
        'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'none',
        "don't", "won't", "can't", "isn't", "aren't", "wasn't", "weren't",
        "doesn't", "didn't", "haven't", "hasn't", "hadn't", "couldn't",
        "wouldn't", "shouldn't", "mustn't"
    ]
    
    print(f"Starting with {len(base_negation_words)} base negation words")
    print("Extracting contexts (this may take a moment)...")
    
    negation_contexts = []
    processed_count = 0
    
    for idx, row in df.iterrows():
        processed_count += 1
        if processed_count % 500 == 0:
            print(f"  Processed {processed_count:,}/{len(df):,} records...")
        
        # Process both customer and agent transcripts
        for transcript_type in ['Customer Transcript', 'Agent Transcript']:
            text = str(row[transcript_type]).lower()
            if len(text) < 10:  # Skip very short texts
                continue
            
            # Find all negation words in this transcript
            for neg_word in base_negation_words:
                import re
                pattern = r'\b' + re.escape(neg_word) + r'\b'
                
                for match in re.finditer(pattern, text):
                    # Extract context: 80 characters before and after
                    start_pos = max(0, match.start() - 80)
                    end_pos = min(len(text), match.end() + 80)
                    context = text[start_pos:end_pos]
                    
                    negation_contexts.append({
                        'UUID': row['UUID'],
                        'Primary_Marker': row['Primary Marker'],
                        'Period': row['Period'],
                        'Year_Month': row['Year_Month'],
                        'Speaker': transcript_type.split()[0].lower(),
                        'Negation_Word': neg_word,
                        'Context': context,
                        'Position_In_Text': match.start() / len(text),  # Relative position
                        'Text_Length': len(text)
                    })
    
    negation_df = pd.DataFrame(negation_contexts)
    
    print(f"\n✅ CONTEXT EXTRACTION COMPLETED")
    print(f"Total negation instances: {len(negation_df):,}")
    print(f"Unique transcripts with negations: {negation_df['UUID'].nunique():,}")
    print(f"Customer negations: {len(negation_df[negation_df['Speaker'] == 'customer']):,}")
    print(f"Agent negations: {len(negation_df[negation_df['Speaker'] == 'agent']):,}")
    
    # Show distribution by marker
    print(f"\nNegation distribution by Primary Marker:")
    marker_dist = negation_df['Primary_Marker'].value_counts()
    for marker, count in marker_dist.items():
        print(f"  {marker}: {count:,} ({count/len(negation_df)*100:.1f}%)")
    
    # Show distribution by period
    print(f"\nNegation distribution by Period:")
    period_dist = negation_df['Period'].value_counts()
    for period, count in period_dist.items():
        print(f"  {period}: {count:,} ({count/len(negation_df)*100:.1f}%)")
    
    return negation_df

# Run negation context discovery
if 'df_prepared' in locals() and df_prepared is not None:
    negation_df = discover_negation_contexts(df_prepared)
    
    print(f"\nSample negation contexts:")
    if len(negation_df) > 0:
        display(negation_df[['UUID', 'Primary_Marker', 'Speaker', 'Negation_Word', 'Context']].head(3))
    else:
        print("⚠️ No negation contexts found! Check your data.")
else:
    print("⚠️ Please run previous cells first to prepare the data.")

# =============================================================================
# CELL 5: Step 2 - Pattern Clustering and Discovery
# =============================================================================

def cluster_negation_patterns(negation_df):
    """
    Cluster negation contexts to discover patterns automatically
    This replaces hardcoded pattern categories!
    """
    
    print("\n" + "="*60)
    print("STEP 2: PATTERN CLUSTERING AND DISCOVERY")
    print("="*60)
    
    if len(negation_df) == 0:
        print("❌ No negation contexts to cluster!")
        return negation_df, pd.DataFrame()
    
    print("1. Creating TF-IDF vectors from contexts...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=500,  # Reduced for notebook performance
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=3,  # Must appear in at least 3 contexts
        max_df=0.8  # Exclude very common words
    )
    
    try:
        print(f"   Vectorizing {len(negation_df):,} contexts...")
        context_vectors = vectorizer.fit_transform(negation_df['Context'])
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"   Created {context_vectors.shape[1]} features")
        
        # Determine number of clusters based on data size
        n_contexts = len(negation_df)
        if n_contexts > 2000:
            n_clusters = 8
        elif n_contexts > 1000:
            n_clusters = 6
        elif n_contexts > 500:
            n_clusters = 5
        elif n_contexts > 100:
            n_clusters = 4
        else:
            n_clusters = 3
        
        print(f"2. Clustering into {n_clusters} patterns...")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(context_vectors)
        negation_df['Pattern_Cluster'] = cluster_labels
        
        print(f"   ✅ Clustering completed")
        
        # Analyze each discovered pattern
        print("3. Analyzing discovered patterns...")
        pattern_analysis = []
        
        for cluster_id in range(n_clusters):
            cluster_data = negation_df[negation_df['Pattern_Cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Get characteristic features for this cluster
            cluster_indices = negation_df[negation_df['Pattern_Cluster'] == cluster_id].index
            cluster_vectors = context_vectors[cluster_indices]
            
            # Calculate mean TF-IDF scores for this cluster
            mean_scores = np.mean(cluster_vectors.toarray(), axis=0)
            top_feature_indices = np.argsort(mean_scores)[-8:][::-1]  # Top 8 features
            top_features = [feature_names[i] for i in top_feature_indices]
            
            # Calculate performance metrics
            tp_count = len(cluster_data[cluster_data['Primary_Marker'] == 'TP'])
            fp_count = len(cluster_data[cluster_data['Primary_Marker'] == 'FP'])
            total_count = len(cluster_data)
            
            pre_count = len(cluster_data[cluster_data['Period'] == 'Pre'])
            post_count = len(cluster_data[cluster_data['Period'] == 'Post'])
            
            customer_count = len(cluster_data[cluster_data['Speaker'] == 'customer'])
            agent_count = len(cluster_data[cluster_data['Speaker'] == 'agent'])
            
            # Performance calculations
            tp_rate = tp_count / total_count if total_count > 0 else 0
            fp_rate = fp_count / total_count if total_count > 0 else 0
            quality_score = tp_rate - fp_rate
            
            # Volume change
            volume_change = post_count - pre_count
            volume_change_pct = (volume_change / max(pre_count, 1)) * 100
            
            pattern_analysis.append({
                'Cluster_ID': cluster_id,
                'Total_Count': total_count,
                'TP_Count': tp_count,
                'FP_Count': fp_count,
                'TP_Rate': tp_rate,
                'FP_Rate': fp_rate,
                'Quality_Score': quality_score,
                'Pre_Count': pre_count,
                'Post_Count': post_count,
                'Volume_Change': volume_change,
                'Volume_Change_Pct': volume_change_pct,
                'Customer_Count': customer_count,
                'Agent_Count': agent_count,
                'Customer_Ratio': customer_count / total_count if total_count > 0 else 0,
                'Top_Features': ', '.join(top_features[:5]),
                'All_Features': top_features
            })
        
        pattern_df = pd.DataFrame(pattern_analysis)
        pattern_df = pattern_df.sort_values('Total_Count', ascending=False)
        
        print(f"   ✅ Analysis completed for {len(pattern_df)} patterns")
        
        return negation_df, pattern_df
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        print("Using fallback: clustering by negation word...")
        
        # Simple fallback clustering
        negation_df['Pattern_Cluster'] = negation_df['Negation_Word'].astype('category').cat.codes
        
        # Create simple analysis
        simple_analysis = negation_df.groupby('Pattern_Cluster').agg({
            'UUID': 'count',
            'Primary_Marker': lambda x: (x == 'TP').sum(),
            'Period': lambda x: (x == 'Pre').sum(),
            'Speaker': lambda x: (x == 'customer').sum(),
            'Negation_Word': 'first'
        }).reset_index()
        
        simple_analysis.columns = ['Cluster_ID', 'Total_Count', 'TP_Count', 'Pre_Count', 'Customer_Count', 'Primary_Word']
        simple_analysis['FP_Count'] = simple_analysis['Total_Count'] - simple_analysis['TP_Count']
        simple_analysis['Post_Count'] = simple_analysis['Total_Count'] - simple_analysis['Pre_Count']
        simple_analysis['TP_Rate'] = simple_analysis['TP_Count'] / simple_analysis['Total_Count']
        simple_analysis['FP_Rate'] = 1 - simple_analysis['TP_Rate']
        simple_analysis['Quality_Score'] = simple_analysis['TP_Rate'] - simple_analysis['FP_Rate']
        simple_analysis['Top_Features'] = simple_analysis['Primary_Word']
        
        return negation_df, simple_analysis

# Run pattern clustering
if 'negation_df' in locals() and len(negation_df) > 0:
    negation_df_clustered, pattern_analysis_df = cluster_negation_patterns(negation_df)
    
    if len(pattern_analysis_df) > 0:
        print(f"\n📊 DISCOVERED PATTERNS SUMMARY:")
        display(pattern_analysis_df[['Cluster_ID', 'Total_Count', 'TP_Rate', 'FP_Rate', 
                                   'Quality_Score', 'Volume_Change', 'Top_Features']])
    else:
        print("⚠️ No patterns discovered!")
else:
    print("⚠️ Please run the negation discovery cell first.")

# =============================================================================
# CELL 6: Pattern Analysis and Insights
# =============================================================================

def analyze_discovered_patterns(pattern_analysis_df, negation_df):
    """Analyze the discovered patterns and generate insights"""
    
    print("\n" + "="*60)
    print("STEP 3: PATTERN ANALYSIS AND INSIGHTS")
    print("="*60)
    
    if len(pattern_analysis_df) == 0:
        print("❌ No patterns to analyze!")
        return
    
    print("🔍 DETAILED PATTERN ANALYSIS:")
    print("-" * 40)
    
    for _, pattern in pattern_analysis_df.iterrows():
        cluster_id = pattern['Cluster_ID']
        total = pattern['Total_Count']
        tp_rate = pattern['TP_Rate']
        fp_rate = pattern['FP_Rate']
        quality = pattern['Quality_Score']
        volume_change = pattern.get('Volume_Change', 0)
        features = pattern['Top_Features']
        
        print(f"\n🏷️  PATTERN {cluster_id} ({total:,} instances)")
        print(f"   Performance: TP={tp_rate:.3f} | FP={fp_rate:.3f} | Quality={quality:+.3f}")
        print(f"   Volume Change: {volume_change:+d} (Pre→Post)")
        print(f"   Key Terms: {features}")
        
        # Risk assessment
        if fp_rate > 0.5 and total > 50:
            print(f"   🚨 HIGH RISK: High false positive rate with significant volume!")
        elif quality < -0.2:
            print(f"   ⚠️  ATTENTION: More false positives than true positives")
        elif volume_change > 100:
            print(f"   📈 VOLUME SURGE: Significant increase in Post period")
        elif tp_rate > 0.8:
            print(f"   ✅ GOOD PATTERN: High true positive rate")
    
    # Summary insights
    print(f"\n📈 SUMMARY INSIGHTS:")
    print(f"   Total patterns discovered: {len(pattern_analysis_df)}")
    print(f"   Average TP rate: {pattern_analysis_df['TP_Rate'].mean():.3f}")
    print(f"   Average FP rate: {pattern_analysis_df['FP_Rate'].mean():.3f}")
    
    # High-risk patterns
    high_risk = pattern_analysis_df[
        (pattern_analysis_df['FP_Rate'] > 0.4) & 
        (pattern_analysis_df['Total_Count'] > 20)
    ]
    
    if len(high_risk) > 0:
        print(f"   🚨 High-risk patterns: {len(high_risk)}")
        print("   These patterns need immediate attention!")
    
    # Volume surge patterns
    if 'Volume_Change' in pattern_analysis_df.columns:
        volume_surge = pattern_analysis_df[pattern_analysis_df['Volume_Change'] > 50]
        if len(volume_surge) > 0:
            print(f"   📈 Volume surge patterns: {len(volume_surge)}")
            print("   These patterns increased significantly in Post period")
    
    return pattern_analysis_df

# Run pattern analysis
if 'pattern_analysis_df' in locals() and len(pattern_analysis_df) > 0:
    analyzed_patterns = analyze_discovered_patterns(pattern_analysis_df, negation_df_clustered)
else:
    print("⚠️ Please run pattern clustering first.")

# =============================================================================
# CELL 7: Visualization of Discovered Patterns
# =============================================================================

def visualize_pattern_analysis(pattern_analysis_df, negation_df):
    """Create visualizations for the discovered patterns"""
    
    print("\n" + "="*60)
    print("STEP 4: PATTERN VISUALIZATION")
    print("="*60)
    
    if len(pattern_analysis_df) == 0:
        print("❌ No patterns to visualize!")
        return
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Pattern Volume Distribution
    axes[0,0].bar(pattern_analysis_df['Cluster_ID'], pattern_analysis_df['Total_Count'], 
                 alpha=0.7, color='skyblue')
    axes[0,0].set_xlabel('Pattern Cluster')
    axes[0,0].set_ylabel('Number of Instances')
    axes[0,0].set_title('Pattern Volume Distribution')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. TP vs FP Rates
    x = range(len(pattern_analysis_df))
    width = 0.35
    
    axes[0,1].bar([i - width/2 for i in x], pattern_analysis_df['TP_Rate'], 
                 width, label='TP Rate', alpha=0.8, color='green')
    axes[0,1].bar([i + width/2 for i in x], pattern_analysis_df['FP_Rate'], 
                 width, label='FP Rate', alpha=0.8, color='red')
    axes[0,1].set_xlabel('Pattern Cluster')
    axes[0,1].set_ylabel('Rate')
    axes[0,1].set_title('True Positive vs False Positive Rates')
    axes[0,1].legend()
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(pattern_analysis_df['Cluster_ID'])
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Pre vs Post Volume (if available)
    if 'Pre_Count' in pattern_analysis_df.columns and 'Post_Count' in pattern_analysis_df.columns:
        axes[1,0].bar([i - width/2 for i in x], pattern_analysis_df['Pre_Count'], 
                     width, label='Pre Period', alpha=0.8, color='blue')
        axes[1,0].bar([i + width/2 for i in x], pattern_analysis_df['Post_Count'], 
                     width, label='Post Period', alpha=0.8, color='orange')
        axes[1,0].set_xlabel('Pattern Cluster')
        axes[1,0].set_ylabel('Count')
        axes[1,0].set_title('Pre vs Post Period Volume')
        axes[1,0].legend()
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(pattern_analysis_df['Cluster_ID'])
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Quality Score (TP Rate - FP Rate)
    colors = ['red' if x < 0 else 'green' for x in pattern_analysis_df['Quality_Score']]
    axes[1,1].bar(pattern_analysis_df['Cluster_ID'], pattern_analysis_df['Quality_Score'], 
                 color=colors, alpha=0.7)
    axes[1,1].set_xlabel('Pattern Cluster')
    axes[1,1].set_ylabel('Quality Score (TP Rate - FP Rate)')
    axes[1,1].set_title('Pattern Quality Assessment')
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Pattern characteristics
    print("\n📊 PATTERN CHARACTERISTICS:")
    
    # Create a summary table
    summary_data = []
    for _, pattern in pattern_analysis_df.iterrows():
        summary_data.append({
            'Pattern': f"Cluster {pattern['Cluster_ID']}",
            'Volume': pattern['Total_Count'],
            'TP Rate': f"{pattern['TP_Rate']:.3f}",
            'FP Rate': f"{pattern['FP_Rate']:.3f}",
            'Quality': f"{pattern['Quality_Score']:+.3f}",
            'Key Terms': pattern['Top_Features'][:50] + "..." if len(pattern['Top_Features']) > 50 else pattern['Top_Features']
        })
    
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)

# Create visualizations
if 'pattern_analysis_df' in locals() and len(pattern_analysis_df) > 0:
    visualize_pattern_analysis(pattern_analysis_df, negation_df_clustered)
else:
    print("⚠️ Please run pattern discovery and clustering first.")

# =============================================================================
# CELL 8: Save Results and Next Steps
# =============================================================================

def save_notebook_results(df_prepared, negation_df, pattern_analysis_df):
    """Save all results from notebook analysis"""
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    import os
    
    # Create output directory
    os.makedirs('notebook_results', exist_ok=True)
    
    try:
        # Save to Excel
        with pd.ExcelWriter('notebook_results/Dynamic_Negation_Analysis_Results.xlsx') as writer:
            # Main datasets
            if df_prepared is not None:
                df_prepared.to_excel(writer, sheet_name='Prepared_Data', index=False)
            
            if len(negation_df) > 0:
                negation_df.to_excel(writer, sheet_name='Negation_Contexts', index=False)
            
            if len(pattern_analysis_df) > 0:
                pattern_analysis_df.to_excel(writer, sheet_name='Discovered_Patterns', index=False)
            
            # Sample contexts for each pattern
            if len(pattern_analysis_df) > 0 and len(negation_df) > 0:
                for _, pattern in pattern_analysis_df.iterrows():
                    cluster_id = pattern['Cluster_ID']
                    cluster_samples = negation_df[negation_df['Pattern_Cluster'] == cluster_id]
                    if len(cluster_samples) > 0:
                        sample_data = cluster_samples.head(20)[['UUID', 'Primary_Marker', 'Speaker', 'Negation_Word', 'Context']]
                        sample_data.to_excel(writer, sheet_name=f'Pattern_{cluster_id}_Samples', index=False)
        
        print("✅ Excel file saved: notebook_results/Dynamic_Negation_Analysis_Results.xlsx")
        
        # Save key results as pickle for quick loading
        if len(negation_df) > 0:
            negation_df.to_pickle('notebook_results/negation_contexts.pkl')
            print("✅ Negation contexts saved: notebook_results/negation_contexts.pkl")
        
        if len(pattern_analysis_df) > 0:
            pattern_analysis_df.to_pickle('notebook_results/pattern_analysis.pkl')
            print("✅ Pattern analysis saved: notebook_results/pattern_analysis.pkl")
        
        print(f"\n📁 All results saved to: notebook_results/")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def display_final_summary(pattern_analysis_df, negation_df):
    """Display final summary of the analysis"""
    
    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY - KEY FINDINGS")
    print("="*60)
    
    if len(pattern_analysis_df) == 0:
        print("❌ No patterns discovered - analysis incomplete")
        return
    
    # Overall statistics
    total_negations = len(negation_df)
    total_patterns = len(pattern_analysis_df)
    avg_tp_rate = pattern_analysis_df['TP_Rate'].mean()
    avg_fp_rate = pattern_analysis_df['FP_Rate'].mean()
    
    print(f"📊 ANALYSIS OVERVIEW:")
    print(f"   Total negation instances analyzed: {total_negations:,}")
    print(f"   Patterns discovered: {total_patterns}")
    print(f"   Average TP rate across patterns: {avg_tp_rate:.3f}")
    print(f"   Average FP rate across patterns: {avg_fp_rate:.3f}")
    
    # Identify key insights
    print(f"\n🔍 KEY INSIGHTS:")
    
    # High-risk patterns
    high_risk = pattern_analysis_df[
        (pattern_analysis_df['FP_Rate'] > 0.4) & 
        (pattern_analysis_df['Total_Count'] > 20)
    ]
    
    if len(high_risk) > 0:
        print(f"   🚨 HIGH-RISK PATTERNS: {len(high_risk)} patterns need immediate attention")
        for _, pattern in high_risk.iterrows():
            print(f"      Pattern {pattern['Cluster_ID']}: {pattern['FP_Rate']:.1%} FP rate, {pattern['Total_Count']} instances")
    
    # Volume changes
    if 'Volume_Change' in pattern_analysis_df.columns:
        volume_increases = pattern_analysis_df[pattern_analysis_df['Volume_Change'] > 50]
        if len(volume_increases) > 0:
            print(f"   📈 VOLUME SURGES: {len(volume_increases)} patterns increased significantly")
    
    # Best performing patterns
    best_patterns = pattern_analysis_df[pattern_analysis_df['Quality_Score'] > 0.5]
    if len(best_patterns) > 0:
        print(f"   ✅ GOOD PATTERNS: {len(best_patterns)} patterns with high quality scores")
    
    # Worst performing patterns
    worst_patterns = pattern_analysis_df[pattern_analysis_df['Quality_Score'] < -0.2]
    if len(worst_patterns) > 0:
        print(f"   ⚠️  PROBLEMATIC PATTERNS: {len(worst_patterns)} patterns with poor performance")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Focus on high-risk patterns for rule refinement")
    print(f"   2. Investigate volume surge patterns for root causes") 
    print(f"   3. Run Step 3 (Complaint Lexicon Mapping) for deeper insights")
    print(f"   4. Create action plan based on pattern characteristics")
    
    # Show top 3 patterns by volume for detailed review
    print(f"\n📋 TOP 3 PATTERNS BY VOLUME (for detailed review):")
    top_3 = pattern_analysis_df.nlargest(3, 'Total_Count')
    for i, (_, pattern) in enumerate(top_3.iterrows(), 1):
        print(f"   {i}. Pattern {pattern['Cluster_ID']}: {pattern['Total_Count']:,} instances")
        print(f"      Performance: TP={pattern['TP_Rate']:.3f}, FP={pattern['FP_Rate']:.3f}")
        print(f"      Key terms: {pattern['Top_Features'][:60]}...")

# Save results and show summary
if 'df_prepared' in locals():
    save_notebook_results(
        df_prepared if 'df_prepared' in locals() else None,
        negation_df_clustered if 'negation_df_clustered' in locals() else pd.DataFrame(),
        pattern_analysis_df if 'pattern_analysis_df' in locals() else pd.DataFrame()
    )
    
    display_final_summary(
        pattern_analysis_df if 'pattern_analysis_df' in locals() else pd.DataFrame(),
        negation_df_clustered if 'negation_df_clustered' in locals() else pd.DataFrame()
    )
else:
    print("⚠️ Please run all previous cells to generate results.")

print(f"\n" + "="*60)
print("🎉 NOTEBOOK ANALYSIS COMPLETED!")
print("="*60)
print("This notebook discovered negation patterns from your data without hardcoded rules.")
print("Review the results and run the next steps for complete analysis.")

# =============================================================================
# CELL 9: Quick Data Quality Check (Optional)
# =============================================================================

def quick_data_quality_check(df_prepared, negation_df, pattern_analysis_df):
    """Perform a quick quality check on the analysis results"""
    
    print("\n" + "="*60)
    print("🔍 DATA QUALITY CHECK")
    print("="*60)
    
    issues = []
    warnings = []
    
    # Check 1: Data volume
    if len(df_prepared) < 1000:
        warnings.append(f"Small dataset: {len(df_prepared)} records (recommend >1000)")
    
    # Check 2: Period balance
    period_counts = df_prepared['Period'].value_counts()
    if period_counts.get('Pre', 0) == 0:
        issues.append("No Pre period data found")
    if period_counts.get('Post', 0) == 0:
        issues.append("No Post period data found")
    
    # Check 3: TP/FP balance
    marker_counts = df_prepared['Primary Marker'].value_counts()
    tp_ratio = marker_counts.get('TP', 0) / len(df_prepared)
    if tp_ratio < 0.3 or tp_ratio > 0.9:
        warnings.append(f"Imbalanced TP/FP ratio: {tp_ratio:.2%} TPs")
    
    # Check 4: Negation discovery
    if len(negation_df) == 0:
        issues.append("No negation contexts discovered")
    elif len(negation_df) < 100:
        warnings.append(f"Few negation contexts: {len(negation_df)} (recommend >100)")
    
    # Check 5: Pattern discovery
    if len(pattern_analysis_df) == 0:
        issues.append("No patterns discovered")
    elif len(pattern_analysis_df) < 3:
        warnings.append(f"Few patterns: {len(pattern_analysis_df)} (recommend 3+)")
    
    # Check 6: Pattern quality
    if len(pattern_analysis_df) > 0:
        avg_quality = pattern_analysis_df['Quality_Score'].mean()
        if avg_quality < 0:
            warnings.append(f"Poor average pattern quality: {avg_quality:.3f}")
    
    # Report results
    if len(issues) == 0 and len(warnings) == 0:
        print("✅ QUALITY CHECK PASSED")
        print("Your analysis results look good for further processing!")
    else:
        if len(issues) > 0:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   - {issue}")
        
        if len(warnings) > 0:
            print("⚠️ WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("\nRECOMMENDATIONS:")
        if len(issues) > 0:
            print("- Address the issues above before proceeding")
        print("- Consider collecting more data if volumes are low")
        print("- Check date ranges for Pre/Post periods")
        print("- Verify transcript quality and completeness")

# Run quality check if data is available
if all(var in locals() for var in ['df_prepared', 'negation_df_clustered', 'pattern_analysis_df']):
    quick_data_quality_check(df_prepared, negation_df_clustered, pattern_analysis_df)
else:
    print("⚠️ Cannot run quality check - please complete previous analysis steps.")

# =============================================================================
# CELL 10: What's Next - Step 3 and Beyond
# =============================================================================

print("\n" + "="*60)
print("🚀 WHAT'S NEXT - CONTINUING THE ANALYSIS")
print("="*60)

print("""
✅ COMPLETED IN THIS NOTEBOOK:
   1. ✓ Environment setup and data loading
   2. ✓ Data preparation with Pre/Post periods
   3. ✓ Dynamic negation context discovery (no hardcoded patterns!)
   4. ✓ Pattern clustering and analysis
   5. ✓ Pattern performance evaluation
   6. ✓ Results visualization and saving

🔄 NEXT STEPS - Run the Full Modules:

STEP 3: Complaint Lexicon Mapping
   → Run: Part2_complaint_lexicon_mapping.py
   → Creates dynamic complaint vs non-complaint lexicons
   → Identifies problematic expressions requiring rule fixes

STEP 4: Advanced Visualizations  
   → Run: Part3_advanced_visualization.py
   → Creates interactive dashboards and comprehensive analytics
   → Generates prioritized action items

STEP 5: Complete Integration
   → Run: Part4_main_integration_script.py
   → Combines all analyses into comprehensive report
   → Exports final Excel with all findings

📁 YOUR RESULTS ARE SAVED TO:
   - notebook_results/Dynamic_Negation_Analysis_Results.xlsx
   - notebook_results/negation_contexts.pkl
   - notebook_results/pattern_analysis.pkl

🎯 IMMEDIATE ACTIONS FROM THIS ANALYSIS:
""")

# Show immediate actions based on discovered patterns
if 'pattern_analysis_df' in locals() and len(pattern_analysis_df) > 0:
    
    # Find patterns requiring immediate attention
    immediate_actions = []
    
    high_risk = pattern_analysis_df[
        (pattern_analysis_df['FP_Rate'] > 0.4) & 
        (pattern_analysis_df['Total_Count'] > 20)
    ]
    
    for _, pattern in high_risk.iterrows():
        immediate_actions.append(
            f"   → Review Pattern {pattern['Cluster_ID']}: {pattern['FP_Rate']:.1%} FP rate, "
            f"key terms: {pattern['Top_Features'][:40]}..."
        )
    
    if immediate_actions:
        print("🚨 HIGH PRIORITY:")
        for action in immediate_actions[:3]:  # Show top 3
            print(action)
    
    # Show volume surge patterns
    if 'Volume_Change' in pattern_analysis_df.columns:
        volume_surge = pattern_analysis_df[pattern_analysis_df['Volume_Change'] > 50]
        if len(volume_surge) > 0:
            print("\n📈 INVESTIGATE VOLUME CHANGES:")
            for _, pattern in volume_surge.head(2).iterrows():
                print(f"   → Pattern {pattern['Cluster_ID']}: +{pattern['Volume_Change']} instances in Post period")

print(f"""
💡 KEY INSIGHTS FROM YOUR DATA:
   - Discovered {len(pattern_analysis_df) if 'pattern_analysis_df' in locals() else 0} distinct negation patterns
   - No hardcoded assumptions - all patterns learned from your actual data
   - Ready for deeper lexicon analysis and actionable recommendations

🔧 TO CONTINUE:
   1. Review the Excel file with discovered patterns
   2. Run the remaining modules (Parts 2-4) for complete analysis  
   3. Focus on high-risk patterns for immediate rule improvements
""")

print("\n" + "="*60)
print("📝 NOTEBOOK ANALYSIS COMPLETE!")
print("="*60)
