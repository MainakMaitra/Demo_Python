# BERT-based Contrast Analysis for True Positives vs False Positives

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing imports
from string import punctuation as punct
import string

# NLTK imports
try:
    import nltk
    from nltk.corpus import stopwords
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    print("Stopwords loaded from NLTK")
except ImportError:
    print("Warning: NLTK not found, using basic stopwords")
    STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
                 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
                 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 
                 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                 'under', 'again', 'further', 'then', 'once'}

# BERT and ML imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
    import seaborn as sns
    from datetime import datetime
    print("BERT and analysis libraries imported successfully")
except ImportError as e:
    print(f"Missing required libraries: {e}")
    print("Please install: pip install sentence-transformers matplotlib seaborn nltk")
    exit()


# LIME imports
try:
    from lime.lime_text import LimeTextExplainer
    from lime import lime_text
    print("LIME library imported successfully")
except ImportError:
    print("LIME not found. Install with: pip install lime")
    exit()

# Additional ML imports for LIME classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Additional utilities
import json
import pickle
from collections import defaultdict, Counter

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
remove_digits = str.maketrans('', '', string.digits)

def clean_text(text):
    """
    Comprehensive text cleaning function
    
    Args:
        text: a string
        
    Returns:
        modified initial string
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    text = text.lower() # lowercase text
    text = text.encode(encoding="ascii", errors="ignore")
    text = text.decode()
    text = " ".join([word for word in text.split()])
    text = re.sub("@\S+", "", text) # removing mentions
    text = re.sub("https?:\/\/.*[\r\n]*", "", text) # remove urls
    text = re.sub("#", "", text) # remove hashtags
    text = "".join([ch for ch in text if ch not in punct]) # remove punctuation
    text = text.translate(remove_digits) # remove digits
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text) # remove symbols which are in BAD_SYMBOLS_RE from text
    text = text.replace('_', ' ')
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'[\W\_]', ' ', text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = text.strip()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    text = re.compile(r"(.)\1{2,}").sub(r"\1\1", text) # Fixing Word Lengthening
    text = ' '.join(word for word in text.split() if len(word)>2)
    return text

def bert_contrast_analysis(data_file_path, output_dir='bert_analysis_results'):
    """
    BERT-based contrast analysis between True Positives and False Positives
    Optimized for Dell Latitude 5450 system specifications
    
    Args:
        data_file_path: str, path to your Excel file
        output_dir: str, directory to save results
        
    Returns:
        dict: Analysis results including embeddings, clusters, and insights
    """
    
    print("="*80)
    print("BERT-BASED CONTRAST ANALYSIS: TRUE POSITIVES vs FALSE POSITIVES")
    print("="*80)
    print("System: Dell Latitude 5450 | Intel Ultra 7 155U | 32GB RAM")
    print("Optimizations: CPU-based BERT, batch processing, memory management")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    try:
        df = pd.read_excel(data_file_path)
        print(f"Data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"File not found: {data_file_path}")
        return None
    
    # Clean column names
    df.columns = df.columns.str.rstrip()
    
    # Prepare customer transcripts
    df['Customer_Transcript_Raw'] = df['Customer Transcript'].fillna('').astype(str)
    
    # Apply text cleaning
    print("Cleaning customer transcripts...")
    df['Customer_Transcript_Clean'] = df['Customer_Transcript_Raw'].apply(clean_text)
    
    # Filter out empty transcripts after cleaning
    df = df[df['Customer_Transcript_Clean'].str.len() > 10].reset_index(drop=True)
    print(f"{len(df)} valid transcripts after cleaning")
    
    # Create True Positive and False Positive labels
    df['Is_TP'] = (df['Primary Marker'] == 'TP').astype(int)
    df['Is_FP'] = (df['Primary Marker'] == 'FP').astype(int)
    
    tp_count = df['Is_TP'].sum()
    fp_count = df['Is_FP'].sum()
    
    print(f"True Positives: {tp_count}")
    print(f"False Positives: {fp_count}")
    
    if tp_count < 10 or fp_count < 10:
        print("Insufficient data for meaningful analysis (need at least 10 of each)")
        return None
    
    # Step 2: Load BERT model (optimized for CPU)
    print("\nStep 2: Loading BERT model...")
    
    # Use a lighter BERT model optimized for CPU inference
    model_name = 'all-MiniLM-L6-v2'  # Fast and efficient for CPU
    try:
        bert_model = SentenceTransformer(model_name)
        print(f"BERT model loaded: {model_name}")
        print("   - Optimized for CPU inference")
        print("   - 384-dimensional embeddings")
        print("   - Good balance of speed and quality")
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None
    
    # Step 3: Generate embeddings in batches (memory optimization)
    print("\nStep 3: Generating BERT embeddings...")
    
    batch_size = 32  # Optimized for 32GB RAM
    texts = df['Customer_Transcript_Clean'].tolist()
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}...")
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = bert_model.encode(batch_texts, 
                                           batch_size=batch_size,
                                           show_progress_bar=True if i == 0 else False,
                                           normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {i + len(batch_texts)} / {len(texts)} texts")
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Add embeddings to dataframe
    df['bert_embedding'] = [emb for emb in embeddings]
    
    # Step 4: Contrast Analysis
    print("\nStep 4: Running contrast analysis...")
    
    # Separate TP and FP embeddings
    tp_mask = df['Is_TP'] == 1
    fp_mask = df['Is_FP'] == 1
    
    tp_embeddings = embeddings[tp_mask]
    fp_embeddings = embeddings[fp_mask]
    
    tp_texts = df[tp_mask]['Customer_Transcript_Clean'].tolist()
    fp_texts = df[fp_mask]['Customer_Transcript_Clean'].tolist()
    
    print(f"TP embeddings: {tp_embeddings.shape}")
    print(f"FP embeddings: {fp_embeddings.shape}")
    
    # Calculate centroids
    tp_centroid = np.mean(tp_embeddings, axis=0)
    fp_centroid = np.mean(fp_embeddings, axis=0)
    
    # Calculate centroid similarity
    centroid_similarity = cosine_similarity([tp_centroid], [fp_centroid])[0][0]
    print(f"TP-FP centroid similarity: {centroid_similarity:.4f}")
    
    # Step 5: Dimensionality reduction for visualization
    print("\nStep 5: Dimensionality reduction and clustering...")
    
    # PCA for visualization (2D and 3D)
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    
    embeddings_2d = pca_2d.fit_transform(embeddings)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    # Add PCA coordinates to dataframe
    df['pca_x'] = embeddings_2d[:, 0]
    df['pca_y'] = embeddings_2d[:, 1]
    df['pca_z'] = embeddings_3d[:, 2]
    
    print(f"PCA 2D explained variance: {pca_2d.explained_variance_ratio_.sum():.3f}")
    print(f"PCA 3D explained variance: {pca_3d.explained_variance_ratio_.sum():.3f}")
    
    # Step 6: Keyword analysis using TF-IDF contrast
    print("\nStep 6: Keyword analysis for FP inclination...")
    
    # TF-IDF analysis for distinctive words
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    
    # Fit on all texts
    all_clean_texts = df['Customer_Transcript_Clean'].tolist()
    tfidf_matrix = tfidf.fit_transform(all_clean_texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Calculate mean TF-IDF scores for TP and FP
    tp_tfidf_mean = np.mean(tfidf_matrix[tp_mask].toarray(), axis=0)
    fp_tfidf_mean = np.mean(tfidf_matrix[fp_mask].toarray(), axis=0)
    
    # Calculate FP inclination score (FP_score - TP_score)
    fp_inclination = fp_tfidf_mean - tp_tfidf_mean
    
    # Create keyword analysis dataframe
    keyword_analysis = pd.DataFrame({
        'keyword': feature_names,
        'tp_score': tp_tfidf_mean,
        'fp_score': fp_tfidf_mean,
        'fp_inclination': fp_inclination,
        'fp_ratio': np.where(tp_tfidf_mean > 0, fp_tfidf_mean / tp_tfidf_mean, np.inf)
    })
    
    # Sort by FP inclination
    keyword_analysis = keyword_analysis.sort_values('fp_inclination', ascending=False)
    
    print("Top 10 keywords most inclined toward False Positives:")
    for i, (_, row) in enumerate(keyword_analysis.head(10).iterrows(), 1):
        print(f"   {i:2d}. '{row['keyword']}' | FP: {row['fp_score']:.4f} | TP: {row['tp_score']:.4f} | Diff: {row['fp_inclination']:+.4f}")
    
    # Step 7: Clustering analysis
    print("\nStep 7: Clustering analysis...")
    
    # K-means clustering on embeddings
    n_clusters = min(8, len(df) // 10)  # Adaptive cluster count
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    df['cluster'] = clusters
    
    # Analyze cluster composition
    cluster_analysis = df.groupby('cluster').agg({
        'Is_TP': ['count', 'sum', 'mean'],
        'Is_FP': ['count', 'sum', 'mean']
    }).round(3)
    
    cluster_analysis.columns = ['total_count', 'tp_count', 'tp_ratio', 'fp_total', 'fp_count', 'fp_ratio']
    cluster_analysis = cluster_analysis[['total_count', 'tp_count', 'tp_ratio', 'fp_count', 'fp_ratio']]
    
    print("Cluster composition analysis:")
    print(cluster_analysis)
    
    # Step 8: Export results
    print(f"\nStep 8: Exporting results to {output_dir}...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export main results
    results_file = os.path.join(output_dir, f'bert_contrast_analysis_{timestamp}.xlsx')
    
    with pd.ExcelWriter(results_file, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Transcripts',
                'True Positives',
                'False Positives',
                'TP-FP Centroid Similarity',
                'PCA 2D Explained Variance',
                'Number of Clusters',
                'Top FP Keyword',
                'Top FP Keyword Score'
            ],
            'Value': [
                len(df),
                tp_count,
                fp_count,
                f"{centroid_similarity:.4f}",
                f"{pca_2d.explained_variance_ratio_.sum():.3f}",
                n_clusters,
                keyword_analysis.iloc[0]['keyword'],
                f"{keyword_analysis.iloc[0]['fp_inclination']:.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Keyword analysis
        keyword_analysis.head(50).to_excel(writer, sheet_name='FP_Keywords', index=False)
        
        # Cluster analysis
        cluster_analysis.to_excel(writer, sheet_name='Cluster_Analysis')
        
        # Sample texts by cluster
        for cluster_id in range(min(5, n_clusters)):  # Top 5 clusters
            cluster_samples = df[df['cluster'] == cluster_id][
                ['Customer_Transcript_Clean', 'Primary Marker', 'Is_TP', 'Is_FP']
            ].head(10)
            cluster_samples.to_excel(writer, sheet_name=f'Cluster_{cluster_id}_Samples', index=False)
    
    print(f"Results exported to: {results_file}")
    
    # Create visualization
    create_contrast_visualization(df, keyword_analysis, cluster_analysis, output_dir, timestamp)
    
    # Step 9: Generate insights
    insights = generate_contrast_insights(df, keyword_analysis, cluster_analysis, 
                                        centroid_similarity, tp_count, fp_count)
    
    # Export insights
    insights_file = os.path.join(output_dir, f'bert_insights_{timestamp}.txt')
    with open(insights_file, 'w') as f:
        f.write(insights)
    
    print(f"Insights exported to: {insights_file}")
    
    # Return comprehensive results
    return {
        'dataframe': df,
        'embeddings': embeddings,
        'keyword_analysis': keyword_analysis,
        'cluster_analysis': cluster_analysis,
        'centroid_similarity': centroid_similarity,
        'insights': insights,
        'model': bert_model
    }

def create_contrast_visualization(df, keyword_analysis, cluster_analysis, output_dir, timestamp):
    """Create visualizations for the contrast analysis"""
    
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('BERT-based TP vs FP Contrast Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: PCA scatter plot
    ax1 = axes[0, 0]
    tp_mask = df['Is_TP'] == 1
    fp_mask = df['Is_FP'] == 1
    
    ax1.scatter(df[tp_mask]['pca_x'], df[tp_mask]['pca_y'], 
               alpha=0.6, label='True Positives', s=50)
    ax1.scatter(df[fp_mask]['pca_x'], df[fp_mask]['pca_y'], 
               alpha=0.6, label='False Positives', s=50)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title('TP vs FP Distribution (PCA)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top FP keywords
    ax2 = axes[0, 1]
    top_fp_keywords = keyword_analysis.head(10)
    bars = ax2.barh(range(len(top_fp_keywords)), top_fp_keywords['fp_inclination'])
    ax2.set_yticks(range(len(top_fp_keywords)))
    ax2.set_yticklabels(top_fp_keywords['keyword'], fontsize=10)
    ax2.set_xlabel('FP Inclination Score')
    ax2.set_title('Top Keywords Inclined toward False Positives')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Color bars by value
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.Reds(0.5 + 0.5 * i / len(bars)))
    
    # Plot 3: Cluster composition
    ax3 = axes[1, 0]
    cluster_data = cluster_analysis[['tp_ratio', 'fp_ratio']].reset_index()
    x_pos = np.arange(len(cluster_data))
    
    ax3.bar(x_pos - 0.2, cluster_data['tp_ratio'], 0.4, label='TP Ratio', alpha=0.8)
    ax3.bar(x_pos + 0.2, cluster_data['fp_ratio'], 0.4, label='FP Ratio', alpha=0.8)
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Ratio')
    ax3.set_title('TP vs FP Ratios by Cluster')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cluster_data['cluster'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Keyword comparison scatter
    ax4 = axes[1, 1]
    keyword_subset = keyword_analysis.head(20)  # Top 20 for readability
    scatter = ax4.scatter(keyword_subset['tp_score'], keyword_subset['fp_score'], 
                         alpha=0.7, s=60, c=keyword_subset['fp_inclination'], 
                         cmap='RdYlBu_r')
    
    # Add diagonal line
    max_val = max(keyword_subset['tp_score'].max(), keyword_subset['fp_score'].max())
    ax4.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal scores')
    
    ax4.set_xlabel('TP Score (TF-IDF)')
    ax4.set_ylabel('FP Score (TF-IDF)')
    ax4.set_title('Keyword Scores: TP vs FP')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('FP Inclination')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'bert_contrast_visualization_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    plt.close()

def generate_contrast_insights(df, keyword_analysis, cluster_analysis, 
                             centroid_similarity, tp_count, fp_count):
    """Generate textual insights from the analysis"""
    
    insights = []
    insights.append("BERT-BASED CONTRAST ANALYSIS INSIGHTS")
    insights.append("=" * 50)
    insights.append("")
    
    # Overall statistics
    insights.append("DATASET OVERVIEW:")
    insights.append(f"   • Total transcripts analyzed: {len(df)}")
    insights.append(f"   • True Positives: {tp_count} ({tp_count/len(df)*100:.1f}%)")
    insights.append(f"   • False Positives: {fp_count} ({fp_count/len(df)*100:.1f}%)")
    insights.append(f"   • TP-FP centroid similarity: {centroid_similarity:.4f}")
    insights.append("")
    
    # Similarity interpretation
    if centroid_similarity > 0.8:
        sim_interpretation = "Very high - TP and FP are semantically very similar"
    elif centroid_similarity > 0.6:
        sim_interpretation = "High - TP and FP share significant semantic overlap"
    elif centroid_similarity > 0.4:
        sim_interpretation = "Moderate - Some semantic differences between TP and FP"
    else:
        sim_interpretation = "Low - TP and FP are semantically distinct"
    
    insights.append(f"SEMANTIC SIMILARITY ANALYSIS:")
    insights.append(f"   • Interpretation: {sim_interpretation}")
    if centroid_similarity > 0.7:
        insights.append("   • Recommendation: Focus on subtle linguistic differences")
        insights.append("   • Challenge: High semantic overlap makes classification difficult")
    insights.append("")
    
    # Top FP keywords
    insights.append("TOP KEYWORDS CAUSING FALSE POSITIVES:")
    top_fp_keywords = keyword_analysis.head(10)
    for i, (_, row) in enumerate(top_fp_keywords.iterrows(), 1):
        keyword = row['keyword']
        fp_incl = row['fp_inclination']
        insights.append(f"   {i:2d}. '{keyword}' (FP inclination: {fp_incl:+.4f})")
    insights.append("")
    
    # Clustering insights
    insights.append("CLUSTERING INSIGHTS:")
    for cluster_id, row in cluster_analysis.iterrows():
        if row['total_count'] >= 5:  # Only analyze meaningful clusters
            if row['fp_ratio'] > 0.7:
                cluster_type = "FP-dominated"
            elif row['tp_ratio'] > 0.7:
                cluster_type = "TP-dominated"
            else:
                cluster_type = "Mixed"
            
            insights.append(f"   • Cluster {cluster_id}: {cluster_type} ({int(row['total_count'])} samples)")
            insights.append(f"     - TP ratio: {row['tp_ratio']:.3f} | FP ratio: {row['fp_ratio']:.3f}")
    insights.append("")
    
    # Actionable recommendations
    insights.append("ACTIONABLE RECOMMENDATIONS:")
    insights.append("")
    
    # Based on top FP keywords
    critical_fp_keywords = keyword_analysis[keyword_analysis['fp_inclination'] > 0.01].head(5)
    if len(critical_fp_keywords) > 0:
        insights.append("   KEYWORD-BASED RULES:")
        for _, row in critical_fp_keywords.iterrows():
            keyword = row['keyword']
            insights.append(f"   • Monitor usage of '{keyword}' - strong FP indicator")
            insights.append(f"     Implementation: Add context check for '{keyword}'")
        insights.append("")
    
    # Based on clustering
    fp_clusters = cluster_analysis[cluster_analysis['fp_ratio'] > 0.6]
    if len(fp_clusters) > 0:
        insights.append("   CLUSTER-BASED IMPROVEMENTS:")
        insights.append(f"   • {len(fp_clusters)} clusters are FP-dominated")
        insights.append("   • Consider separate handling for these text patterns")
        insights.append("   • Review manual labeling for these clusters")
        insights.append("")
    
    # System optimization
    insights.append("   SYSTEM OPTIMIZATION:")
    insights.append("   • Use BERT embeddings for semantic similarity checks")
    insights.append("   • Implement ensemble approach: rule-based + semantic")
    insights.append("   • Set up monitoring for FP-inclined keywords")
    insights.append("   • Regular re-clustering to detect new patterns")
    insights.append("")
    
    insights.append("=" * 50)
    insights.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(insights)
    
# Example usage function
def run_bert_contrast_analysis():
    """
    Example function to run the complete BERT contrast analysis
    """
    
    print("Starting BERT Contrast Analysis...")
    
    # Run the analysis
    data_file_path = 'Precision_Drop_Analysis_OG.xlsx'
    results = bert_contrast_analysis(data_file_path)
    
    if results:
        print("\n" + "="*60)
        print("BERT CONTRAST ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey findings:")
        print(f"• TP-FP Centroid Similarity: {results['centroid_similarity']:.4f}")
        print(f"• Top FP Keyword: '{results['keyword_analysis'].iloc[0]['keyword']}'")
        print(f"• FP Inclination Score: {results['keyword_analysis'].iloc[0]['fp_inclination']:+.4f}")
        
        return results
    else:
        print("Analysis failed")
        return Nonereturn results
    else:
        print("Analysis failed")
        return None


# if __name__ == "__main__":
#     # Run the BERT contrast analysis
#     result = run_bert_contrast_analysis()

def train_bert_classifier_for_lime(df, embeddings):
    """
    Train classifiers on BERT embeddings to predict TP vs FP
    This model will be explained by LIME
    
    Args:
        df: DataFrame with TP/FP labels
        embeddings: BERT embeddings array
        
    Returns:
        dict: Contains trained models, scores, and metadata
    """
    
    print("\nStep 8.5: Training BERT-based classifiers for LIME explanation...")
    
    # Prepare labels (1 for TP, 0 for FP)
    tp_fp_mask = (df['Primary Marker'] == 'TP') | (df['Primary Marker'] == 'FP')
    
    if tp_fp_mask.sum() < 20:
        print("Insufficient TP/FP samples for classifier training")
        return None
    
    # Filter data for TP/FP only
    filtered_df = df[tp_fp_mask].copy()
    filtered_embeddings = embeddings[tp_fp_mask]
    
    # Create binary labels (1 = TP, 0 = FP)
    labels = (filtered_df['Primary Marker'] == 'TP').astype(int)
    
    print(f"Training data: {len(filtered_df)} samples")
    print(f"  - True Positives: {labels.sum()}")
    print(f"  - False Positives: {(labels == 0).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        filtered_embeddings, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    # Scale embeddings for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple classifiers
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        'logistic_regression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        'svm': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for SVM, original for others
        X_train_use = X_train_scaled if name == 'svm' else X_train
        X_test_use = X_test_scaled if name == 'svm' else X_test
        
        # Train model
        clf.fit(X_train_use, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_use)
        y_prob = clf.predict_proba(X_test_use)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(clf, X_train_use, y_train, cv=5)
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Store results
        results[name] = {
            'model': clf,
            'scaler': scaler if name == 'svm' else None,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    # Select best model based on CV score
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model_info = results[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best CV score: {best_model_info['cv_mean']:.4f}")
    
    # Add metadata
    results['metadata'] = {
        'best_model': best_model_name,
        'filtered_df': filtered_df,
        'filtered_embeddings': filtered_embeddings,
        'train_indices': X_train,
        'test_indices': X_test,
        'feature_dim': filtered_embeddings.shape[1]
    }
    
    return results

def create_lime_prediction_function(model_info, bert_model, scaler=None):
    """
    Create prediction function for LIME that takes text input
    
    Args:
        model_info: Dictionary containing trained model
        bert_model: SentenceTransformer model
        scaler: StandardScaler if needed
        
    Returns:
        function: Prediction function for LIME
    """
    
    def predict_function(texts):
        """
        Prediction function for LIME
        Takes list of texts, returns probabilities
        """
        # Clean texts
        cleaned_texts = [clean_text(text) for text in texts]
        
        # Generate BERT embeddings
        embeddings = bert_model.encode(cleaned_texts, normalize_embeddings=True)
        
        # Scale if needed
        if scaler is not None:
            embeddings = scaler.transform(embeddings)
        
        # Get probabilities
        probabilities = model_info['model'].predict_proba(embeddings)
        
        return probabilities
    
    return predict_function

def generate_lime_explanations_bert(df, bert_model, classifier_results, output_dir, max_explanations=20):
    """
    Generate LIME explanations for representative TP and FP examples
    
    Args:
        df: DataFrame with labeled data
        bert_model: Trained BERT model
        classifier_results: Results from train_bert_classifier_for_lime
        output_dir: Directory to save explanations
        max_explanations: Maximum number of explanations per category
        
    Returns:
        dict: LIME explanation results
    """
    
    print("\nStep 9: Generating LIME explanations...")
    
    if classifier_results is None:
        print("No classifier results available for LIME")
        return None
    
    # Get best model info
    best_model_name = classifier_results['metadata']['best_model']
    best_model_info = classifier_results[best_model_name]
    filtered_df = classifier_results['metadata']['filtered_df']
    
    print(f"Using {best_model_name} for LIME explanations")
    print(f"Model accuracy: {best_model_info['accuracy']:.4f}")
    
    # Create prediction function for LIME
    predict_fn = create_lime_prediction_function(
        best_model_info, 
        bert_model, 
        best_model_info.get('scaler')
    )
    
    # Initialize LIME explainer
    explainer = LimeTextExplainer(
        class_names=['False Positive', 'True Positive'],
        mode='classification',
        random_state=42
    )
    
    # Select representative examples for explanation
    tp_samples = filtered_df[filtered_df['Primary Marker'] == 'TP'].copy()
    fp_samples = filtered_df[filtered_df['Primary Marker'] == 'FP'].copy()
    
    # Get model predictions for confidence-based selection
    all_embeddings = classifier_results['metadata']['filtered_embeddings']
    
    if best_model_info.get('scaler'):
        all_embeddings_scaled = best_model_info['scaler'].transform(all_embeddings)
        all_probabilities = best_model_info['model'].predict_proba(all_embeddings_scaled)
    else:
        all_probabilities = best_model_info['model'].predict_proba(all_embeddings)
    
    filtered_df['predicted_prob_tp'] = all_probabilities[:, 1]
    filtered_df['predicted_prob_fp'] = all_probabilities[:, 0]
    filtered_df['prediction_confidence'] = np.max(all_probabilities, axis=1)
    
    # Select examples for explanation
    explanation_samples = select_explanation_samples(filtered_df, max_explanations)
    
    print(f"Selected {len(explanation_samples)} samples for LIME explanation")
    
    # Generate explanations
    lime_results = {
        'explanations': [],
        'word_importance_summary': defaultdict(list),
        'sample_info': []
    }
    
    explanation_dir = os.path.join(output_dir, 'lime_explanations')
    os.makedirs(explanation_dir, exist_ok=True)
    
    for idx, (_, row) in enumerate(explanation_samples.iterrows()):
        try:
            text = row['Customer_Transcript_Raw']
            true_label = row['Primary Marker']
            predicted_prob = row['predicted_prob_tp']
            
            print(f"Explaining sample {idx+1}/{len(explanation_samples)}: {true_label}")
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                text,
                predict_fn,
                num_features=20,  # Top 20 most important words
                num_samples=300   # Reduced for memory optimization
            )
            
            # Extract word importance
            word_importance = dict(explanation.as_list())
            
            # Store explanation info
            sample_info = {
                'index': row.name,
                'true_label': true_label,
                'predicted_prob_tp': predicted_prob,
                'text_length': len(text.split()),
                'explanation': explanation,
                'word_importance': word_importance
            }
            
            lime_results['explanations'].append(explanation)
            lime_results['sample_info'].append(sample_info)
            
            # Aggregate word importance by true label
            for word, importance in word_importance.items():
                lime_results['word_importance_summary'][true_label].append({
                    'word': word,
                    'importance': importance,
                    'sample_idx': idx
                })
            
            # Save individual explanation as HTML
            html_file = os.path.join(explanation_dir, f'explanation_{idx+1}_{true_label}.html')
            explanation.save_to_file(html_file)
            
        except Exception as e:
            print(f"Error explaining sample {idx+1}: {e}")
            continue
    
    print(f"Generated {len(lime_results['explanations'])} LIME explanations")
    print(f"HTML files saved to: {explanation_dir}")
    
    return lime_results

def select_explanation_samples(df, max_explanations):
    """
    Select representative samples for LIME explanation
    
    Args:
        df: Filtered DataFrame
        max_explanations: Maximum explanations per category
        
    Returns:
        DataFrame: Selected samples
    """
    
    tp_samples = df[df['Primary Marker'] == 'TP'].copy()
    fp_samples = df[df['Primary Marker'] == 'FP'].copy()
    
    selected_samples = []
    
    # For each category, select diverse examples
    for samples, label in [(tp_samples, 'TP'), (fp_samples, 'FP')]:
        if len(samples) == 0:
            continue
            
        # Strategy: High confidence, medium confidence, low confidence examples
        samples_sorted = samples.sort_values('prediction_confidence', ascending=False)
        
        n_samples = min(max_explanations // 2, len(samples))
        
        if n_samples >= 3:
            # High confidence (top 40%)
            high_conf_n = max(1, n_samples // 3)
            high_conf = samples_sorted.head(int(len(samples) * 0.4)).sample(
                n=high_conf_n, random_state=42
            )
            
            # Medium confidence (middle 40%)
            mid_start = int(len(samples) * 0.3)
            mid_end = int(len(samples) * 0.7)
            mid_conf_n = max(1, n_samples // 3)
            mid_conf = samples_sorted.iloc[mid_start:mid_end].sample(
                n=min(mid_conf_n, mid_end - mid_start), random_state=42
            )
            
            # Low confidence (bottom 30%)
            low_conf_n = n_samples - high_conf_n - len(mid_conf)
            if low_conf_n > 0:
                low_conf = samples_sorted.tail(int(len(samples) * 0.3)).sample(
                    n=min(low_conf_n, int(len(samples) * 0.3)), random_state=42
                )
                category_samples = pd.concat([high_conf, mid_conf, low_conf])
            else:
                category_samples = pd.concat([high_conf, mid_conf])
        else:
            # If too few samples, just take random selection
            category_samples = samples.sample(n=n_samples, random_state=42)
        
        selected_samples.append(category_samples)
    
    return pd.concat(selected_samples) if selected_samples else pd.DataFrame()

def analyze_lime_patterns(lime_results, keyword_analysis):
    """
    Analyze LIME explanation patterns to extract insights
    
    Args:
        lime_results: Results from generate_lime_explanations_bert
        keyword_analysis: TF-IDF keyword analysis from original function
        
    Returns:
        dict: Aggregated LIME insights
    """
    
    print("\nStep 10: Analyzing LIME explanation patterns...")
    
    if not lime_results or len(lime_results['explanations']) == 0:
        print("No LIME results to analyze")
        return None
    
    # Aggregate word importance by label
    tp_word_importance = defaultdict(list)
    fp_word_importance = defaultdict(list)
    
    for sample in lime_results['sample_info']:
        true_label = sample['true_label']
        word_importance = sample['word_importance']
        
        target_dict = tp_word_importance if true_label == 'TP' else fp_word_importance
        
        for word, importance in word_importance.items():
            target_dict[word].append(importance)
    
    # Calculate aggregated statistics
    def aggregate_word_stats(word_dict):
        """Calculate mean, std, count for each word"""
        stats = {}
        for word, importance_list in word_dict.items():
            stats[word] = {
                'mean_importance': np.mean(importance_list),
                'std_importance': np.std(importance_list),
                'count': len(importance_list),
                'total_importance': sum(importance_list)
            }
        return stats
    
    tp_stats = aggregate_word_stats(tp_word_importance)
    fp_stats = aggregate_word_stats(fp_word_importance)
    
    # Create comprehensive word analysis
    all_words = set(tp_stats.keys()) | set(fp_stats.keys())
    
    lime_word_analysis = []
    for word in all_words:
        tp_info = tp_stats.get(word, {'mean_importance': 0, 'count': 0, 'total_importance': 0})
        fp_info = fp_stats.get(word, {'mean_importance': 0, 'count': 0, 'total_importance': 0})
        
        # Calculate difference metrics
        importance_diff = tp_info['mean_importance'] - fp_info['mean_importance']
        
        # LIME inclination (positive = TP inclination, negative = FP inclination)
        lime_inclination = importance_diff
        
        lime_word_analysis.append({
            'word': word,
            'tp_mean_importance': tp_info['mean_importance'],
            'fp_mean_importance': fp_info['mean_importance'],
            'tp_count': tp_info['count'],
            'fp_count': fp_info['count'],
            'total_count': tp_info['count'] + fp_info['count'],
            'lime_inclination': lime_inclination,
            'abs_importance_diff': abs(importance_diff)
        })
    
    # Convert to DataFrame and sort
    lime_word_df = pd.DataFrame(lime_word_analysis)
    lime_word_df = lime_word_df.sort_values('abs_importance_diff', ascending=False)
    
    print(f"Analyzed {len(lime_word_df)} unique words from LIME explanations")
    
    # Find words that drive FP classifications (negative LIME importance)
    fp_driving_words = lime_word_df[
        (lime_word_df['lime_inclination'] < -0.01) & 
        (lime_word_df['total_count'] >= 2)
    ].sort_values('lime_inclination').head(10)
    
    # Find words that drive TP classifications (positive LIME importance)
    tp_driving_words = lime_word_df[
        (lime_word_df['lime_inclination'] > 0.01) & 
        (lime_word_df['total_count'] >= 2)
    ].sort_values('lime_inclination', ascending=False).head(10)
    
    print(f"Found {len(fp_driving_words)} words driving FP classifications")
    print(f"Found {len(tp_driving_words)} words driving TP classifications")
    
    # Compare with TF-IDF analysis
    tfidf_comparison = compare_lime_with_tfidf(lime_word_df, keyword_analysis)
    
    # Generate actionable rules
    actionable_rules = generate_lime_rules(fp_driving_words, tp_driving_words, lime_results)
    
    return {
        'lime_word_analysis': lime_word_df,
        'fp_driving_words': fp_driving_words,
        'tp_driving_words': tp_driving_words,
        'tfidf_comparison': tfidf_comparison,
        'actionable_rules': actionable_rules,
        'summary_stats': {
            'total_words_analyzed': len(lime_word_df),
            'fp_driving_count': len(fp_driving_words),
            'tp_driving_count': len(tp_driving_words)
        }
    }

def compare_lime_with_tfidf(lime_word_df, keyword_analysis):
    """
    Compare LIME word importance with TF-IDF keyword analysis
    
    Args:
        lime_word_df: LIME word analysis DataFrame
        keyword_analysis: TF-IDF analysis from original function
        
    Returns:
        dict: Comparison results
    """
    
    print("Comparing LIME with TF-IDF analysis...")
    
    # Merge LIME and TF-IDF results
    tfidf_words = set(keyword_analysis['keyword'])
    lime_words = set(lime_word_df['word'])
    
    common_words = tfidf_words & lime_words
    lime_only_words = lime_words - tfidf_words
    tfidf_only_words = tfidf_words - lime_words
    
    print(f"Common words: {len(common_words)}")
    print(f"LIME-only words: {len(lime_only_words)}")
    print(f"TF-IDF-only words: {len(tfidf_only_words)}")
    
    # For common words, compare rankings
    comparison_data = []
    for word in common_words:
        lime_row = lime_word_df[lime_word_df['word'] == word].iloc[0]
        tfidf_row = keyword_analysis[keyword_analysis['keyword'] == word].iloc[0]
        
        comparison_data.append({
            'word': word,
            'lime_inclination': lime_row['lime_inclination'],
            'tfidf_fp_inclination': tfidf_row['fp_inclination'],
            'agreement': np.sign(lime_row['lime_inclination']) == np.sign(tfidf_row['fp_inclination']),
            'lime_rank': lime_word_df[lime_word_df['word'] == word].index[0],
            'tfidf_rank': keyword_analysis[keyword_analysis['keyword'] == word].index[0]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        agreement_rate = comparison_df['agreement'].mean()
        print(f"LIME-TF-IDF agreement rate: {agreement_rate:.3f}")
    else:
        agreement_rate = 0
    
    return {
        'comparison_df': comparison_df,
        'common_words': list(common_words),
        'lime_only_words': list(lime_only_words),
        'tfidf_only_words': list(tfidf_only_words),
        'agreement_rate': agreement_rate
    }

def generate_lime_rules(fp_driving_words, tp_driving_words, lime_results):
    """
    Generate actionable rules based on LIME analysis
    
    Args:
        fp_driving_words: Words that drive FP classifications
        tp_driving_words: Words that drive TP classifications
        lime_results: Full LIME results
        
    Returns:
        dict: Actionable rules and recommendations
    """
    
    rules = {
        'fp_prevention_rules': [],
        'tp_enhancement_rules': [],
        'context_rules': [],
        'threshold_recommendations': {}
    }
    
    # FP Prevention Rules
    for _, word_info in fp_driving_words.iterrows():
        word = word_info['word']
        importance = word_info['lime_inclination']
        count = word_info['total_count']
        
        rule = {
            'word': word,
            'rule_type': 'FP_PREVENTION',
            'importance_score': abs(importance),
            'confidence': 'HIGH' if count >= 3 else 'MEDIUM',
            'recommendation': f"Flag transcripts containing '{word}' for manual review",
            'threshold': f"LIME importance < {importance:.3f}"
        }
        rules['fp_prevention_rules'].append(rule)
    
    # TP Enhancement Rules
    for _, word_info in tp_driving_words.iterrows():
        word = word_info['word']
        importance = word_info['lime_inclination']
        count = word_info['total_count']
        
        rule = {
            'word': word,
            'rule_type': 'TP_ENHANCEMENT',
            'importance_score': importance,
            'confidence': 'HIGH' if count >= 3 else 'MEDIUM',
            'recommendation': f"Boost confidence for transcripts containing '{word}'",
            'threshold': f"LIME importance > {importance:.3f}"
        }
        rules['tp_enhancement_rules'].append(rule)
    
    # Generate threshold recommendations
    if len(fp_driving_words) > 0:
        fp_threshold = fp_driving_words['lime_inclination'].median()
        rules['threshold_recommendations']['fp_flag_threshold'] = fp_threshold
    
    if len(tp_driving_words) > 0:
        tp_threshold = tp_driving_words['lime_inclination'].median()
        rules['threshold_recommendations']['tp_boost_threshold'] = tp_threshold
    
    return rules

def export_lime_results(lime_results, lime_patterns, classifier_results, output_dir, timestamp):
    """
    Export LIME analysis results to Excel and other formats
    
    Args:
        lime_results: Results from generate_lime_explanations_bert
        lime_patterns: Results from analyze_lime_patterns
        classifier_results: Classifier training results
        output_dir: Output directory
        timestamp: Timestamp for file naming
    """
    
    print("\nStep 11: Exporting LIME results...")
    
    # Create LIME-specific output file
    lime_file = os.path.join(output_dir, f'lime_analysis_results_{timestamp}.xlsx')
    
    with pd.ExcelWriter(lime_file, engine='xlsxwriter') as writer:
        
        # 1. LIME Summary Sheet
        if classifier_results:
            best_model = classifier_results['metadata']['best_model']
            best_accuracy = classifier_results[best_model]['accuracy']
            
            lime_summary_data = {
                'Metric': [
                    'Best Classifier Model',
                    'Classifier Accuracy',
                    'Total LIME Explanations Generated',
                    'Unique Words Analyzed',
                    'FP-Driving Words Found',
                    'TP-Driving Words Found',
                    'LIME-TF-IDF Agreement Rate',
                    'Top FP-Driving Word',
                    'Top TP-Driving Word'
                ],
                'Value': [
                    best_model,
                    f"{best_accuracy:.4f}",
                    len(lime_results['explanations']) if lime_results else 0,
                    lime_patterns['summary_stats']['total_words_analyzed'] if lime_patterns else 0,
                    lime_patterns['summary_stats']['fp_driving_count'] if lime_patterns else 0,
                    lime_patterns['summary_stats']['tp_driving_count'] if lime_patterns else 0,
                    f"{lime_patterns['tfidf_comparison']['agreement_rate']:.3f}" if lime_patterns else 'N/A',
                    lime_patterns['fp_driving_words'].iloc[0]['word'] if lime_patterns and len(lime_patterns['fp_driving_words']) > 0 else 'N/A',
                    lime_patterns['tp_driving_words'].iloc[0]['word'] if lime_patterns and len(lime_patterns['tp_driving_words']) > 0 else 'N/A'
                ]
            }
            
            lime_summary_df = pd.DataFrame(lime_summary_data)
            lime_summary_df.to_excel(writer, sheet_name='LIME_Summary', index=False)
        
        # 2. Word Importance Analysis
        if lime_patterns and 'lime_word_analysis' in lime_patterns:
            lime_word_df = lime_patterns['lime_word_analysis'].copy()
            lime_word_df.to_excel(writer, sheet_name='LIME_Word_Analysis', index=False)
        
        # 3. FP-Driving Words
        if lime_patterns and len(lime_patterns['fp_driving_words']) > 0:
            lime_patterns['fp_driving_words'].to_excel(writer, sheet_name='FP_Driving_Words', index=False)
        
        # 4. TP-Driving Words
        if lime_patterns and len(lime_patterns['tp_driving_words']) > 0:
            lime_patterns['tp_driving_words'].to_excel(writer, sheet_name='TP_Driving_Words', index=False)
        
        # 5. LIME vs TF-IDF Comparison
        if lime_patterns and 'tfidf_comparison' in lime_patterns:
            comparison_df = lime_patterns['tfidf_comparison']['comparison_df']
            if len(comparison_df) > 0:
                comparison_df.to_excel(writer, sheet_name='LIME_TFIDF_Comparison', index=False)
        
        # 6. Actionable Rules
        if lime_patterns and 'actionable_rules' in lime_patterns:
            rules = lime_patterns['actionable_rules']
            
            # FP Prevention Rules
            if rules['fp_prevention_rules']:
                fp_rules_df = pd.DataFrame(rules['fp_prevention_rules'])
                fp_rules_df.to_excel(writer, sheet_name='FP_Prevention_Rules', index=False)
            
            # TP Enhancement Rules
            if rules['tp_enhancement_rules']:
                tp_rules_df = pd.DataFrame(rules['tp_enhancement_rules'])
                tp_rules_df.to_excel(writer, sheet_name='TP_Enhancement_Rules', index=False)
        
        # 7. Sample Explanations
        if lime_results and 'sample_info' in lime_results:
            sample_data = []
            for sample in lime_results['sample_info']:
                # Top 5 most important words for each sample
                word_importance = sample['word_importance']
                top_words = sorted(word_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                
                sample_data.append({
                    'sample_index': sample['index'],
                    'true_label': sample['true_label'],
                    'predicted_prob_tp': sample['predicted_prob_tp'],
                    'text_length': sample['text_length'],
                    'top_word_1': top_words[0][0] if len(top_words) > 0 else '',
                    'top_word_1_importance': top_words[0][1] if len(top_words) > 0 else 0,
                    'top_word_2': top_words[1][0] if len(top_words) > 1 else '',
                    'top_word_2_importance': top_words[1][1] if len(top_words) > 1 else 0,
                    'top_word_3': top_words[2][0] if len(top_words) > 2 else '',
                    'top_word_3_importance': top_words[2][1] if len(top_words) > 2 else 0,
                    'top_word_4': top_words[3][0] if len(top_words) > 3 else '',
                    'top_word_4_importance': top_words[3][1] if len(top_words) > 3 else 0,
                    'top_word_5': top_words[4][0] if len(top_words) > 4 else '',
                    'top_word_5_importance': top_words[4][1] if len(top_words) > 4 else 0
                })
            
            if sample_data:
                sample_explanations_df = pd.DataFrame(sample_data)
                sample_explanations_df.to_excel(writer, sheet_name='Sample_Explanations', index=False)
    
    print(f"LIME results exported to: {lime_file}")
    
    # Export actionable rules as text file
    if lime_patterns and 'actionable_rules' in lime_patterns:
        rules_file = os.path.join(output_dir, f'lime_actionable_rules_{timestamp}.txt')
        export_lime_rules_text(lime_patterns['actionable_rules'], rules_file)
        print(f"Actionable rules exported to: {rules_file}")

def export_lime_rules_text(actionable_rules, output_file):
    """
    Export LIME actionable rules as formatted text file
    
    Args:
        actionable_rules: Rules dictionary from generate_lime_rules
        output_file: Output file path
    """
    
    with open(output_file, 'w') as f:
        f.write("LIME-BASED ACTIONABLE RULES FOR TP/FP CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        
        # FP Prevention Rules
        f.write("FALSE POSITIVE PREVENTION RULES:\n")
        f.write("-" * 40 + "\n")
        
        for i, rule in enumerate(actionable_rules['fp_prevention_rules'], 1):
            f.write(f"{i}. WORD: '{rule['word']}'\n")
            f.write(f"   Risk Level: {rule['confidence']}\n")
            f.write(f"   LIME Importance: {rule['importance_score']:.4f}\n")
            f.write(f"   Action: {rule['recommendation']}\n")
            f.write(f"   Threshold: {rule['threshold']}\n")
            f.write(f"   Implementation: if word_importance['{rule['word']}'] < {rule['threshold'].split('<')[1].strip()}: flag_for_review()\n\n")
        
        # TP Enhancement Rules
        f.write("\nTRUE POSITIVE ENHANCEMENT RULES:\n")
        f.write("-" * 40 + "\n")
        
        for i, rule in enumerate(actionable_rules['tp_enhancement_rules'], 1):
            f.write(f"{i}. WORD: '{rule['word']}'\n")
            f.write(f"   Confidence Level: {rule['confidence']}\n")
            f.write(f"   LIME Importance: {rule['importance_score']:.4f}\n")
            f.write(f"   Action: {rule['recommendation']}\n")
            f.write(f"   Threshold: {rule['threshold']}\n")
            f.write(f"   Implementation: if word_importance['{rule['word']}'] > {rule['threshold'].split('>')[1].strip()}: boost_confidence()\n\n")
        
        # Threshold Recommendations
        if actionable_rules['threshold_recommendations']:
            f.write("\nRECOMMENDED THRESHOLDS:\n")
            f.write("-" * 30 + "\n")
            for threshold_name, threshold_value in actionable_rules['threshold_recommendations'].items():
                f.write(f"{threshold_name}: {threshold_value:.4f}\n")

def create_lime_visualization(lime_patterns, tfidf_comparison, output_dir, timestamp):
    """
    Create visualizations combining LIME and TF-IDF results
    
    Args:
        lime_patterns: LIME pattern analysis results
        tfidf_comparison: TF-IDF comparison results
        output_dir: Output directory
        timestamp: Timestamp for file naming
    """
    
    print("Creating LIME visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('LIME-based Explainable AI Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Top FP-driving words from LIME
    ax1 = axes[0, 0]
    if len(lime_patterns['fp_driving_words']) > 0:
        fp_words = lime_patterns['fp_driving_words'].head(10)
        bars1 = ax1.barh(range(len(fp_words)), fp_words['lime_inclination'])
        ax1.set_yticks(range(len(fp_words)))
        ax1.set_yticklabels(fp_words['word'], fontsize=10)
        ax1.set_xlabel('LIME Inclination Score (Negative = FP-driving)')
        ax1.set_title('Top Words Driving False Positives (LIME)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Color bars by value
        for i, bar in enumerate(bars1):
            bar.set_color(plt.cm.Reds(0.5 + 0.5 * abs(fp_words.iloc[i]['lime_inclination']) / abs(fp_words['lime_inclination'].min())))
    
    # Plot 2: Top TP-driving words from LIME
    ax2 = axes[0, 1]
    if len(lime_patterns['tp_driving_words']) > 0:
        tp_words = lime_patterns['tp_driving_words'].head(10)
        bars2 = ax2.barh(range(len(tp_words)), tp_words['lime_inclination'])
        ax2.set_yticks(range(len(tp_words)))
        ax2.set_yticklabels(tp_words['word'], fontsize=10)
        ax2.set_xlabel('LIME Inclination Score (Positive = TP-driving)')
        ax2.set_title('Top Words Driving True Positives (LIME)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Color bars by value
        for i, bar in enumerate(bars2):
            bar.set_color(plt.cm.Greens(0.5 + 0.5 * tp_words.iloc[i]['lime_inclination'] / tp_words['lime_inclination'].max()))
    
    # Plot 3: LIME vs TF-IDF comparison
    ax3 = axes[1, 0]
    if 'comparison_df' in tfidf_comparison and len(tfidf_comparison['comparison_df']) > 0:
        comp_df = tfidf_comparison['comparison_df']
        scatter = ax3.scatter(comp_df['tfidf_fp_inclination'], comp_df['lime_inclination'], 
                             alpha=0.7, s=60, c=comp_df['agreement'].astype(int), 
                             cmap='RdYlGn')
        
        # Add diagonal reference line
        max_val = max(comp_df['tfidf_fp_inclination'].abs().max(), comp_df['lime_inclination'].abs().max())
        ax3.plot([-max_val, max_val], [-max_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')
        
        ax3.set_xlabel('TF-IDF FP Inclination')
        ax3.set_ylabel('LIME Inclination')
        ax3.set_title('LIME vs TF-IDF Agreement')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Agreement (1=Yes, 0=No)')
    
    # Plot 4: Word importance distribution
    ax4 = axes[1, 1]
    if 'lime_word_analysis' in lime_patterns:
        word_df = lime_patterns['lime_word_analysis']
        
        # Plot histogram of LIME inclination scores
        ax4.hist(word_df['lime_inclination'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', label='Neutral (0)')
        ax4.set_xlabel('LIME Inclination Score')
        ax4.set_ylabel('Number of Words')
        ax4.set_title('Distribution of Word Importance Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add annotations
        fp_words_count = (word_df['lime_inclination'] < -0.01).sum()
        tp_words_count = (word_df['lime_inclination'] > 0.01).sum()
        ax4.text(0.02, 0.95, f'FP-driving: {fp_words_count} words', transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax4.text(0.02, 0.88, f'TP-driving: {tp_words_count} words', transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f'lime_analysis_visualization_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"LIME visualization saved to: {plot_file}")
    
    plt.close()

def bert_contrast_analysis_with_lime(data_file_path, output_dir='bert_lime_analysis_results'):
    """
    Enhanced BERT-based contrast analysis with LIME explanations
    Complete integration of BERT clustering and LIME explainability
    
    Args:
        data_file_path: str, path to your Excel file
        output_dir: str, directory to save results
        
    Returns:
        dict: Comprehensive analysis results including BERT, TF-IDF, and LIME
    """
    
    print("="*80)
    print("ENHANCED BERT + LIME CONTRAST ANALYSIS: TRUE POSITIVES vs FALSE POSITIVES")
    print("="*80)
    print("System: Dell Latitude 5450 | Intel Ultra 7 155U | 32GB RAM")
    print("Features: BERT embeddings + LIME explanations + TF-IDF analysis")
    print("="*80)
    
    # Step 1-7: Original BERT analysis (keeping existing functionality)
    # [Previous steps 1-7 remain the same as in original function]
    
    # Step 1: Load and prepare data
    print("\nStep 1: Loading and preparing data...")
    try:
        df = pd.read_excel(data_file_path)
        print(f"Data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"File not found: {data_file_path}")
        return None
    
    # Clean column names
    df.columns = df.columns.str.rstrip()
    
    # Prepare customer transcripts
    df['Customer_Transcript_Raw'] = df['Customer Transcript'].fillna('').astype(str)
    
    # Apply text cleaning
    print("Cleaning customer transcripts...")
    df['Customer_Transcript_Clean'] = df['Customer_Transcript_Raw'].apply(clean_text)
    
    # Filter out empty transcripts after cleaning
    df = df[df['Customer_Transcript_Clean'].str.len() > 10].reset_index(drop=True)
    print(f"{len(df)} valid transcripts after cleaning")
    
    # Create True Positive and False Positive labels
    df['Is_TP'] = (df['Primary Marker'] == 'TP').astype(int)
    df['Is_FP'] = (df['Primary Marker'] == 'FP').astype(int)
    
    tp_count = df['Is_TP'].sum()
    fp_count = df['Is_FP'].sum()
    
    print(f"True Positives: {tp_count}")
    print(f"False Positives: {fp_count}")
    
    if tp_count < 10 or fp_count < 10:
        print("Insufficient data for meaningful analysis (need at least 10 of each)")
        return None
    
    # Step 2: Load BERT model
    print("\nStep 2: Loading BERT model...")
    model_name = 'all-MiniLM-L6-v2'
    try:
        bert_model = SentenceTransformer(model_name)
        print(f"BERT model loaded: {model_name}")
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        return None
    
    # Step 3: Generate embeddings
    print("\nStep 3: Generating BERT embeddings...")
    batch_size = 32
    texts = df['Customer_Transcript_Clean'].tolist()
    
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = bert_model.encode(batch_texts, 
                                           batch_size=batch_size,
                                           show_progress_bar=True if i == 0 else False,
                                           normalize_embeddings=True)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")
    df['bert_embedding'] = [emb for emb in embeddings]
    
    # Step 4: Contrast Analysis
    print("\nStep 4: Running contrast analysis...")
    tp_mask = df['Is_TP'] == 1
    fp_mask = df['Is_FP'] == 1
    
    tp_embeddings = embeddings[tp_mask]
    fp_embeddings = embeddings[fp_mask]
    
    tp_centroid = np.mean(tp_embeddings, axis=0)
    fp_centroid = np.mean(fp_embeddings, axis=0)
    
    centroid_similarity = cosine_similarity([tp_centroid], [fp_centroid])[0][0]
    print(f"TP-FP centroid similarity: {centroid_similarity:.4f}")
    
    # Step 5: Dimensionality reduction
    print("\nStep 5: Dimensionality reduction and clustering...")
    pca_2d = PCA(n_components=2, random_state=42)
    pca_3d = PCA(n_components=3, random_state=42)
    
    embeddings_2d = pca_2d.fit_transform(embeddings)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    df['pca_x'] = embeddings_2d[:, 0]
    df['pca_y'] = embeddings_2d[:, 1]
    df['pca_z'] = embeddings_3d[:, 2]
    
    # Step 6: TF-IDF keyword analysis
    print("\nStep 6: Keyword analysis for FP inclination...")
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    
    all_clean_texts = df['Customer_Transcript_Clean'].tolist()
    tfidf_matrix = tfidf.fit_transform(all_clean_texts)
    feature_names = tfidf.get_feature_names_out()
    
    tp_tfidf_mean = np.mean(tfidf_matrix[tp_mask].toarray(), axis=0)
    fp_tfidf_mean = np.mean(tfidf_matrix[fp_mask].toarray(), axis=0)
    
    fp_inclination = fp_tfidf_mean - tp_tfidf_mean
    
    keyword_analysis = pd.DataFrame({
        'keyword': feature_names,
        'tp_score': tp_tfidf_mean,
        'fp_score': fp_tfidf_mean,
        'fp_inclination': fp_inclination,
        'fp_ratio': np.where(tp_tfidf_mean > 0, fp_tfidf_mean / tp_tfidf_mean, np.inf)
    })
    
    keyword_analysis = keyword_analysis.sort_values('fp_inclination', ascending=False)
    
    # Step 7: Clustering
    print("\nStep 7: Clustering analysis...")
    n_clusters = min(8, len(df) // 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    df['cluster'] = clusters
    
    cluster_analysis = df.groupby('cluster').agg({
        'Is_TP': ['count', 'sum', 'mean'],
        'Is_FP': ['count', 'sum', 'mean']
    }).round(3)
    
    cluster_analysis.columns = ['total_count', 'tp_count', 'tp_ratio', 'fp_total', 'fp_count', 'fp_ratio']
    cluster_analysis = cluster_analysis[['total_count', 'tp_count', 'tp_ratio', 'fp_count', 'fp_ratio']]
    
    # NEW LIME INTEGRATION STEPS
    
    # Step 8.5: Train BERT classifier for LIME
    classifier_results = train_bert_classifier_for_lime(df, embeddings)
    
    # Step 9: Generate LIME explanations
    lime_results = None
    lime_patterns = None
    
    if classifier_results and classifier_results[classifier_results['metadata']['best_model']]['accuracy'] > 0.6:
        lime_results = generate_lime_explanations_bert(
            df, bert_model, classifier_results, output_dir, max_explanations=20
        )
        
        # Step 10: Analyze LIME patterns
        if lime_results:
            lime_patterns = analyze_lime_patterns(lime_results, keyword_analysis)
    else:
        print("Classifier accuracy too low for reliable LIME explanations")
    
    # Step 8: Export results (Enhanced with LIME)
    print(f"\nStep 8: Exporting enhanced results to {output_dir}...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export main BERT results (original functionality)
    results_file = os.path.join(output_dir, f'bert_contrast_analysis_{timestamp}.xlsx')
    
    with pd.ExcelWriter(results_file, engine='xlsxwriter') as writer:
        # Original sheets
        summary_data = {
            'Metric': [
                'Total Transcripts',
                'True Positives',
                'False Positives',
                'TP-FP Centroid Similarity',
                'PCA 2D Explained Variance',
                'Number of Clusters',
                'Top FP Keyword (TF-IDF)',
                'Best Classifier Model',
                'Classifier Accuracy',
                'LIME Explanations Generated'
            ],
            'Value': [
                len(df),
                tp_count,
                fp_count,
                f"{centroid_similarity:.4f}",
                f"{pca_2d.explained_variance_ratio_.sum():.3f}",
                n_clusters,
                keyword_analysis.iloc[0]['keyword'],
                classifier_results['metadata']['best_model'] if classifier_results else 'N/A',
                f"{classifier_results[classifier_results['metadata']['best_model']]['accuracy']:.4f}" if classifier_results else 'N/A',
                len(lime_results['explanations']) if lime_results else 0
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        keyword_analysis.head(50).to_excel(writer, sheet_name='TF-IDF_Keywords', index=False)
        cluster_analysis.to_excel(writer, sheet_name='Cluster_Analysis')
    
    print(f"BERT results exported to: {results_file}")
    
    # Export LIME-specific results
    if lime_results and lime_patterns:
        export_lime_results(lime_results, lime_patterns, classifier_results, output_dir, timestamp)
        
        # Create combined visualization
        create_lime_visualization(lime_patterns, lime_patterns['tfidf_comparison'], output_dir, timestamp)
    
    # Create original visualization
    create_contrast_visualization(df, keyword_analysis, cluster_analysis, output_dir, timestamp)
    
    # Step 11: Generate enhanced insights
    enhanced_insights = generate_enhanced_insights(
        df, keyword_analysis, cluster_analysis, centroid_similarity, 
        tp_count, fp_count, classifier_results, lime_patterns
    )
    
    # Export enhanced insights
    insights_file = os.path.join(output_dir, f'enhanced_insights_{timestamp}.txt')
    with open(insights_file, 'w') as f:
        f.write(enhanced_insights)
    
    print(f"Enhanced insights exported to: {insights_file}")
    
    return {
        'dataframe': df,
        'embeddings': embeddings,
        'keyword_analysis': keyword_analysis,
        'cluster_analysis': cluster_analysis,
        'centroid_similarity': centroid_similarity,
        'classifier_results': classifier_results,
        'lime_results': lime_results,
        'lime_patterns': lime_patterns,
        'insights': enhanced_insights,
        'model': bert_model
    }

def generate_enhanced_insights(df, keyword_analysis, cluster_analysis, centroid_similarity, 
                             tp_count, fp_count, classifier_results, lime_patterns):
    """
    Generate enhanced insights combining BERT, TF-IDF, and LIME analysis
    """
    
    insights = []
    insights.append("ENHANCED BERT + LIME CONTRAST ANALYSIS INSIGHTS")
    insights.append("=" * 60)
    insights.append("")
    
    # Original BERT insights
    insights.append("BERT SEMANTIC ANALYSIS:")
    insights.append(f"   • Total transcripts analyzed: {len(df)}")
    insights.append(f"   • True Positives: {tp_count} ({tp_count/len(df)*100:.1f}%)")
    insights.append(f"   • False Positives: {fp_count} ({fp_count/len(df)*100:.1f}%)")
    insights.append(f"   • TP-FP centroid similarity: {centroid_similarity:.4f}")
    insights.append("")
    
    # Classifier performance
    if classifier_results:
        best_model = classifier_results['metadata']['best_model']
        accuracy = classifier_results[best_model]['accuracy']
        insights.append("CLASSIFICATION MODEL PERFORMANCE:")
        insights.append(f"   • Best model: {best_model}")
        insights.append(f"   • Accuracy: {accuracy:.4f}")
        insights.append(f"   • Model reliability: {'HIGH' if accuracy > 0.8 else 'MEDIUM' if accuracy > 0.6 else 'LOW'}")
        insights.append("")
    
    # LIME-specific insights
    if lime_patterns:
        insights.append("LIME EXPLAINABILITY INSIGHTS:")
        insights.append(f"   • Words analyzed: {lime_patterns['summary_stats']['total_words_analyzed']}")
        insights.append(f"   • FP-driving words: {lime_patterns['summary_stats']['fp_driving_count']}")
        insights.append(f"   • TP-driving words: {lime_patterns['summary_stats']['tp_driving_count']}")
        
        if 'tfidf_comparison' in lime_patterns:
            agreement = lime_patterns['tfidf_comparison']['agreement_rate']
            insights.append(f"   • LIME-TF-IDF agreement: {agreement:.3f}")
        insights.append("")
        
        # Top LIME findings
        if len(lime_patterns['fp_driving_words']) > 0:
            insights.append("TOP LIME-IDENTIFIED FP-DRIVING WORDS:")
            for i, (_, row) in enumerate(lime_patterns['fp_driving_words'].head(5).iterrows(), 1):
                insights.append(f"   {i}. '{row['word']}' (importance: {row['lime_inclination']:.4f})")
        insights.append("")
        
        if len(lime_patterns['tp_driving_words']) > 0:
            insights.append("TOP LIME-IDENTIFIED TP-DRIVING WORDS:")
            for i, (_, row) in enumerate(lime_patterns['tp_driving_words'].head(5).iterrows(), 1):
                insights.append(f"   {i}. '{row['word']}' (importance: {row['lime_inclination']:.4f})")
        insights.append("")
    
    # Combined recommendations
    insights.append("INTEGRATED RECOMMENDATIONS:")
    insights.append("")
    
    insights.append("   IMMEDIATE ACTIONS:")
    if lime_patterns and 'actionable_rules' in lime_patterns:
        fp_rules = lime_patterns['actionable_rules']['fp_prevention_rules']
        if fp_rules:
            insights.append(f"   • Implement FP prevention for {len(fp_rules)} critical words")
            insights.append(f"   • Priority word: '{fp_rules[0]['word']}'")
    
    insights.append("   • Use BERT embeddings for semantic similarity scoring")
    insights.append("   • Combine TF-IDF and LIME insights for robust classification")
    insights.append("")
    
    insights.append("   SYSTEM ENHANCEMENTS:")
    insights.append("   • Deploy LIME explanations for prediction transparency")
    insights.append("   • Set up monitoring dashboard for identified keywords")
    insights.append("   • Implement ensemble approach: BERT + Rules + LIME")
    insights.append("   • Regular model retraining with new LIME insights")
    insights.append("")
    
    insights.append("=" * 60)
    insights.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(insights)

# Updated main execution function
def run_enhanced_bert_lime_analysis():
    """
    Execute the complete enhanced BERT + LIME contrast analysis
    """
    
    print("Starting Enhanced BERT + LIME Contrast Analysis...")
    
    # Run the enhanced analysis
    data_file_path = 'Precision_Drop_Analysis_OG.xlsx'
    results = bert_contrast_analysis_with_lime(data_file_path)
    
    if results:
        print("\n" + "="*70)
        print("ENHANCED BERT + LIME ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nKey findings:")
        print(f"• TP-FP Centroid Similarity: {results['centroid_similarity']:.4f}")
        print(f"• Top TF-IDF FP Keyword: '{results['keyword_analysis'].iloc[0]['keyword']}'")
        
        if results['classifier_results']:
            best_model = results['classifier_results']['metadata']['best_model']
            accuracy = results['classifier_results'][best_model]['accuracy']
            print(f"• Best Classifier: {best_model} (Accuracy: {accuracy:.4f})")
        
        if results['lime_patterns']:
            fp_words = len(results['lime_patterns']['fp_driving_words'])
            tp_words = len(results['lime_patterns']['tp_driving_words'])
            print(f"• LIME Analysis: {fp_words} FP-driving, {tp_words} TP-driving words")
        
        return results
    else:
        print("Enhanced analysis failed")
        return None

if __name__ == "__main__":
    # Run the enhanced BERT + LIME analysis
    result = run_enhanced_bert_lime_analysis()


# pip install pandas numpy scipy sentence-transformers lime scikit-learn nltk matplotlib seaborn openpyxl xlsxwriter torch --index-url https://download.pytorch.org/whl/cpu






