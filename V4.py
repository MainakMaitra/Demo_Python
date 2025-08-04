# Optimal Clustering with Elbow Method and Silhouette Analysis
# =============================================================

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import numpy as np

def find_optimal_clusters(embeddings, max_clusters=15, min_clusters=2):
    """
    Find optimal number of clusters using multiple methods
    """
    
    print(f"Finding optimal clusters (testing {min_clusters} to {max_clusters})...")
    
    # Storage for metrics
    cluster_range = range(min_clusters, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    
    for k in cluster_range:
        print(f"Testing k={k}...")
        
        # Fit K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        sil_score = silhouette_score(embeddings, cluster_labels)
        ch_score = calinski_harabasz_score(embeddings, cluster_labels)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        calinski_harabasz_scores.append(ch_score)
        
        print(f"  k={k}: Inertia={inertia:.0f}, Silhouette={sil_score:.3f}, CH_Score={ch_score:.1f}")
    
    # Find optimal k using different methods
    optimal_k_results = analyze_clustering_metrics(
        cluster_range, inertias, silhouette_scores, calinski_harabasz_scores
    )
    
    # Create visualization
    plot_clustering_metrics(
        cluster_range, inertias, silhouette_scores, calinski_harabasz_scores, optimal_k_results
    )
    
    return optimal_k_results

def analyze_clustering_metrics(cluster_range, inertias, silhouette_scores, calinski_harabasz_scores):
    """
    Analyze clustering metrics to find optimal k
    """
    
    cluster_range = list(cluster_range)
    
    # Method 1: Elbow method (rate of change in inertia)
    deltas = np.diff(inertias)
    second_deltas = np.diff(deltas)
    
    # Find elbow (maximum second derivative)
    elbow_idx = np.argmax(second_deltas) + 2  # +2 because of double diff
    elbow_k = cluster_range[elbow_idx] if elbow_idx < len(cluster_range) else cluster_range[0]
    
    # Method 2: Maximum silhouette score
    max_sil_idx = np.argmax(silhouette_scores)
    silhouette_k = cluster_range[max_sil_idx]
    
    # Method 3: Maximum Calinski-Harabasz score
    max_ch_idx = np.argmax(calinski_harabasz_scores)
    calinski_k = cluster_range[max_ch_idx]
    
    # Method 4: Stability analysis (rate of change in silhouette)
    sil_stability = np.diff(silhouette_scores)
    stable_idx = np.argmin(np.abs(sil_stability)) + 1  # Most stable point
    stability_k = cluster_range[stable_idx] if stable_idx < len(cluster_range) else cluster_range[0]
    
    results = {
        'elbow_method': elbow_k,
        'max_silhouette': silhouette_k,
        'max_calinski': calinski_k,
        'stability_point': stability_k,
        'all_metrics': {
            'cluster_range': cluster_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_harabasz_scores
        }
    }
    
    print("\nOptimal Cluster Analysis Results:")
    print(f"  Elbow Method: k = {elbow_k}")
    print(f"  Max Silhouette: k = {silhouette_k} (score: {silhouette_scores[max_sil_idx]:.3f})")
    print(f"  Max Calinski-Harabasz: k = {calinski_k} (score: {calinski_harabasz_scores[max_ch_idx]:.1f})")
    print(f"  Stability Point: k = {stability_k}")
    
    # Recommend final k (weighted decision)
    k_votes = [elbow_k, silhouette_k, calinski_k]
    recommended_k = max(set(k_votes), key=k_votes.count)  # Most common vote
    
    # If no consensus, use silhouette (most reliable for semantic data)
    if k_votes.count(recommended_k) == 1:
        recommended_k = silhouette_k
    
    results['recommended_k'] = recommended_k
    print(f"\nRecommended k: {recommended_k}")
    
    return results

def plot_clustering_metrics(cluster_range, inertias, silhouette_scores, calinski_scores, results):
    """
    Plot clustering evaluation metrics
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Clustering Optimization Analysis', fontsize=16, fontweight='bold')
    
    cluster_range = list(cluster_range)
    
    # Plot 1: Elbow Method (Inertia)
    ax1 = axes[0, 0]
    ax1.plot(cluster_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=results['elbow_method'], color='red', linestyle='--', 
                label=f"Elbow: k={results['elbow_method']}")
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Within-cluster Sum of Squares)')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2 = axes[0, 1]
    ax2.plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=results['max_silhouette'], color='red', linestyle='--',
                label=f"Max: k={results['max_silhouette']}")
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Calinski-Harabasz Score
    ax3 = axes[1, 0]
    ax3.plot(cluster_range, calinski_scores, 'mo-', linewidth=2, markersize=8)
    ax3.axvline(x=results['max_calinski'], color='red', linestyle='--',
                label=f"Max: k={results['max_calinski']}")
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Calinski-Harabasz Score')
    ax3.set_title('Calinski-Harabasz Index')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Summary with recommendation
    ax4 = axes[1, 1]
    
    # Normalize scores for comparison
    norm_inertias = [(max(inertias) - x) / (max(inertias) - min(inertias)) for x in inertias]
    norm_silhouette = [(x - min(silhouette_scores)) / (max(silhouette_scores) - min(silhouette_scores)) 
                      for x in silhouette_scores]
    norm_calinski = [(x - min(calinski_scores)) / (max(calinski_scores) - min(calinski_scores)) 
                    for x in calinski_scores]
    
    ax4.plot(cluster_range, norm_inertias, 'b-', label='Elbow (inverted)', alpha=0.7)
    ax4.plot(cluster_range, norm_silhouette, 'g-', label='Silhouette', alpha=0.7)
    ax4.plot(cluster_range, norm_calinski, 'm-', label='Calinski-Harabasz', alpha=0.7)
    
    ax4.axvline(x=results['recommended_k'], color='red', linestyle='-', linewidth=3,
                label=f"Recommended: k={results['recommended_k']}")
    
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Normalized Score')
    ax4.set_title('Method Comparison')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def advanced_clustering_analysis(embeddings, df, tp_mask, fp_mask):
    """
    Complete clustering analysis with optimal k selection
    """
    
    print("="*60)
    print("ADVANCED CLUSTERING ANALYSIS")
    print("="*60)
    
    # Find optimal number of clusters
    optimal_results = find_optimal_clusters(embeddings, max_clusters=15)
    
    # Use recommended k for final clustering
    recommended_k = optimal_results['recommended_k']
    
    print(f"\nFinal clustering with k={recommended_k}...")
    final_kmeans = KMeans(n_clusters=recommended_k, random_state=42, n_init=20)
    final_clusters = final_kmeans.fit_predict(embeddings)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['optimal_cluster'] = final_clusters
    
    # Analyze cluster composition with optimal k
    cluster_composition = analyze_optimal_clusters(df_clustered, tp_mask, fp_mask)
    
    return {
        'optimal_k': recommended_k,
        'cluster_labels': final_clusters,
        'clustering_metrics': optimal_results,
        'cluster_composition': cluster_composition,
        'clustered_df': df_clustered
    }

def analyze_optimal_clusters(df, tp_mask, fp_mask):
    """
    Analyze composition of optimally determined clusters
    """
    
    cluster_analysis = df.groupby('optimal_cluster').agg({
        'Is_TP': ['count', 'sum', 'mean'],
        'Is_FP': ['count', 'sum', 'mean']
    }).round(3)
    
    cluster_analysis.columns = ['total_count', 'tp_count', 'tp_ratio', 'fp_total', 'fp_count', 'fp_ratio']
    cluster_analysis = cluster_analysis[['total_count', 'tp_count', 'tp_ratio', 'fp_count', 'fp_ratio']]
    
    print("\nOptimal Cluster Composition Analysis:")
    print(cluster_analysis)
    
    # Identify problem clusters
    fp_dominated = cluster_analysis[cluster_analysis['fp_ratio'] > 0.6]
    tp_dominated = cluster_analysis[cluster_analysis['tp_ratio'] > 0.8]
    mixed_clusters = cluster_analysis[
        (cluster_analysis['fp_ratio'] <= 0.6) & (cluster_analysis['tp_ratio'] <= 0.8)
    ]
    
    print(f"\nCluster Analysis Summary:")
    print(f"  FP-dominated clusters: {len(fp_dominated)} (>60% FP)")
    print(f"  TP-dominated clusters: {len(tp_dominated)} (>80% TP)")
    print(f"  Mixed clusters: {len(mixed_clusters)}")
    
    return {
        'full_analysis': cluster_analysis,
        'fp_dominated': fp_dominated,
        'tp_dominated': tp_dominated,
        'mixed_clusters': mixed_clusters
    }

# INTEGRATION: Replace your clustering section with this
def replace_clustering_in_your_code():
    """
    Code to replace the arbitrary clustering in your main function
    """
    
    replacement_code = '''
    # REPLACE THIS SECTION (around line 280):
    # 
    # # Step 7: Clustering analysis
    # print("\\nStep 7: Clustering analysis...")
    # 
    # # K-means clustering on embeddings
    # n_clusters = min(8, len(df) // 10)  # Adaptive cluster count
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # clusters = kmeans.fit_predict(embeddings)
    
    # WITH THIS:
    
    # Step 7: Advanced clustering analysis with optimal k
    print("\\nStep 7: Advanced clustering analysis...")
    
    # Find optimal clusters using multiple methods
    clustering_results = advanced_clustering_analysis(embeddings, df, tp_mask, fp_mask)
    
    # Update dataframe with optimal clusters
    df = clustering_results['clustered_df']
    cluster_analysis = clustering_results['cluster_composition']['full_analysis']
    n_clusters = clustering_results['optimal_k']
    
    print(f"Optimal number of clusters: {n_clusters}")
    '''
    
    return replacement_code

if __name__ == "__main__":
    print("CLUSTERING OPTIMIZATION GUIDE")
    print("="*40)
    print(replace_clustering_in_your_code())
