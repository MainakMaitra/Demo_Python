# Step 7: Advanced clustering analysis with optimal k
print("\nStep 7: Advanced clustering analysis...")

# Find optimal clusters using multiple methods
clustering_results = advanced_clustering_analysis(embeddings, df, tp_mask, fp_mask)

# Extract results and ensure column compatibility
optimal_k = clustering_results['optimal_k']
optimal_clusters = clustering_results['cluster_labels']
cluster_composition = clustering_results['cluster_composition']

# CRITICAL: Use 'cluster' column name for compatibility with export code
df['cluster'] = optimal_clusters
cluster_analysis = cluster_composition['full_analysis']
n_clusters = optimal_k

# Remove any duplicate cluster columns
if 'optimal_cluster' in df.columns:
    df = df.drop('optimal_cluster', axis=1)

print(f"Optimal number of clusters determined: {optimal_k}")
print("Cluster composition analysis:")
print(cluster_analysis)
