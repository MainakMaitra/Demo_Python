# Quick patch for visualization cluster column
print("\nPreparing visualization data...")

# Ensure cluster column exists for visualization
if 'cluster' not in df.columns:
    print("Warning: Adding dummy cluster column for visualization")
    df['cluster'] = 0  # Single cluster

# Verify visualization data
print(f"Visualization ready - clusters: {df['cluster'].nunique()}")

# Create visualization
create_contrast_visualization(df, keyword_analysis, cluster_analysis, output_dir, timestamp)
