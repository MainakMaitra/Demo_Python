def get_failing_categories(df, category_col='Prosodica L1'):
    """
    Identifies categories where precision is below 70%.
    """
    # Calculate TP and FP counts by category
    category_stats = (
        df.groupby(category_col)['Primary Marker']
        .value_counts()
        .unstack()
        .fillna(0)
    )

    # Ensure both TP and FP columns are present
    for col in ['TP', 'FP']:
        if col not in category_stats.columns:
            category_stats[col] = 0

    # Compute precision
    category_stats['Precision'] = category_stats['TP'] / (category_stats['TP'] + category_stats['FP'])
    category_stats['Precision'] = category_stats['Precision'].fillna(0).round(3)

    # Filter failing categories (precision < 0.70)
    failing_categories = category_stats[category_stats['Precision'] < 0.70].copy()
    failing_categories = failing_categories.sort_values(by='Precision')

    return failing_categories.reset_index()

# Example usage:
# failing_df = get_failing_categories(df_main)
# print(failing_df.head(10))  # Show worst-performing categories
