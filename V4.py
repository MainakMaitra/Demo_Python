# =============================================================================
# FIXED DIAGNOSTIC FUNCTION - Replace the previous one
# =============================================================================

def diagnose_visualization_data_fixed(negation_df, pattern_analysis_df):
    """Fixed diagnostic function - handles data type issues"""
    
    print("\n" + "="*60)
    print("DATA DIAGNOSTIC FOR VISUALIZATIONS (FIXED)")
    print("="*60)
    
    print("1. BASIC DATA CHECK:")
    print(f"   Negation DataFrame shape: {negation_df.shape}")
    print(f"   Pattern Analysis DataFrame shape: {pattern_analysis_df.shape}")
    
    print("\n2. REQUIRED COLUMNS CHECK:")
    required_cols = ['UUID', 'Primary_Marker', 'Speaker', 'Negation_Word', 'Context']
    missing_cols = [col for col in required_cols if col not in negation_df.columns]
    if missing_cols:
        print(f"   MISSING: {missing_cols}")
    else:
        print("   All required columns present")
    
    print("\n3. DATA DISTRIBUTION:")
    print(f"   Primary Marker: {dict(negation_df['Primary_Marker'].value_counts())}")
    print(f"   Speaker: {dict(negation_df['Speaker'].value_counts())}")
    
    if 'Pattern_Cluster' in negation_df.columns:
        print(f"   Pattern Clusters: {sorted(negation_df['Pattern_Cluster'].unique())}")
        pattern_dist = negation_df['Pattern_Cluster'].value_counts().sort_index()
        print(f"   Pattern distribution: {dict(pattern_dist)}")
    
    print("\n4. CONTEXT DATA CHECK:")
    if 'Context' in negation_df.columns:
        non_empty_contexts = negation_df['Context'].dropna()
        print(f"   Non-empty contexts: {len(non_empty_contexts)}/{len(negation_df)}")
        if len(non_empty_contexts) > 0:
            # Convert to string first to handle any type issues
            context_lengths = non_empty_contexts.astype(str).str.len()
            avg_length = context_lengths.mean()
            print(f"   Average context length: {avg_length:.1f} characters")
    
    print("\n5. TEMPORAL DATA CHECK:")
    if 'Year_Month' in negation_df.columns:
        # FIXED: Handle mixed data types in Year_Month column
        year_months = negation_df['Year_Month'].dropna().astype(str)
        unique_months = sorted(year_months.unique())
        print(f"   Available months: {unique_months}")
        
        # FIXED: Use string conversion for counting
        monthly_counts = year_months.value_counts().sort_index()
        print(f"   Monthly distribution: {dict(monthly_counts)}")
        
        # Check for data type issues
        original_types = negation_df['Year_Month'].apply(type).value_counts()
        print(f"   Year_Month data types: {dict(original_types)}")
    
    print("\n6. PATTERN ANALYSIS CHECK:")
    if len(pattern_analysis_df) > 0:
        print(f"   Pattern analysis clusters: {sorted(pattern_analysis_df['Cluster_ID'].unique())}")
        if 'Total_Count' in pattern_analysis_df.columns:
            total_patterns = pattern_analysis_df['Total_Count'].sum()
            print(f"   Total instances across patterns: {total_patterns}")
        
        # Check for patterns with sufficient data for word clouds
        if 'Pattern_Cluster' in negation_df.columns:
            pattern_contexts = []
            for cluster_id in pattern_analysis_df['Cluster_ID']:
                cluster_data = negation_df[negation_df['Pattern_Cluster'] == cluster_id]
                context_data = cluster_data['Context'].dropna().astype(str)
                total_context_length = context_data.str.len().sum()
                pattern_contexts.append({
                    'Cluster': cluster_id,
                    'Instances': len(cluster_data),
                    'Context_Length': total_context_length,
                    'Suitable_for_Wordcloud': len(cluster_data) >= 5 and total_context_length >= 100
                })
            
            print(f"\n   WORDCLOUD SUITABILITY:")
            for pc in pattern_contexts:
                status = "YES" if pc['Suitable_for_Wordcloud'] else "NO"
                print(f"     Pattern {pc['Cluster']}: {pc['Instances']} instances, {pc['Context_Length']} chars - {status}")
    
    print("\n7. RECOMMENDATIONS:")
    if len(negation_df) < 100:
        print("   WARNING: Low data volume may result in sparse visualizations")
    
    if 'Pattern_Cluster' in negation_df.columns:
        pattern_counts = negation_df['Pattern_Cluster'].value_counts()
        small_patterns = pattern_counts[pattern_counts < 5]
        if len(small_patterns) > 0:
            print(f"   NOTE: {len(small_patterns)} patterns have <5 instances (will skip word clouds)")
    
    # Check for temporal data issues
    if 'Year_Month' in negation_df.columns:
        mixed_types = negation_df['Year_Month'].apply(type).nunique() > 1
        if mixed_types:
            print("   WARNING: Year_Month column has mixed data types - will convert to string")
    
    print("\n8. DATA QUALITY SUMMARY:")
    total_score = 0
    max_score = 6
    
    # Basic data check
    if len(negation_df) > 0 and len(pattern_analysis_df) > 0:
        total_score += 1
        print("   ✓ Data loaded successfully")
    
    # Required columns
    if len(missing_cols) == 0:
        total_score += 1
        print("   ✓ All required columns present")
    
    # TP/FP balance
    if 'Primary_Marker' in negation_df.columns:
        tp_count = len(negation_df[negation_df['Primary_Marker'] == 'TP'])
        fp_count = len(negation_df[negation_df['Primary_Marker'] == 'FP'])
        if tp_count > 0 and fp_count > 0:
            total_score += 1
            print("   ✓ Both TP and FP data present")
    
    # Pattern clusters
    if 'Pattern_Cluster' in negation_df.columns and negation_df['Pattern_Cluster'].nunique() >= 3:
        total_score += 1
        print("   ✓ Multiple pattern clusters found")
    
    # Context data
    if 'Context' in negation_df.columns:
        context_coverage = len(negation_df['Context'].dropna()) / len(negation_df)
        if context_coverage > 0.8:
            total_score += 1
            print("   ✓ Good context data coverage")
    
    # Temporal data
    if 'Year_Month' in negation_df.columns and negation_df['Year_Month'].nunique() >= 2:
        total_score += 1
        print("   ✓ Multiple time periods available")
    
    print(f"\n   OVERALL SCORE: {total_score}/{max_score}")
    if total_score >= 5:
        print("   STATUS: Ready for comprehensive visualization")
    elif total_score >= 3:
        print("   STATUS: Ready for basic visualization")
    else:
        print("   STATUS: Data quality issues - check recommendations above")
    
    return total_score >= 3

# =============================================================================
# RUN THIS DIAGNOSTIC FIRST
# =============================================================================

# Replace your diagnostic call with:
# diagnose_visualization_data_fixed(negation_df_clustered, pattern_analysis_df)
