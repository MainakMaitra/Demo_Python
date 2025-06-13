def cross_category_analysis_enhanced(df):
    """Enhanced cross-category analysis with monthly breakdown"""
    
    print("2.1 Multi-Category Transcript Analysis")
    
    # Original multi-category transcript analysis (unchanged)
    transcript_categories = df.groupby('variable5').agg({
        'Prosodica L1': 'nunique',
        'Prosodica L2': 'nunique',
        'Is_TP': 'mean',
        'Is_FP': 'mean'
    }).reset_index()
    
    transcript_categories.columns = ['Transcript_ID', 'L1_Categories', 'L2_Categories', 'Avg_Precision', 'Avg_FP_Rate']
    
    # Multi-category transcripts
    multi_category = transcript_categories[transcript_categories['L2_Categories'] > 1]
    single_category = transcript_categories[transcript_categories['L2_Categories'] == 1]
    
    print(f"Multi-Category Transcripts Analysis:")
    print(f"  Single category: {len(single_category)} transcripts")
    print(f"  Multi-category: {len(multi_category)} transcripts")
    
    if len(multi_category) > 0 and len(single_category) > 0:
        multi_precision = multi_category['Avg_Precision'].mean()
        single_precision = single_category['Avg_Precision'].mean()
        
        print(f"  Single category avg precision: {single_precision:.3f}")
        print(f"  Multi-category avg precision: {multi_precision:.3f}")
        print(f"  Precision difference: {multi_precision - single_precision:+.3f}")
        
        if multi_precision < single_precision - 0.05:
            print("  FINDING: Multi-category transcripts have LOWER precision")
    
    print("\n2.2 Category Overlap and Rule Conflicts")
    
    # FIXED: Filter out NaN values before processing category pairs
    category_pairs = df.dropna(subset=['Prosodica L2']).groupby('variable5')['Prosodica L2'].apply(list).reset_index()
    category_pairs['Category_Count'] = category_pairs['Prosodica L2'].apply(len)
    
    # Find common category combinations
    multi_cat_pairs = category_pairs[category_pairs['Category_Count'] > 1]
    
    if len(multi_cat_pairs) > 0:
        # Extract all category combinations
        combinations = []
        for categories in multi_cat_pairs['Prosodica L2']:
            if len(categories) == 2:
                # FIXED: Filter out any None/NaN values and convert to string before sorting
                clean_categories = [str(cat) for cat in categories if pd.notna(cat) and cat is not None]
                if len(clean_categories) == 2:
                    combinations.append(tuple(sorted(clean_categories)))
        
        if combinations:
            from collections import Counter
            common_combinations = Counter(combinations).most_common(5)
            
            print("Most Common Category Combinations:")
            for combo, count in common_combinations:
                print(f"  {combo[0]} + {combo[1]}: {count} transcripts")
                
                # Check precision for this combination
                combo_data = df[
                    df['variable5'].isin(
                        multi_cat_pairs[
                            multi_cat_pairs['Prosodica L2'].apply(
                                lambda x: len(x) == 2 and combo[0] in [str(cat) for cat in x if pd.notna(cat)] and combo[1] in [str(cat) for cat in x if pd.notna(cat)]
                            )
                        ]['variable5']
                    )
                ]
                
                if len(combo_data) > 0:
                    combo_precision = combo_data['Is_TP'].mean()
                    print(f"    Precision: {combo_precision:.3f}")
        else:
            print("No valid category combinations found after filtering")
    else:
        print("No multi-category transcripts found")
    
    # NEW: Monthly Cross-Category Analysis
    create_monthly_insight_table(
        df,
        "Cross-Category Performance",
        ['Is_TP', 'Is_FP'],
        "Monthly cross-category analysis: Performance patterns across category interactions"
    )
    
    print("\n2.3 New Category Cannibalization Analysis")
    
    # Original new category analysis (unchanged)
    new_categories = df[df['Is_New_Category']]['Prosodica L2'].dropna().unique()
    
    if len(new_categories) > 0:
        print(f"New Categories: {list(new_categories)}")
        
        for new_cat in new_categories:
            # Performance of new category
            new_cat_data = df[df['Prosodica L2'] == new_cat]
            new_cat_precision = new_cat_data['Is_TP'].mean()
            new_cat_volume = len(new_cat_data)
            
            print(f"\n{new_cat}:")
            print(f"  Volume: {new_cat_volume}")
            print(f"  Precision: {new_cat_precision:.3f}")
            
            # Check if similar existing categories lost volume
            # (This would require more sophisticated semantic similarity analysis)
            print(f"  Cannibalization analysis: Requires semantic similarity comparison")
    else:
        print("No new categories identified for cannibalization analysis")
    
    return transcript_categories, multi_category
