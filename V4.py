# =============================================================================
# ROOT CAUSE ANALYSIS
# =============================================================================


# Enhanced Root Cause Analysis with Monthly Breakdown
# Modification 4: Add monthly tracking to all root cause analyses

def category_specific_investigation_enhanced(df, df_rules, top_categories):
    """Enhanced category-specific investigation with monthly breakdown"""
    
    print("1.1 Review Query Rules/Keywords for Top 5 Categories")
    
    # Original query rules analysis (unchanged)
    if len(top_categories) > 0:
        top_5_worst = top_categories.head(5)
        
        print("Top 5 Categories with Worst Precision Drop:")
        print(top_5_worst[['L2_Category', 'Precision', 'Drop_Impact']].round(3))
        
        # Query rule analysis for each category
        for _, category in top_5_worst.iterrows():
            l2_cat = category['L2_Category']
            
            print(f"\n--- Analysis for {l2_cat} ---")
            
            # Find matching rules
            if df_rules is not None:
                matching_rules = df_rules[df_rules['Query'].str.contains(l2_cat, case=False, na=False)]
                
                if len(matching_rules) > 0:
                    for _, rule in matching_rules.iterrows():
                        query_text = rule['Query Text']
                        channel = rule['Channel']
                        
                        print(f"Query: {query_text}")
                        print(f"Channel: {channel}")
                        
                        # Query complexity analysis
                        or_count = len(re.findall(r'\bOR\b', query_text, re.IGNORECASE))
                        and_count = len(re.findall(r'\bAND\b', query_text, re.IGNORECASE))
                        not_count = len(re.findall(r'\bNOT\b', query_text, re.IGNORECASE))
                        
                        print(f"Query Complexity: OR({or_count}), AND({and_count}), NOT({not_count})")
                        
                        # Issues identification
                        issues = []
                        if or_count > 10:
                            issues.append("High OR complexity")
                        if not_count == 0:
                            issues.append("No negation handling")
                        if channel == 'both':
                            issues.append("Both channel usage")
                        
                        if issues:
                            print(f"Potential Issues: {', '.join(issues)}")
                        else:
                            print("No obvious query issues detected")
                else:
                    print("No matching query rules found")
            else:
                print("No query rules data available")
    
    # NEW: Monthly Analysis for Top Problem Categories
    if len(top_categories) > 0:
        top_3_categories = top_categories.head(3)['L2_Category'].tolist()
        category_subset = df[df['Prosodica L2'].isin(top_3_categories)]
        
        if len(category_subset) > 0:
            create_monthly_insight_table(
                category_subset,
                "Top Problem Categories Deep Dive",
                ['Customer_Negation_Count', 'Customer_Qualifying_Count', 'Transcript_Length'],
                "Monthly deep dive: Top problem categories' negation, qualifying, and length patterns"
            )
    
    print("\n1.2 Rule Degradation Analysis")
    
    # Original rule degradation analysis (unchanged)
    monthly_rule_performance = {}
    
    if len(top_categories) > 0:
        top_5_worst = top_categories.head(5)
        
        for _, category in top_5_worst.iterrows():
            l2_cat = category['L2_Category']
            
            # Monthly FP analysis for this category
            category_monthly = df[df['Prosodica L2'] == l2_cat].groupby('Year_Month').agg({
                'Is_FP': ['sum', 'count'],
                'Is_TP': 'sum'
            }).reset_index()
            
            if len(category_monthly) > 0:
                category_monthly.columns = ['Year_Month', 'FP_Count', 'Total_Flagged', 'TP_Count']
                category_monthly['FP_Rate'] = category_monthly['FP_Count'] / category_monthly['Total_Flagged']
                category_monthly = category_monthly.sort_values('Year_Month')
                
                # Check for degradation trend
                if len(category_monthly) > 2:
                    months = list(range(len(category_monthly)))
                    fp_rates = category_monthly['FP_Rate'].tolist()
                    
                    try:
                        slope, _, r_value, p_value, _ = stats.linregress(months, fp_rates)
                        
                        monthly_rule_performance[l2_cat] = {
                            'slope': slope,
                            'r_squared': r_value**2,
                            'p_value': p_value,
                            'trend': 'Degrading' if slope > 0.01 else 'Stable'
                        }
                    except:
                        monthly_rule_performance[l2_cat] = {'trend': 'Unable to calculate'}
    
    print("Rule Degradation Analysis:")
    for category, performance in monthly_rule_performance.items():
        trend = performance.get('trend', 'Unknown')
        if 'slope' in performance:
            slope = performance['slope']
            r_sq = performance['r_squared']
            print(f"  {category}: {trend} (slope: {slope:.4f}, R²: {r_sq:.3f})")
        else:
            print(f"  {category}: {trend}")
    
    print("\n1.3 Language Evolution Analysis")
    
    # Original language evolution analysis (unchanged)
    all_months = sorted(df['Year_Month'].unique())
    if len(all_months) >= 4:
        early_months = all_months[:2]
        recent_months = all_months[-2:]
    else:
        early_months = all_months[:1]
        recent_months = all_months[-1:]
    
    language_evolution = {}
    
    if len(top_categories) > 0:
        top_5_worst = top_categories.head(5)
        
        for _, category in top_5_worst.iterrows():
            l2_cat = category['L2_Category']
            
            early_data = df[(df['Prosodica L2'] == l2_cat) & (df['Year_Month'].isin(early_months))]
            recent_data = df[(df['Prosodica L2'] == l2_cat) & (df['Year_Month'].isin(recent_months))]
            
            if len(early_data) > 0 and len(recent_data) > 0:
                # Vocabulary analysis
                early_vocab = set(' '.join(early_data['Customer Transcript'].fillna('')).lower().split())
                recent_vocab = set(' '.join(recent_data['Customer Transcript'].fillna('')).lower().split())
                
                new_words = recent_vocab - early_vocab
                vocab_growth = len(new_words) / len(early_vocab) if len(early_vocab) > 0 else 0
                
                # Text characteristics
                early_avg_length = early_data['Transcript_Length'].mean()
                recent_avg_length = recent_data['Transcript_Length'].mean()
                length_change = (recent_avg_length - early_avg_length) / early_avg_length if early_avg_length > 0 else 0
                
                # Qualifying language change
                early_qualifying = early_data['Customer_Qualifying_Count'].mean()
                recent_qualifying = recent_data['Customer_Qualifying_Count'].mean()
                qualifying_change = recent_qualifying - early_qualifying
                
                language_evolution[l2_cat] = {
                    'vocab_growth': vocab_growth,
                    'length_change': length_change,
                    'qualifying_change': qualifying_change,
                    'new_words_count': len(new_words)
                }
    
    print("Language Evolution Analysis:")
    for category, evolution in language_evolution.items():
        print(f"\n{category}:")
        print(f"  Vocabulary growth: {evolution['vocab_growth']:.1%}")
        print(f"  Length change: {evolution['length_change']:+.1%}")
        print(f"  Qualifying language change: {evolution['qualifying_change']:+.2f}")
        print(f"  New words: {evolution['new_words_count']}")
    
    return monthly_rule_performance, language_evolution

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
    
    # Original category overlap analysis (unchanged)
    category_pairs = df.groupby('variable5')['Prosodica L2'].apply(list).reset_index()
    category_pairs['Category_Count'] = category_pairs['Prosodica L2'].apply(len)
    
    # Find common category combinations
    multi_cat_pairs = category_pairs[category_pairs['Category_Count'] > 1]
    
    if len(multi_cat_pairs) > 0:
        # Extract all category combinations
        combinations = []
        for categories in multi_cat_pairs['Prosodica L2']:
            if len(categories) == 2:
                combinations.append(tuple(sorted(categories)))
        
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
                                lambda x: len(x) == 2 and combo[0] in x and combo[1] in x
                            )
                        ]['variable5']
                    )
                ]
                
                if len(combo_data) > 0:
                    combo_precision = combo_data['Is_TP'].mean()
                    print(f"    Precision: {combo_precision:.3f}")
    
    # NEW: Monthly Cross-Category Analysis
    create_monthly_insight_table(
        df,
        "Cross-Category Performance",
        ['Is_TP', 'Is_FP'],
        "Monthly cross-category analysis: Performance patterns across category interactions"
    )
    
    print("\n2.3 New Category Cannibalization Analysis")
    
    # Original new category analysis (unchanged)
    new_categories = df[df['Is_New_Category']]['Prosodica L2'].unique()
    
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

def content_pattern_analysis_enhanced(df):
    """Enhanced content pattern analysis with monthly breakdown - THE CORE ANALYSIS"""
    
    print("3.1 Average Transcript Length Comparison")
    
    # Original transcript length analysis (unchanged)
    tp_data = df[df['Primary Marker'] == 'TP']
    fp_data = df[df['Primary Marker'] == 'FP']
    
    length_comparison = pd.DataFrame({
        'Metric': ['Count', 'Avg_Transcript_Length', 'Median_Length', 'Std_Length'],
        'True_Positives': [
            len(tp_data),
            tp_data['Transcript_Length'].mean(),
            tp_data['Transcript_Length'].median(),
            tp_data['Transcript_Length'].std()
        ],
        'False_Positives': [
            len(fp_data),
            fp_data['Transcript_Length'].mean(),
            fp_data['Transcript_Length'].median(),
            fp_data['Transcript_Length'].std()
        ]
    })
    
    length_comparison['Difference'] = length_comparison['False_Positives'] - length_comparison['True_Positives']
    
    print("Transcript Length Analysis:")
    print(length_comparison.round(2))
    
    # Statistical significance
    if len(tp_data) > 0 and len(fp_data) > 0:
        from scipy.stats import mannwhitneyu
        
        try:
            _, p_value = mannwhitneyu(tp_data['Transcript_Length'].dropna(), 
                                    fp_data['Transcript_Length'].dropna(), 
                                    alternative='two-sided')
            print(f"Statistical significance (p-value): {p_value:.6f}")
            print(f"Significant difference: {'YES' if p_value < 0.05 else 'NO'}")
        except:
            print("Statistical test could not be performed")
    
    print("\n3.2 Agent to Customer Word Ratio Analysis")
    
    # Original word ratio analysis (unchanged)
    ratio_comparison = pd.DataFrame({
        'Metric': ['Avg_Customer_Words', 'Avg_Agent_Words', 'Avg_Customer_Agent_Ratio'],
        'True_Positives': [
            tp_data['Customer_Word_Count'].mean(),
            tp_data['Agent_Word_Count'].mean(),
            tp_data['Customer_Agent_Ratio'].mean()
        ],
        'False_Positives': [
            fp_data['Customer_Word_Count'].mean(),
            fp_data['Agent_Word_Count'].mean(),
            fp_data['Customer_Agent_Ratio'].mean()
        ]
    })
    
    ratio_comparison['Difference'] = ratio_comparison['False_Positives'] - ratio_comparison['True_Positives']
    
    print("Word Ratio Analysis:")
    print(ratio_comparison.round(3))
    
    # Interpret findings
    if ratio_comparison.loc[2, 'Difference'] > 0.2:  # Customer_Agent_Ratio difference
        print("  FINDING: FPs have higher customer-to-agent word ratio")
        print("  IMPLICATION: Customers speak more relative to agents in FPs")
    elif ratio_comparison.loc[2, 'Difference'] < -0.2:
        print("  FINDING: FPs have lower customer-to-agent word ratio")
        print("  IMPLICATION: Agents speak more relative to customers in FPs")
    
    print("\n3.3 Presence of Qualifying Indicators")
    
    # Original qualifying language analysis (unchanged)
    qualifying_comparison = pd.DataFrame({
        'Indicator': ['Negation_Count', 'Qualifying_Count', 'Question_Count', 'Exclamation_Count', 'Caps_Ratio'],
        'TP_Avg': [
            tp_data['Customer_Negation_Count'].mean(),
            tp_data['Customer_Qualifying_Count'].mean(),
            tp_data['Customer_Question_Count'].mean(),
            tp_data['Customer_Exclamation_Count'].mean(),
            tp_data['Customer_Caps_Ratio'].mean()
        ],
        'FP_Avg': [
            fp_data['Customer_Negation_Count'].mean(),
            fp_data['Customer_Qualifying_Count'].mean(),
            fp_data['Customer_Question_Count'].mean(),
            fp_data['Customer_Exclamation_Count'].mean(),
            fp_data['Customer_Caps_Ratio'].mean()
        ]
    })
    
    qualifying_comparison['Difference'] = qualifying_comparison['FP_Avg'] - qualifying_comparison['TP_Avg']
    qualifying_comparison['Risk_Factor'] = qualifying_comparison['FP_Avg'] / (qualifying_comparison['TP_Avg'] + 0.001)
    
    print("Qualifying Indicators Analysis:")
    print(qualifying_comparison.round(3))
    
    # NEW: CORE MONTHLY ANALYSIS - ALL QUALIFYING INDICATORS
    print("\n3.3.1 MONTHLY BREAKDOWN: ALL QUALIFYING INDICATORS")
    create_monthly_insight_table(
        df,
        "Complete Qualifying Indicators Analysis",
        ['Customer_Negation_Count', 'Customer_Qualifying_Count', 'Customer_Question_Count', 'Customer_Caps_Ratio'],
        "COMPREHENSIVE: Monthly breakdown of all qualifying indicators - negation, qualifying words, questions, caps"
    )
    
    print("\n3.4 Precision of Qualifying Words Analysis")
    
    # Original advanced pattern analysis (unchanged)
    advanced_patterns = {
        'Strong_Negation': r'\b(absolutely not|definitely not|certainly not|never ever)\b',
        'Weak_Negation': r'\b(not really|not quite|not exactly|hardly|barely)\b',
        'Uncertainty': r'\b(i think|i believe|i guess|maybe|perhaps|possibly)\b',
        'Frustration': r'\b(frustrated|annoyed|upset|angry|mad|ridiculous|stupid)\b',
        'Politeness': r'\b(please|thank you|thanks|appreciate|grateful)\b',
        'Agent_Explanations': r'\b(let me explain|what this means|for example|in other words)\b',
        'Hypotheticals': r'\b(if i|suppose i|what if|let\'s say|imagine if)\b'
    }
    
    pattern_analysis = []
    
    for pattern_name, pattern in advanced_patterns.items():
        tp_matches = tp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        fp_matches = fp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        
        tp_rate = tp_matches.mean() * 100 if len(tp_data) > 0 else 0
        fp_rate = fp_matches.mean() * 100 if len(fp_data) > 0 else 0
        
        # Calculate risk factor
        risk_factor = fp_rate / max(tp_rate, 0.1)  # Avoid division by zero
        
        pattern_analysis.append({
            'Pattern': pattern_name,
            'TP_Rate': tp_rate,
            'FP_Rate': fp_rate,
            'Difference': fp_rate - tp_rate,
            'Risk_Factor': risk_factor
        })
    
    pattern_df = pd.DataFrame(pattern_analysis).sort_values('Risk_Factor', ascending=False)
    
    print("Advanced Pattern Analysis (% of transcripts containing pattern):")
    print(f"{'Pattern':<20} {'TP%':<8} {'FP%':<8} {'Diff':<8} {'Risk':<8}")
    print("-" * 60)
    
    for _, row in pattern_df.iterrows():
        risk_level = "HIGH" if row['Risk_Factor'] > 2 else "MED" if row['Risk_Factor'] > 1.5 else "LOW"
        print(f"{row['Pattern']:<20} {row['TP_Rate']:<8.1f} {row['FP_Rate']:<8.1f} "
              f"{row['Difference']:<8.1f} {risk_level:<8}")
    
    # Key insights
    print("\nKey Content Pattern Insights:")
    
    high_risk_patterns = pattern_df[pattern_df['Risk_Factor'] > 2]
    if len(high_risk_patterns) > 0:
        print("High-risk patterns (>2x more likely in FPs):")
        for _, pattern in high_risk_patterns.iterrows():
            print(f"  - {pattern['Pattern']}: {pattern['Risk_Factor']:.1f}x risk")
    
    # Text length insights
    length_diff = length_comparison.loc[1, 'Difference']  # Avg_Transcript_Length difference
    if abs(length_diff) > 100:
        direction = "longer" if length_diff > 0 else "shorter"
        print(f"  - FPs are {abs(length_diff):.0f} characters {direction} on average")
    
    return length_comparison, ratio_comparison, qualifying_comparison, pattern_df
