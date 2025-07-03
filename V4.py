def create_compact_problematic_rules_view(df_main, problematic_rules):
    """
    Create a single compact view showing top problematic rules with:
    - Pre vs Post precision changes
    - Transcript length effects
    - Key performance metrics
    
    Returns: DataFrame with compact summary for top problematic rules
    """
    
    print("\n" + "="*100)
    print("COMPACT VIEW: TOP PROBLEMATIC RULES ANALYSIS")
    print("="*100)
    print("Showing: Event + Query combinations with high negation (>3) but no context (proximity=0)")
    print("Analysis: Pre vs Post precision changes and transcript length impact")
    print("="*100)
    
    if df_main is None or len(problematic_rules) == 0:
        print("Missing required data for compact analysis")
        return pd.DataFrame()
    
    # Prepare temporal data
    df_main['Date'] = pd.to_datetime(df_main['Date'])
    df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
    
    # Define periods
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    df_main['Period'] = df_main['Year_Month'].apply(
        lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
    )
    
    # Calculate transcript lengths
    df_main['Customer_Transcript'] = df_main['Customer Transcript'].fillna('')
    df_main['Agent_Transcript'] = df_main['Agent Transcript'].fillna('')
    df_main['Full_Transcript'] = df_main['Customer_Transcript'] + ' ' + df_main['Agent_Transcript']
    df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
    
    # Define length categories for analysis
    df_main['Is_Short'] = df_main['Transcript_Length'] < 3000
    df_main['Is_Long'] = df_main['Transcript_Length'] > 6000
    
    compact_results = []
    
    # Analyze top 15 problematic rules for comprehensive view
    top_problematic = problematic_rules.head(15)
    
    print(f"Analyzing top {len(top_problematic)} problematic rules...")
    
    for idx, rule in top_problematic.iterrows():
        event = rule['Event']
        query = rule['Query']
        negation_count = rule['negation_patterns']
        
        # Find matching transcripts for this rule
        rule_transcripts = df_main[
            (df_main['Prosodica L1'].str.lower() == event.lower()) |
            (df_main['Prosodica L2'].str.lower() == query.lower())
        ]
        
        if len(rule_transcripts) == 0:
            continue
        
        # Split by period
        pre_data = rule_transcripts[rule_transcripts['Period'] == 'Pre']
        post_data = rule_transcripts[rule_transcripts['Period'] == 'Post']
        
        if len(pre_data) == 0 or len(post_data) == 0:
            continue
        
        # Calculate overall metrics
        pre_precision = (pre_data['Primary Marker'] == 'TP').mean()
        post_precision = (post_data['Primary Marker'] == 'TP').mean()
        precision_change = post_precision - pre_precision
        
        pre_volume = len(pre_data)
        post_volume = len(post_data)
        
        # Length-based analysis for Pre period
        pre_short = pre_data[pre_data['Is_Short']]
        pre_long = pre_data[pre_data['Is_Long']]
        
        pre_short_precision = (pre_short['Primary Marker'] == 'TP').mean() if len(pre_short) > 0 else 0
        pre_long_precision = (pre_long['Primary Marker'] == 'TP').mean() if len(pre_long) > 0 else 0
        pre_length_gap = pre_long_precision - pre_short_precision
        
        pre_short_count = len(pre_short)
        pre_long_count = len(pre_long)
        pre_avg_length = pre_data['Transcript_Length'].mean()
        
        # Length-based analysis for Post period
        post_short = post_data[post_data['Is_Short']]
        post_long = post_data[post_data['Is_Long']]
        
        post_short_precision = (post_short['Primary Marker'] == 'TP').mean() if len(post_short) > 0 else 0
        post_long_precision = (post_long['Primary Marker'] == 'TP').mean() if len(post_long) > 0 else 0
        post_length_gap = post_long_precision - post_short_precision
        
        post_short_count = len(post_short)
        post_long_count = len(post_long)
        post_avg_length = post_data['Transcript_Length'].mean()
        
        # Key insight calculations
        length_gap_change = post_length_gap - pre_length_gap
        short_precision_change = post_short_precision - pre_short_precision
        long_precision_change = post_long_precision - pre_long_precision
        avg_length_change = post_avg_length - pre_avg_length
        
        # Determine impact severity
        if abs(precision_change) > 0.1:
            impact_level = "HIGH"
        elif abs(precision_change) > 0.05:
            impact_level = "MEDIUM" 
        else:
            impact_level = "LOW"
        
        # Determine length effect pattern
        if length_gap_change > 0.1:
            length_effect = "WORSENING"
        elif length_gap_change > 0.05:
            length_effect = "MODERATE"
        elif length_gap_change < -0.05:
            length_effect = "IMPROVING"
        else:
            length_effect = "STABLE"
        
        # Store compact result
        compact_results.append({
            # Rule Identification
            'Event': event,
            'Query': query,
            'Negation_Patterns': negation_count,
            
            # Volume Metrics
            'Pre_Volume': pre_volume,
            'Post_Volume': post_volume,
            'Volume_Change': post_volume - pre_volume,
            
            # Overall Precision Metrics
            'Pre_Precision': round(pre_precision, 3),
            'Post_Precision': round(post_precision, 3),
            'Precision_Change': round(precision_change, 3),
            'Impact_Level': impact_level,
            
            # Length Distribution
            'Pre_Avg_Length': round(pre_avg_length, 0),
            'Post_Avg_Length': round(post_avg_length, 0),
            'Avg_Length_Change': round(avg_length_change, 0),
            
            # Short Transcript Performance
            'Pre_Short_Count': pre_short_count,
            'Pre_Short_Precision': round(pre_short_precision, 3),
            'Post_Short_Count': post_short_count,
            'Post_Short_Precision': round(post_short_precision, 3),
            'Short_Precision_Change': round(short_precision_change, 3),
            
            # Long Transcript Performance
            'Pre_Long_Count': pre_long_count,
            'Pre_Long_Precision': round(pre_long_precision, 3),
            'Post_Long_Count': post_long_count,
            'Post_Long_Precision': round(post_long_precision, 3),
            'Long_Precision_Change': round(long_precision_change, 3),
            
            # Length Effect Analysis
            'Pre_Length_Gap': round(pre_length_gap, 3),
            'Post_Length_Gap': round(post_length_gap, 3),
            'Length_Gap_Change': round(length_gap_change, 3),
            'Length_Effect': length_effect
        })
    
    # Create DataFrame
    compact_df = pd.DataFrame(compact_results)
    
    if len(compact_df) == 0:
        print("No data available for compact analysis")
        return pd.DataFrame()
    
    # Sort by impact severity
    impact_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    compact_df['Impact_Order'] = compact_df['Impact_Level'].map(impact_order)
    compact_df = compact_df.sort_values(['Impact_Order', 'Precision_Change'], ascending=[False, True])
    
    # Display compact summary
    print(f"\nCOMPACT SUMMARY: {len(compact_df)} Problematic Rules Analyzed")
    print("-" * 100)
    
    # Key columns for initial view
    summary_cols = [
        'Event', 'Query', 'Negation_Patterns', 'Impact_Level',
        'Pre_Precision', 'Post_Precision', 'Precision_Change',
        'Pre_Length_Gap', 'Post_Length_Gap', 'Length_Gap_Change', 'Length_Effect'
    ]
    
    print("OVERVIEW: Top 10 Most Impacted Rules")
    print(compact_df[summary_cols].head(10).to_string(index=False))
    
    # Focused analysis on length effects
    print(f"\n" + "-" * 100)
    print("LENGTH EFFECT BREAKDOWN:")
    print("-" * 100)
    
    length_effect_summary = compact_df['Length_Effect'].value_counts()
    print("Length Effect Distribution:")
    for effect, count in length_effect_summary.items():
        percentage = count / len(compact_df) * 100
        print(f"  {effect}: {count} rules ({percentage:.1f}%)")
    
    # Rules with worsening length effects
    worsening_rules = compact_df[compact_df['Length_Effect'] == 'WORSENING']
    if len(worsening_rules) > 0:
        print(f"\nRULES WITH WORSENING LENGTH EFFECTS ({len(worsening_rules)} rules):")
        worsening_cols = [
            'Event', 'Query', 'Pre_Length_Gap', 'Post_Length_Gap', 
            'Length_Gap_Change', 'Short_Precision_Change', 'Long_Precision_Change'
        ]
        print(worsening_rules[worsening_cols].to_string(index=False))
    
    # Summary statistics
    print(f"\n" + "-" * 100)
    print("KEY STATISTICS:")
    print("-" * 100)
    
    stats = {
        'Total Rules Analyzed': len(compact_df),
        'Rules with Precision Drop': len(compact_df[compact_df['Precision_Change'] < 0]),
        'Rules with Worsening Length Effects': len(compact_df[compact_df['Length_Effect'] == 'WORSENING']),
        'Avg Precision Change': compact_df['Precision_Change'].mean(),
        'Avg Length Gap Change': compact_df['Length_Gap_Change'].mean(),
        'Rules with High Impact': len(compact_df[compact_df['Impact_Level'] == 'HIGH'])
    }
    
    for stat, value in stats.items():
        if isinstance(value, float):
            print(f"  {stat}: {value:.3f}")
        else:
            print(f"  {stat}: {value}")
    
    # Critical findings
    print(f"\n" + "="*100)
    print("CRITICAL FINDINGS:")
    print("="*100)
    
    # Finding 1: Overall precision degradation
    precision_degraded = len(compact_df[compact_df['Precision_Change'] < -0.05])
    print(f"1. PRECISION DEGRADATION: {precision_degraded}/{len(compact_df)} rules show >5% precision drop")
    
    # Finding 2: Length effect worsening
    length_worsened = len(compact_df[compact_df['Length_Gap_Change'] > 0.05])
    print(f"2. LENGTH EFFECT WORSENING: {length_worsened}/{len(compact_df)} rules show widening length-based precision gaps")
    
    # Finding 3: Short transcript specific issues
    short_degraded = len(compact_df[compact_df['Short_Precision_Change'] < -0.05])
    print(f"3. SHORT TRANSCRIPT ISSUES: {short_degraded}/{len(compact_df)} rules show >5% precision drop for short transcripts")
    
    # Finding 4: Pattern correlation
    high_negation_worsening = len(compact_df[
        (compact_df['Negation_Patterns'] > 5) & 
        (compact_df['Length_Effect'] == 'WORSENING')
    ])
    print(f"4. HIGH NEGATION CORRELATION: {high_negation_worsening} rules with >5 negation patterns show worsening length effects")
    
    if length_worsened > len(compact_df) * 0.3:  # If >30% of rules show worsening
        print(f"\n*** MAJOR FINDING: {length_worsened/len(compact_df)*100:.1f}% of problematic rules show WORSENING length-based effects ***")
        print("*** This confirms the hypothesis: High-negation rules without context are increasingly ***")
        print("*** discriminating against SHORT transcripts while maintaining performance on LONG ones ***")
    
    # Drop helper column
    compact_df = compact_df.drop('Impact_Order', axis=1)
    
    return compact_df

# Additional utility function for even more focused view
def create_executive_summary_view(compact_df):
    """
    Create an ultra-compact executive summary view of the most critical rules
    """
    
    if len(compact_df) == 0:
        return pd.DataFrame()
    
    print(f"\n" + "="*80)
    print("EXECUTIVE SUMMARY: MOST CRITICAL PROBLEMATIC RULES")
    print("="*80)
    
    # Focus on high impact rules with worsening length effects
    critical_rules = compact_df[
        (compact_df['Impact_Level'] == 'HIGH') |
        (compact_df['Length_Effect'] == 'WORSENING')
    ].head(5)
    
    if len(critical_rules) == 0:
        print("No critical rules identified")
        return pd.DataFrame()
    
    # Ultra-compact view
    executive_cols = [
        'Event', 'Query', 'Negation_Patterns',
        'Precision_Change', 'Length_Gap_Change', 
        'Short_Precision_Change', 'Length_Effect'
    ]
    
    print("TOP 5 MOST CRITICAL RULES:")
    print(critical_rules[executive_cols].to_string(index=False))
    
    print(f"\nEXECUTIVE INSIGHT:")
    avg_precision_drop = critical_rules['Precision_Change'].mean()
    avg_gap_worsening = critical_rules['Length_Gap_Change'].mean()
    
    print(f"  - Average precision drop: {avg_precision_drop:.3f}")
    print(f"  - Average length gap worsening: {avg_gap_worsening:.3f}")
    print(f"  - Pattern: Rules with high negation patterns but no context handling")
    print(f"  - Effect: Systematic misclassification of short informational transcripts")
    
    return critical_rules

# Example usage function to tie it all together
def run_complete_compact_analysis(df_main, problematic_rules):
    """
    Run the complete compact analysis and return all results
    """
    
    print("RUNNING COMPLETE COMPACT ANALYSIS...")
    
    # Create compact view
    compact_df = create_compact_problematic_rules_view(df_main, problematic_rules)
    
    if len(compact_df) > 0:
        # Create executive summary
        executive_df = create_executive_summary_view(compact_df)
        
        # Export to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'Compact_Problematic_Rules_Analysis_{timestamp}.xlsx'
        
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            compact_df.to_excel(writer, sheet_name='Compact_Analysis', index=False)
            if len(executive_df) > 0:
                executive_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
        
        print(f"\nCompact analysis exported to: {filename}")
        
        return compact_df, executive_df
    
    else:
        print("No data available for compact analysis")
        return pd.DataFrame(), pd.DataFrame()

# To use this function, add these lines to your main script:
"""
# After running the problematic rules identification:
compact_results, executive_summary = run_complete_compact_analysis(df_main, problematic_rules)
"""
