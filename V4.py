def create_compact_problematic_rules_view_formatted(df_main, problematic_rules):
    """
    Create a compact view in the exact requested format:
    Event | Query | Negation Patterns | Short_Precision (Pre/Post/Change) | Medium_Precision (Pre/Post/Change) | Long_Precision (Pre/Post/Change)
    """
    
    print("\n" + "="*120)
    print("COMPACT ANALYSIS: PROBLEMATIC RULES - LENGTH-BASED PRECISION BREAKDOWN")
    print("="*120)
    print("Format: Event | Query | Negation Patterns | Short/Medium/Long Precision (Pre→Post→Change)")
    print("="*120)
    
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
    
    # Define length categories based on distribution analysis
    df_main['Length_Category'] = pd.cut(
        df_main['Transcript_Length'],
        bins=[0, 4000, 8000, float('inf')],
        labels=['Short', 'Medium', 'Long']
    )
    
    print("Length Category Distribution:")
    length_dist = df_main['Length_Category'].value_counts()
    for category, count in length_dist.items():
        percentage = count / len(df_main) * 100
        print(f"  {category}: {count:,} transcripts ({percentage:.1f}%)")
    
    compact_results = []
    
    # Analyze top 15 problematic rules
    top_problematic = problematic_rules.head(15)
    
    print(f"\nAnalyzing top {len(top_problematic)} problematic rules...")
    
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
        
        # Initialize result row
        result_row = {
            'Event': event,
            'Query': query,
            'Negation_Patterns': negation_count
        }
        
        # Calculate precision for each length category in each period
        for period_name, period_data in [('Pre', pre_data), ('Post', post_data)]:
            
            for length_cat in ['Short', 'Medium', 'Long']:
                # Filter data for this length category
                length_data = period_data[period_data['Length_Category'] == length_cat]
                
                if len(length_data) > 0:
                    precision = (length_data['Primary Marker'] == 'TP').mean()
                else:
                    precision = 0.0  # No data available
                
                # Store in result row
                result_row[f'{length_cat}_{period_name}'] = round(precision, 3)
        
        # Calculate precision changes
        for length_cat in ['Short', 'Medium', 'Long']:
            pre_key = f'{length_cat}_Pre'
            post_key = f'{length_cat}_Post'
            change_key = f'{length_cat}_Precision_Change'
            
            if pre_key in result_row and post_key in result_row:
                change = result_row[post_key] - result_row[pre_key]
                result_row[change_key] = round(change, 3)
            else:
                result_row[change_key] = 0.0
        
        compact_results.append(result_row)
    
    # Create DataFrame
    compact_df = pd.DataFrame(compact_results)
    
    if len(compact_df) == 0:
        print("No data available for compact analysis")
        return pd.DataFrame()
    
    # Reorder columns to match requested format
    column_order = [
        'Event', 'Query', 'Negation_Patterns',
        'Short_Pre', 'Short_Post', 'Short_Precision_Change',
        'Medium_Pre', 'Medium_Post', 'Medium_Precision_Change',
        'Long_Pre', 'Long_Post', 'Long_Precision_Change'
    ]
    
    # Ensure all columns exist
    for col in column_order:
        if col not in compact_df.columns:
            compact_df[col] = 0.0
    
    compact_df = compact_df[column_order]
    
    # Rename columns for display
    compact_df.columns = [
        'Event', 'Query', 'Negation Patterns',
        'Pre', 'Post', 'Precision_Change',  # Short
        'Pre', 'Post', 'Precision_Change',  # Medium  
        'Pre', 'Post', 'Precision_Change'   # Long
    ]
    
    # Create multi-level column headers for better display
    display_df = compact_df.copy()
    
    # Sort by biggest precision drops in short transcripts
    display_df = display_df.sort_values('Short_Precision_Change')
    
    print(f"\nCOMPACT RESULTS: {len(display_df)} Problematic Rules")
    print("-" * 120)
    
    # Custom display with proper headers
    print("                                                    Short_Precision              Medium_Precision             Long_Precision")
    print("Event                    Query                   Neg  Pre   Post  Change    Pre   Post  Change    Pre   Post  Change")
    print("-" * 120)
    
    for idx, row in display_df.head(10).iterrows():
        event_short = row['Event'][:20].ljust(20)
        query_short = row['Query'][:20].ljust(20)
        neg_patterns = str(int(row['Negation Patterns'])).rjust(3)
        
        # Short precision
        short_pre = f"{row.iloc[3]:.3f}".rjust(5)
        short_post = f"{row.iloc[4]:.3f}".rjust(5)
        short_change = f"{row.iloc[5]:+.3f}".rjust(7)
        
        # Medium precision  
        medium_pre = f"{row.iloc[6]:.3f}".rjust(5)
        medium_post = f"{row.iloc[7]:.3f}".rjust(5)
        medium_change = f"{row.iloc[8]:+.3f}".rjust(7)
        
        # Long precision
        long_pre = f"{row.iloc[9]:.3f}".rjust(5)
        long_post = f"{row.iloc[10]:.3f}".rjust(5)
        long_change = f"{row.iloc[11]:+.3f}".rjust(7)
        
        print(f"{event_short} {query_short} {neg_patterns} {short_pre} {short_post} {short_change}  {medium_pre} {medium_post} {medium_change}  {long_pre} {long_post} {long_change}")
    
    # Statistical Summary
    print(f"\n" + "-" * 120)
    print("STATISTICAL SUMMARY:")
    print("-" * 120)
    
    # Calculate summary statistics
    short_changes = display_df.iloc[:, 5]  # Short precision changes
    medium_changes = display_df.iloc[:, 8]  # Medium precision changes  
    long_changes = display_df.iloc[:, 11]  # Long precision changes
    
    print(f"Average Precision Changes:")
    print(f"  Short Transcripts:  {short_changes.mean():+.3f} (Range: {short_changes.min():+.3f} to {short_changes.max():+.3f})")
    print(f"  Medium Transcripts: {medium_changes.mean():+.3f} (Range: {medium_changes.min():+.3f} to {medium_changes.max():+.3f})")
    print(f"  Long Transcripts:   {long_changes.mean():+.3f} (Range: {long_changes.min():+.3f} to {long_changes.max():+.3f})")
    
    # Count rules with significant drops
    short_drops = (short_changes < -0.05).sum()
    medium_drops = (medium_changes < -0.05).sum()
    long_drops = (long_changes < -0.05).sum()
    
    print(f"\nRules with >5% Precision Drop:")
    print(f"  Short Transcripts:  {short_drops}/{len(display_df)} rules ({short_drops/len(display_df)*100:.1f}%)")
    print(f"  Medium Transcripts: {medium_drops}/{len(display_df)} rules ({medium_drops/len(display_df)*100:.1f}%)")
    print(f"  Long Transcripts:   {long_drops}/{len(display_df)} rules ({long_drops/len(display_df)*100:.1f}%)")
    
    # Key insights
    print(f"\n" + "="*120)
    print("KEY INSIGHTS:")
    print("="*120)
    
    if short_drops > medium_drops and short_drops > long_drops:
        print("1. *** SHORT TRANSCRIPTS MOST AFFECTED *** - Confirms length-based discrimination hypothesis")
    
    if short_changes.mean() < medium_changes.mean() and short_changes.mean() < long_changes.mean():
        print("2. *** SHORT TRANSCRIPTS SHOW WORST PERFORMANCE DEGRADATION *** - Pattern matches expectations")
    
    worst_rules = display_df.head(3)
    print(f"3. *** TOP 3 MOST PROBLEMATIC RULES ***:")
    for idx, rule in worst_rules.iterrows():
        short_change = rule.iloc[5]
        print(f"   {rule['Event']} | {rule['Query']} | Short precision change: {short_change:+.3f}")
    
    # Return properly formatted DataFrame for export
    export_df = pd.DataFrame()
    export_df['Event'] = display_df['Event']
    export_df['Query'] = display_df['Query'] 
    export_df['Negation_Patterns'] = display_df['Negation Patterns']
    
    # Short precision columns
    export_df['Short_Pre'] = display_df.iloc[:, 3]
    export_df['Short_Post'] = display_df.iloc[:, 4]
    export_df['Short_Precision_Change'] = display_df.iloc[:, 5]
    
    # Medium precision columns
    export_df['Medium_Pre'] = display_df.iloc[:, 6]
    export_df['Medium_Post'] = display_df.iloc[:, 7]
    export_df['Medium_Precision_Change'] = display_df.iloc[:, 8]
    
    # Long precision columns
    export_df['Long_Pre'] = display_df.iloc[:, 9]
    export_df['Long_Post'] = display_df.iloc[:, 10]
    export_df['Long_Precision_Change'] = display_df.iloc[:, 11]
    
    return export_df

formatted_results = create_compact_problematic_rules_view_formatted(df_main, problematic_rules)
