def create_unified_feature_dataframe(df_main, df_enhanced, df_qualifying):
    """
    Create a unified dataframe with all original columns and feature-engineered columns
    aggregated at the variable5 level
    """
    
    print("="*80)
    print("CREATING UNIFIED FEATURE DATAFRAME")
    print("="*80)
    
    # 1. Start with original columns aggregated at variable5 level
    print("1. AGGREGATING ORIGINAL COLUMNS AT VARIABLE5 LEVEL")
    print("-" * 50)
    
    # Identify categorical and numerical columns
    categorical_cols = [
        'Prosodica L1', 'Prosodica L2', 'Primary L1', 'Primary L2', 
        'Primary Marker', 'Secondary L1', 'Secondary L2', 'Secondary Marker',
        'Primary Rater Name', 'Year_Month', 'DayOfWeek', 'Period'
    ]
    
    # Define aggregation functions for different column types
    agg_functions = {}
    
    # For categorical columns, take the most frequent value (mode)
    for col in categorical_cols:
        if col in df_main.columns:
            agg_functions[col] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    
    # For text columns, concatenate
    text_cols = ['Customer Transcript', 'Agent Transcript']
    for col in text_cols:
        if col in df_main.columns:
            agg_functions[col] = lambda x: ' '.join(x.astype(str))
    
    # For date columns, take the first value
    if 'Date' in df_main.columns:
        agg_functions['Date'] = 'first'
    
    # For binary/numerical columns, take mean or sum as appropriate
    binary_cols = [
        'Is_TP', 'Is_FP', 'Has_Secondary_Validation', 'Is_Holiday_Season', 
        'Is_Month_End', 'Is_New_Category'
    ]
    for col in binary_cols:
        if col in df_main.columns:
            agg_functions[col] = 'mean'  # Will give the proportion
    
    # For other numerical columns, take mean
    numerical_cols = [
        'Transcript_Length', 'Customer_Word_Count', 'Agent_Word_Count',
        'Customer_Agent_Ratio', 'Customer_Question_Count', 'Customer_Exclamation_Count',
        'Customer_Caps_Ratio', 'Customer_Negation_Count', 'Agent_Negation_Count',
        'Customer_Qualifying_Count', 'WeekOfMonth', 'Quarter', 'Category_Age_Days'
    ]
    for col in numerical_cols:
        if col in df_main.columns:
            agg_functions[col] = 'mean'
    
    # Perform aggregation
    unified_base = df_main.groupby('variable5').agg(agg_functions).reset_index()
    
    print(f"Base aggregated dataframe shape: {unified_base.shape}")
    
    # MODIFICATION 1: Add One-Hot Encoded columns for Prosodica L1 and L2
    print("\n1.1. ADDING ONE-HOT ENCODED CATEGORY COLUMNS")
    print("-" * 50)
    
    # Get distinct values for Prosodica L1 and L2
    distinct_l1_values = df_main['Prosodica L1'].dropna().unique()
    distinct_l2_values = df_main['Prosodica L2'].dropna().unique()
    
    print(f"Found {len(distinct_l1_values)} distinct L1 categories")
    print(f"Found {len(distinct_l2_values)} distinct L2 categories")
    
    # Create L1 one-hot encoded features
    l1_features = {}
    for l1_value in distinct_l1_values:
        # Clean the value for column name (remove special characters)
        clean_l1_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(l1_value))
        col_name = f'L1_{clean_l1_name}'
        
        # For each variable5, check if it has this L1 category
        l1_features[col_name] = df_main.groupby('variable5')['Prosodica L1'].apply(
            lambda x: 1 if l1_value in x.values else 0
        )
    
    # Create L2 one-hot encoded features
    l2_features = {}
    for l2_value in distinct_l2_values:
        # Clean the value for column name (remove special characters)
        clean_l2_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(l2_value))
        col_name = f'L2_{clean_l2_name}'
        
        # For each variable5, check if it has this L2 category
        l2_features[col_name] = df_main.groupby('variable5')['Prosodica L2'].apply(
            lambda x: 1 if l2_value in x.values else 0
        )
    
    # Convert to DataFrames and merge with unified_base
    l1_features_df = pd.DataFrame(l1_features).reset_index()
    l2_features_df = pd.DataFrame(l2_features).reset_index()
    
    unified_base = unified_base.merge(l1_features_df, on='variable5', how='left')
    unified_base = unified_base.merge(l2_features_df, on='variable5', how='left')
    
    print(f"Added {len(l1_features)} L1 category features")
    print(f"Added {len(l2_features)} L2 category features")
    print(f"Updated dataframe shape: {unified_base.shape}")
    
    # 2. Add enhanced features from agent contamination analysis
    print("\n2. ADDING AGENT CONTAMINATION FEATURES")
    print("-" * 50)
    
    if 'Agent_Contamination_Score' in df_enhanced.columns:
        agent_features = df_enhanced.groupby('variable5').agg({
            'Agent_Contamination_Score': 'mean',
            'Has_Agent_Contamination': 'mean'
        }).reset_index()
        
        agent_features.columns = ['variable5', 'Avg_Agent_Contamination_Score', 'Agent_Contamination_Rate']
        
        unified_base = unified_base.merge(agent_features, on='variable5', how='left')
        print(f"Added agent contamination features. Shape: {unified_base.shape}")
    
    # 3. Add qualifying language features
    print("\n3. ADDING QUALIFYING LANGUAGE FEATURES")
    print("-" * 50)
    
    # Get all qualifying language columns
    qualifying_cols = [col for col in df_qualifying.columns if any(pattern in col for pattern in 
                      ['Customer_Uncertainty', 'Customer_Hedging', 'Customer_Approximation', 
                       'Customer_Conditional', 'Customer_Doubt', 'Customer_Politeness',
                       'Agent_Uncertainty', 'Agent_Hedging', 'Agent_Approximation',
                       'Agent_Conditional', 'Agent_Doubt', 'Agent_Politeness'])]
    
    if qualifying_cols:
        qualifying_agg = {col: 'mean' for col in qualifying_cols}
        qualifying_features = df_qualifying.groupby('variable5').agg(qualifying_agg).reset_index()
        
        unified_base = unified_base.merge(qualifying_features, on='variable5', how='left')
        print(f"Added {len(qualifying_cols)} qualifying language features. Shape: {unified_base.shape}")
    
    # 4. Add derived conversation-level features
    print("\n4. ADDING DERIVED CONVERSATION-LEVEL FEATURES")
    print("-" * 50)
    
    # Count of categories per conversation
    category_counts = df_main.groupby('variable5').agg({
        'Prosodica L1': 'nunique',
        'Prosodica L2': 'nunique',
        'UUID': 'count'
    }).reset_index()
    
    category_counts.columns = ['variable5', 'Unique_L1_Categories', 'Unique_L2_Categories', 'Total_UUIDs']
    
    unified_base = unified_base.merge(category_counts, on='variable5', how='left')
    
    # Add multi-category flags
    unified_base['Is_Multi_L1_Category'] = (unified_base['Unique_L1_Categories'] > 1).astype(int)
    unified_base['Is_Multi_L2_Category'] = (unified_base['Unique_L2_Categories'] > 1).astype(int)
    
    # 5. Add validation consistency features (if available)
    print("\n5. ADDING VALIDATION CONSISTENCY FEATURES")
    print("-" * 50)
    
    if 'Primary_Secondary_Agreement' in df_main.columns:
        validation_features = df_main.groupby('variable5').agg({
            'Primary_Secondary_Agreement': ['mean', 'count', 'std']
        }).reset_index()
        
        validation_features.columns = ['variable5', 'Avg_Validation_Agreement', 'Validation_Count', 'Validation_Std']
        validation_features['Has_Validation_Data'] = (validation_features['Validation_Count'] > 0).astype(int)
        
        unified_base = unified_base.merge(validation_features, on='variable5', how='left')
        print(f"Added validation consistency features. Shape: {unified_base.shape}")
    
    # 6. Add temporal and contextual features
    print("\n6. ADDING TEMPORAL AND CONTEXTUAL FEATURES")
    print("-" * 50)
    
    # Calculate conversation complexity metrics
    unified_base['Conversation_Complexity_Score'] = (
        unified_base['Total_UUIDs'] * 0.3 +
        unified_base['Unique_L2_Categories'] * 0.4 +
        (unified_base['Transcript_Length'] / 1000) * 0.3
    )
    
    # Calculate precision risk score
    unified_base['Precision_Risk_Score'] = 0
    
    # Add risk from negations
    if 'Customer_Negation_Count' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Customer_Negation_Count'] * 0.1
    
    # Add risk from agent contamination
    if 'Agent_Contamination_Rate' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Agent_Contamination_Rate'] * 0.3
    
    # Add risk from qualifying language
    if 'Customer_Doubt_Count' in unified_base.columns:
        unified_base['Precision_Risk_Score'] += unified_base['Customer_Doubt_Count'] * 0.2
    
    # Add risk from multi-category
    unified_base['Precision_Risk_Score'] += unified_base['Is_Multi_L2_Category'] * 0.15
    
    # Normalize risk score
    max_risk = unified_base['Precision_Risk_Score'].max()
    if max_risk > 0:
        unified_base['Precision_Risk_Score_Normalized'] = unified_base['Precision_Risk_Score'] / max_risk
    else:
        unified_base['Precision_Risk_Score_Normalized'] = 0
    
    # 7. Add feature summary statistics
    print("\n7. ADDING FEATURE SUMMARY STATISTICS")
    print("-" * 50)
    
    # Calculate feature counts and ratios
    text_features = ['Customer_Negation_Count', 'Customer_Qualifying_Count', 'Customer_Question_Count']
    
    for feature in text_features:
        if feature in unified_base.columns:
            # Feature density per 1000 characters
            unified_base[f'{feature}_Density'] = (
                unified_base[feature] / (unified_base['Transcript_Length'] / 1000)
            ).fillna(0)
    
    # Customer engagement score
    engagement_features = ['Customer_Question_Count', 'Customer_Word_Count']
    if all(feature in unified_base.columns for feature in engagement_features):
        unified_base['Customer_Engagement_Score'] = (
            unified_base['Customer_Question_Count'] * 0.4 +
            (unified_base['Customer_Word_Count'] / 100) * 0.6
        )
    
    # 8. Final data quality checks and summary
    print("\n8. FINAL DATA QUALITY CHECKS AND SUMMARY")
    print("-" * 50)
    
    # Check for missing values
    missing_summary = unified_base.isnull().sum()
    missing_percentage = (missing_summary / len(unified_base)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_summary.index,
        'Missing_Count': missing_summary.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print("Columns with missing values:")
        print(missing_df.head(10))
    else:
        print("No missing values detected in unified dataframe")
    
    # Feature categories summary
    original_features = [col for col in unified_base.columns if col in df_main.columns or col == 'variable5']
    engineered_features = [col for col in unified_base.columns if col not in original_features]
    
    print(f"\nFeature Summary:")
    print(f"Total features: {len(unified_base.columns)}")
    print(f"Original features: {len(original_features)}")
    print(f"Engineered features: {len(engineered_features)}")
    print(f"L1 category features: {len([col for col in unified_base.columns if col.startswith('L1_')])}")
    print(f"L2 category features: {len([col for col in unified_base.columns if col.startswith('L2_')])}")
    print(f"Total conversations: {len(unified_base)}")
    
    print(f"\nEngineered Feature Categories:")
    print(f"- Category one-hot features: {len([col for col in engineered_features if col.startswith('L1_') or col.startswith('L2_')])}")
    print(f"- Agent contamination features: {len([col for col in engineered_features if 'Agent' in col and 'Contamination' in col])}")
    print(f"- Qualifying language features: {len([col for col in engineered_features if any(pattern in col for pattern in ['Uncertainty', 'Hedging', 'Doubt', 'Politeness'])])}")
    print(f"- Category complexity features: {len([col for col in engineered_features if 'Category' in col or 'Multi' in col])}")
    print(f"- Validation features: {len([col for col in engineered_features if 'Validation' in col])}")
    print(f"- Risk scoring features: {len([col for col in engineered_features if 'Risk' in col or 'Score' in col])}")
    print(f"- Density features: {len([col for col in engineered_features if 'Density' in col])}")
    
    # Data distribution summary
    print(f"\nData Distribution Summary:")
    if 'Is_TP' in unified_base.columns:
        tp_rate = unified_base['Is_TP'].mean()
        print(f"Overall TP rate: {tp_rate:.3f}")
    
    if 'Is_Multi_L2_Category' in unified_base.columns:
        multi_cat_rate = unified_base['Is_Multi_L2_Category'].mean()
        print(f"Multi-category conversations: {multi_cat_rate:.3f}")
    
    if 'Agent_Contamination_Rate' in unified_base.columns:
        contamination_rate = unified_base['Agent_Contamination_Rate'].mean()
        print(f"Average agent contamination rate: {contamination_rate:.3f}")
    
    # 9. Create feature dictionary for documentation
    print("\n9. CREATING FEATURE DOCUMENTATION")
    print("-" * 50)
    
    feature_documentation = []
    
    # Original features
    for col in original_features:
        if col != 'variable5':
            feature_documentation.append({
                'Feature_Name': col,
                'Feature_Type': 'Original',
                'Data_Type': str(unified_base[col].dtype),
                'Description': 'Original column from input data',
                'Missing_Count': unified_base[col].isnull().sum(),
                'Unique_Values': unified_base[col].nunique()
            })
    
    # Engineered features with descriptions
    engineered_descriptions = {
        'Unique_L1_Categories': 'Number of unique L1 categories per conversation',
        'Unique_L2_Categories': 'Number of unique L2 categories per conversation',
        'Total_UUIDs': 'Total number of UUIDs per conversation',
        'Is_Multi_L1_Category': 'Binary flag for multi-L1 category conversations',
        'Is_Multi_L2_Category': 'Binary flag for multi-L2 category conversations',
        'Avg_Agent_Contamination_Score': 'Average agent contamination score per conversation',
        'Agent_Contamination_Rate': 'Rate of agent contamination patterns per conversation',
        'Conversation_Complexity_Score': 'Composite score measuring conversation complexity',
        'Precision_Risk_Score': 'Raw precision risk score based on multiple factors',
        'Precision_Risk_Score_Normalized': 'Normalized precision risk score (0-1)',
        'Customer_Engagement_Score': 'Score measuring customer engagement level',
        'Avg_Validation_Agreement': 'Average primary-secondary validation agreement',
        'Validation_Count': 'Number of validation records per conversation',
        'Has_Validation_Data': 'Binary flag for presence of validation data'
    }
    
    for col in engineered_features:
        description = engineered_descriptions.get(col, 'Engineered feature')
        
        # Determine feature category
        if col.startswith('L1_'):
            category = 'L1_Category_OneHot'
            description = f'One-hot encoded feature for L1 category: {col[3:]}'
        elif col.startswith('L2_'):
            category = 'L2_Category_OneHot'
            description = f'One-hot encoded feature for L2 category: {col[3:]}'
        elif 'Agent' in col and 'Contamination' in col:
            category = 'Agent_Contamination'
        elif any(pattern in col for pattern in ['Uncertainty', 'Hedging', 'Doubt', 'Politeness']):
            category = 'Qualifying_Language'
        elif 'Category' in col or 'Multi' in col:
            category = 'Category_Complexity'
        elif 'Validation' in col:
            category = 'Validation_Quality'
        elif 'Risk' in col or 'Score' in col:
            category = 'Risk_Scoring'
        elif 'Density' in col:
            category = 'Feature_Density'
        else:
            category = 'Other_Engineered'
        
        feature_documentation.append({
            'Feature_Name': col,
            'Feature_Type': 'Engineered',
            'Feature_Category': category,
            'Data_Type': str(unified_base[col].dtype),
            'Description': description,
            'Missing_Count': unified_base[col].isnull().sum(),
            'Unique_Values': unified_base[col].nunique()
        })
    
    feature_doc_df = pd.DataFrame(feature_documentation)
    
    print("Feature documentation created successfully")
    print(f"Documented {len(feature_doc_df)} features")
    
    # 10. Export unified dataframe and documentation
    print("\n10. EXPORTING UNIFIED DATAFRAME")
    print("-" * 50)
    
    # Export to Excel with multiple sheets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'Unified_Feature_Dataset_{timestamp}.xlsx'
    
    with pd.ExcelWriter(output_filename, engine='xlsxwriter') as writer:
        # Main unified dataset
        unified_base.to_excel(writer, sheet_name='Unified_Dataset', index=False)
        
        # Feature documentation
        feature_doc_df.to_excel(writer, sheet_name='Feature_Documentation', index=False)
        
        # Data quality summary
        if len(missing_df) > 0:
            missing_df.to_excel(writer, sheet_name='Data_Quality', index=False)
        
        # Sample data for reference
        unified_base.head(100).to_excel(writer, sheet_name='Sample_Data', index=False)
        
        # Feature statistics
        numeric_features = unified_base.select_dtypes(include=[np.number]).columns
        feature_stats = unified_base[numeric_features].describe().round(3)
        feature_stats.to_excel(writer, sheet_name='Feature_Statistics')
    
    print(f"Unified dataframe exported to: {output_filename}")
    print(f"Sheets included: Unified_Dataset, Feature_Documentation, Data_Quality, Sample_Data, Feature_Statistics")
    
    return unified_base, feature_doc_df
