## 1. **Deeper Query Rule Analysis**

### Query Evolution Timeline

def analyze_query_rule_evolution(df_main, df_rules):
    """Analyze how query modifications correlate with precision changes"""
    
    # Track when each query was last modified
    query_modifications = df_rules.groupby(['Event', 'Query']).agg({
        'begin_date': ['min', 'max', 'count']
    }).reset_index()
    
    # Correlate query changes with precision drops
    for query in query_modifications.itertuples():
        # Find precision before and after query modification
        before_data = df_main[
            (df_main['Prosodica L1'] == query.Event) & 
            (df_main['Prosodica L2'] == query.Query) &
            (df_main['Date'] < query.begin_date_max)
        ]
        after_data = df_main[
            (df_main['Prosodica L1'] == query.Event) & 
            (df_main['Prosodica L2'] == query.Query) &
            (df_main['Date'] >= query.begin_date_max)
        ]
        
        if len(before_data) > 10 and len(after_data) > 10:
            before_precision = before_data['Is_TP'].mean()
            after_precision = after_data['Is_TP'].mean()
            precision_change = after_precision - before_precision
            
            print(f"{query.Query}: {precision_change:+.3f} after modification")


### Query Component Effectiveness

def analyze_query_components(df_main, df_rules):
    """Break down which parts of queries are causing FPs"""
    
    # Extract individual query components
    for _, rule in df_rules.iterrows():
        query_text = rule['Query Text']
        
        # Parse query into components
        components = re.findall(r'"([^"]+)"|\b(\w+)\b', query_text)
        
        # Test each component's correlation with FPs
        for component in components:
            if component[0]:  # Quoted phrase
                phrase = component[0]
                # Check FP rate when this phrase appears
                contains_phrase = df_main[
                    df_main['Full_Transcript'].str.contains(phrase, case=False, na=False)
                ]
                fp_rate = contains_phrase['Is_FP'].mean()
                print(f"Phrase '{phrase}': FP rate = {fp_rate:.3f}")


## 2. **Context Window Analysis**

### Surrounding Context Extraction

def analyze_context_windows(df_main):
    """Analyze text surrounding complaint keywords"""
    
    complaint_keywords = ['problem', 'issue', 'wrong', 'error', 'frustrated']
    context_window = 50  # characters before/after
    
    context_patterns = []
    
    for keyword in complaint_keywords:
        tp_contexts = []
        fp_contexts = []
        
        for _, row in df_main.iterrows():
            text = row['Customer Transcript'].lower()
            if keyword in text:
                # Extract context around keyword
                positions = [m.start() for m in re.finditer(keyword, text)]
                
                for pos in positions:
                    start = max(0, pos - context_window)
                    end = min(len(text), pos + len(keyword) + context_window)
                    context = text[start:end]
                    
                    if row['Primary Marker'] == 'TP':
                        tp_contexts.append(context)
                    else:
                        fp_contexts.append(context)
        
        # Analyze common patterns in contexts
        tp_patterns = extract_common_patterns(tp_contexts)
        fp_patterns = extract_common_patterns(fp_contexts)
        
        print(f"\nKeyword: {keyword}")
        print(f"TP contexts often include: {tp_patterns}")
        print(f"FP contexts often include: {fp_patterns}")


## 3. **Conversation Flow Analysis**

### Turn-Taking Patterns

def analyze_conversation_dynamics(df_main):
    """Analyze conversation flow patterns"""
    
    # Split conversations into turns
    for _, row in df_main.iterrows():
        customer_turns = len(re.findall(r'[.!?]+', row['Customer Transcript']))
        agent_turns = len(re.findall(r'[.!?]+', row['Agent Transcript']))
        
        # Calculate conversation metrics
        turn_ratio = customer_turns / (agent_turns + 1)
        avg_customer_turn_length = row['Customer_Word_Count'] / (customer_turns + 1)
        avg_agent_turn_length = row['Agent_Word_Count'] / (agent_turns + 1)
        
        # Add to dataframe
        df_main.loc[_, 'Turn_Ratio'] = turn_ratio
        df_main.loc[_, 'Avg_Customer_Turn_Length'] = avg_customer_turn_length
        df_main.loc[_, 'Avg_Agent_Turn_Length'] = avg_agent_turn_length
    
    # Compare TP vs FP conversation dynamics
    tp_dynamics = df_main[df_main['Primary Marker'] == 'TP'][
        ['Turn_Ratio', 'Avg_Customer_Turn_Length', 'Avg_Agent_Turn_Length']
    ].mean()
    
    fp_dynamics = df_main[df_main['Primary Marker'] == 'FP'][
        ['Turn_Ratio', 'Avg_Customer_Turn_Length', 'Avg_Agent_Turn_Length']
    ].mean()
    
    print("Conversation Dynamics Comparison:")
    print(f"TP conversations: {tp_dynamics}")
    print(f"FP conversations: {fp_dynamics}")


## 4. **Semantic Similarity Analysis**

### Category Confusion Matrix

def analyze_category_confusion(df_main):
    """Identify which categories are often confused"""
    
    # For transcripts with secondary validation
    confusion_data = df_main[
        df_main['Has_Secondary_Validation'] & 
        (df_main['Primary L1'] != df_main['Secondary L1'])
    ]
    
    # Build confusion matrix
    confusion_matrix = pd.crosstab(
        confusion_data['Primary L1'], 
        confusion_data['Secondary L1']
    )
    
    print("Category Confusion Matrix:")
    print(confusion_matrix)
    
    # Find most confused pairs
    confused_pairs = []
    for primary in confusion_matrix.index:
        for secondary in confusion_matrix.columns:
            if confusion_matrix.loc[primary, secondary] > 5:
                confused_pairs.append({
                    'Primary': primary,
                    'Secondary': secondary,
                    'Count': confusion_matrix.loc[primary, secondary]
                })
    
    return pd.DataFrame(confused_pairs).sort_values('Count', ascending=False)


## 5. **Call Metadata Analysis**

### Call Duration and Precision

def analyze_call_metadata(df_main):
    """Analyze if call characteristics affect precision"""
    
    # Estimate call duration from transcript length
    df_main['Estimated_Duration'] = df_main['Transcript_Length'] / 150  # chars per second
    
    # Bin calls by duration
    df_main['Duration_Bin'] = pd.qcut(
        df_main['Estimated_Duration'], 
        q=5, 
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    
    # Analyze precision by duration
    duration_precision = df_main.groupby('Duration_Bin').agg({
        'Is_TP': 'mean',
        'Is_FP': 'mean',
        'Customer_Negation_Count': 'mean',
        'Customer_Agent_Ratio': 'mean'
    })
    
    print("Precision by Call Duration:")
    print(duration_precision)


## 6. **Error Propagation Analysis**

### Track How Errors Cascade

def analyze_error_propagation(df_main):
    """Analyze how one misclassification leads to others"""
    
    # Group by conversation (variable5)
    conversation_errors = df_main.groupby('variable5').agg({
        'Is_FP': ['sum', 'count'],
        'Prosodica L1': 'nunique',
        'UUID': 'count'
    })
    
    conversation_errors.columns = ['FP_Count', 'Total_Classifications', 'Unique_Categories', 'UUID_Count']
    conversation_errors['FP_Rate'] = conversation_errors['FP_Count'] / conversation_errors['Total_Classifications']
    
    # Identify conversations with cascading errors
    cascading_errors = conversation_errors[
        (conversation_errors['FP_Count'] > 1) & 
        (conversation_errors['FP_Rate'] > 0.5)
    ]
    
    print(f"Conversations with cascading errors: {len(cascading_errors)}")
    
    # Analyze patterns in cascading errors
    for conv_id in cascading_errors.index[:5]:  # Top 5
        conv_data = df_main[df_main['variable5'] == conv_id]
        print(f"\nConversation {conv_id}:")
        print(f"Categories flagged: {conv_data['Prosodica L2'].unique()}")
        print(f"FP rate: {conv_data['Is_FP'].mean():.3f}")


## 7. **Real-Time Performance Monitoring**

### Daily Precision Tracking

def create_daily_monitoring_dashboard(df_main):
    """Create daily precision monitoring metrics"""
    
    daily_metrics = df_main.groupby('Date').agg({
        'Is_TP': ['sum', 'count'],
        'Is_FP': 'sum',
        'variable5': 'nunique',
        'Customer_Negation_Count': 'mean',
        'Customer_Agent_Ratio': 'mean'
    })
    
    daily_metrics.columns = ['TP_Count', 'Total_Flagged', 'FP_Count', 
                           'Unique_Calls', 'Avg_Negations', 'Avg_Ratio']
    daily_metrics['Precision'] = daily_metrics['TP_Count'] / daily_metrics['Total_Flagged']
    daily_metrics['Precision_7day_MA'] = daily_metrics['Precision'].rolling(7).mean()
    
    # Detect anomalies
    daily_metrics['Z_Score'] = (
        (daily_metrics['Precision'] - daily_metrics['Precision_7day_MA']) / 
        daily_metrics['Precision'].rolling(7).std()
    )
    
    anomaly_days = daily_metrics[abs(daily_metrics['Z_Score']) > 2]
    
    print(f"Days with anomalous precision: {len(anomaly_days)}")
    return daily_metrics


## 8. **Machine Learning Insights**

### Feature Importance for FP Prediction

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def analyze_fp_drivers_ml(df_main):
    """Use ML to identify key FP drivers"""
    
    # Prepare features
    feature_cols = [
        'Transcript_Length', 'Customer_Word_Count', 'Agent_Word_Count',
        'Customer_Agent_Ratio', 'Customer_Question_Count', 'Customer_Exclamation_Count',
        'Customer_Caps_Ratio', 'Customer_Negation_Count', 'Agent_Negation_Count',
        'Customer_Qualifying_Count', 'WeekOfMonth', 'Is_Holiday_Season'
    ]
    
    X = df_main[feature_cols].fillna(0)
    y = df_main['Is_FP']
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top FP Predictors:")
    print(importance.head(10))
    
    return rf, importance
