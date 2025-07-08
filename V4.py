def extract_real_examples_agent_contamination_and_qualifying(df, num_examples=10):
    """
    Extract real examples from the data for Agent Contamination and Qualifying Language insights
    """
    
    print("="*80)
    print("REAL EXAMPLES EXTRACTION")
    print("="*80)
    
    # 1. Agent Contamination Examples
    print("1. AGENT CONTAMINATION EXAMPLES")
    print("-" * 50)
    
    # Define agent explanation patterns
    agent_explanation_patterns = [
        r'\b(let me explain|i\'ll explain|what this means|this means that)\b',
        r'\b(for example|for instance|let\'s say|suppose)\b', 
        r'\b(if you|what if|in case|should you|were to)\b',
        r'\b(to clarify|what i mean|in other words|basically)\b',
        r'\b(you need to|you should|you can|you have to)\b'
    ]
    
    # Find FPs with high agent contamination
    fp_data = df[df['Primary Marker'] == 'FP'].copy()
    
    # Check for agent contamination
    fp_data['Agent_Contamination_Score'] = 0
    for pattern in agent_explanation_patterns:
        fp_data['Agent_Contamination_Score'] += fp_data['Agent Transcript'].str.lower().str.contains(pattern, regex=True, na=False).astype(int)
    
    # Get high contamination examples
    high_contamination = fp_data[fp_data['Agent_Contamination_Score'] >= 2].head(num_examples)
    
    print("HIGH AGENT CONTAMINATION EXAMPLES (False Positives):")
    print("="*60)
    
    agent_contamination_examples = []
    
    for idx, row in high_contamination.iterrows():
        example = {
            'UUID': row['UUID'],
            'Category_L1': row['Prosodica L1'],
            'Category_L2': row['Prosodica L2'],
            'Customer_Transcript': row['Customer Transcript'][:500] + "..." if len(row['Customer Transcript']) > 500 else row['Customer Transcript'],
            'Agent_Transcript': row['Agent Transcript'][:500] + "..." if len(row['Agent Transcript']) > 500 else row['Agent Transcript'],
            'Contamination_Score': row['Agent_Contamination_Score'],
            'Customer_Agent_Ratio': row['Customer_Agent_Ratio'] if 'Customer_Agent_Ratio' in row else 'N/A',
            'Analysis': ''
        }
        
        # Analyze what type of agent contamination
        contamination_types = []
        agent_text = str(row['Agent Transcript']).lower()
        
        if re.search(r'\b(let me explain|i\'ll explain|what this means)\b', agent_text):
            contamination_types.append("Direct Explanation")
        if re.search(r'\b(for example|for instance|let\'s say)\b', agent_text):
            contamination_types.append("Examples Given")
        if re.search(r'\b(if you|what if|in case)\b', agent_text):
            contamination_types.append("Hypothetical Scenarios")
        if re.search(r'\b(to clarify|what i mean|basically)\b', agent_text):
            contamination_types.append("Clarifications")
        if re.search(r'\b(you need to|you should|you can)\b', agent_text):
            contamination_types.append("Instructions")
        
        example['Analysis'] = f"Agent contamination types: {', '.join(contamination_types)}"
        agent_contamination_examples.append(example)
        
        print(f"\nExample {len(agent_contamination_examples)}:")
        print(f"UUID: {example['UUID']}")
        print(f"Category: {example['Category_L1']} -> {example['Category_L2']}")
        print(f"Contamination Score: {example['Contamination_Score']}")
        print(f"Customer-Agent Ratio: {example['Customer_Agent_Ratio']}")
        print(f"Analysis: {example['Analysis']}")
        print(f"Customer: {example['Customer_Transcript']}")
        print(f"Agent: {example['Agent_Transcript']}")
        print("-" * 60)
    
    # 2. Qualifying Language Examples
    print("\n2. QUALIFYING LANGUAGE EXAMPLES")
    print("-" * 50)
    
    # Define qualifying language patterns
    qualifying_patterns = {
        'Uncertainty': r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely|i think|i believe|i guess)\b',
        'Hedging': r'\b(sort of|kind of|more or less|somewhat|relatively|fairly|quite|rather)\b',
        'Doubt': r'\b(not sure|uncertain|unclear|confused|don\'t know|no idea)\b',
        'Politeness': r'\b(please|thank you|thanks|appreciate|grateful|excuse me|pardon|sorry)\b',
        'Questions': r'\?'
    }
    
    # Get TPs with high qualifying language (real complaints with polite/uncertain language)
    tp_data = df[df['Primary Marker'] == 'TP'].copy()
    
    tp_data['Qualifying_Score'] = 0
    tp_data['Qualifying_Types'] = ''
    
    for pattern_name, pattern in qualifying_patterns.items():
        matches = tp_data['Customer Transcript'].str.lower().str.count(pattern)
        tp_data['Qualifying_Score'] += matches
        
        # Track which types are present
        has_pattern = matches > 0
        tp_data.loc[has_pattern, 'Qualifying_Types'] += f"{pattern_name}, "
    
    # Get high qualifying examples
    high_qualifying = tp_data[tp_data['Qualifying_Score'] >= 5].head(num_examples)
    
    print("HIGH QUALIFYING LANGUAGE EXAMPLES (True Positives - Real Complaints):")
    print("="*60)
    
    qualifying_examples = []
    
    for idx, row in high_qualifying.iterrows():
        example = {
            'UUID': row['UUID'],
            'Category_L1': row['Prosodica L1'],
            'Category_L2': row['Prosodica L2'],
            'Customer_Transcript': row['Customer Transcript'][:500] + "..." if len(row['Customer Transcript']) > 500 else row['Customer Transcript'],
            'Qualifying_Score': row['Qualifying_Score'],
            'Qualifying_Types': row['Qualifying_Types'].rstrip(', '),
            'Transcript_Length': len(row['Customer Transcript']),
            'Analysis': ''
        }
        
        # Analyze the qualifying language
        customer_text = str(row['Customer Transcript']).lower()
        question_count = customer_text.count('?')
        polite_words = len(re.findall(r'\b(please|thank|sorry|appreciate)\b', customer_text))
        uncertain_words = len(re.findall(r'\b(maybe|might|think|believe|not sure)\b', customer_text))
        
        example['Analysis'] = f"Questions: {question_count}, Polite words: {polite_words}, Uncertain words: {uncertain_words}"
        qualifying_examples.append(example)
        
        print(f"\nExample {len(qualifying_examples)}:")
        print(f"UUID: {example['UUID']}")
        print(f"Category: {example['Category_L1']} -> {example['Category_L2']}")
        print(f"Qualifying Score: {example['Qualifying_Score']}")
        print(f"Types: {example['Qualifying_Types']}")
        print(f"Length: {example['Transcript_Length']} chars")
        print(f"Analysis: {example['Analysis']}")
        print(f"Customer: {example['Customer_Transcript']}")
        print("-" * 60)
    
    # 3. Contrasting Examples - FPs that lack qualifying language
    print("\n3. CONTRASTING EXAMPLES - FPs WITHOUT QUALIFYING LANGUAGE")
    print("-" * 50)
    
    # Find FPs with very low qualifying language scores
    fp_data['Qualifying_Score'] = 0
    for pattern in qualifying_patterns.values():
        fp_data['Qualifying_Score'] += fp_data['Customer Transcript'].str.lower().str.count(pattern)
    
    low_qualifying_fps = fp_data[fp_data['Qualifying_Score'] <= 2].head(5)
    
    print("LOW QUALIFYING LANGUAGE EXAMPLES (False Positives - Should NOT be complaints):")
    print("="*60)
    
    contrasting_examples = []
    
    for idx, row in low_qualifying_fps.iterrows():
        example = {
            'UUID': row['UUID'],
            'Category_L1': row['Prosodica L1'],
            'Category_L2': row['Prosodica L2'],
            'Customer_Transcript': row['Customer Transcript'][:400] + "..." if len(row['Customer Transcript']) > 400 else row['Customer Transcript'],
            'Qualifying_Score': row['Qualifying_Score'],
            'Transcript_Length': len(row['Customer Transcript']),
            'Why_Misclassified': ''
        }
        
        # Analyze why it was misclassified
        customer_text = str(row['Customer Transcript']).lower()
        
        reasons = []
        if 'not' in customer_text or 'no' in customer_text or 'never' in customer_text:
            reasons.append("Contains negations")
        if any(word in customer_text for word in ['problem', 'issue', 'wrong', 'error']):
            reasons.append("Contains complaint keywords")
        if len(customer_text) < 200:
            reasons.append("Very short transcript")
        if customer_text.count('?') == 0:
            reasons.append("No questions asked")
        
        example['Why_Misclassified'] = "; ".join(reasons) if reasons else "Unclear classification reason"
        contrasting_examples.append(example)
        
        print(f"\nContrasting Example {len(contrasting_examples)}:")
        print(f"UUID: {example['UUID']}")
        print(f"Category: {example['Category_L1']} -> {example['Category_L2']}")
        print(f"Qualifying Score: {example['Qualifying_Score']} (Very Low)")
        print(f"Length: {example['Transcript_Length']} chars")
        print(f"Why Misclassified: {example['Why_Misclassified']}")
        print(f"Customer: {example['Customer_Transcript']}")
        print("-" * 60)
    
    # 4. Summary Statistics
    print("\n4. SUMMARY STATISTICS")
    print("-" * 50)
    
    summary_stats = {
        'Agent_Contamination_Examples': len(agent_contamination_examples),
        'Avg_Contamination_Score': np.mean([ex['Contamination_Score'] for ex in agent_contamination_examples]),
        'Qualifying_Language_Examples': len(qualifying_examples),
        'Avg_Qualifying_Score': np.mean([ex['Qualifying_Score'] for ex in qualifying_examples]),
        'Contrasting_Examples': len(contrasting_examples),
        'Avg_Contrasting_Score': np.mean([ex['Qualifying_Score'] for ex in contrasting_examples])
    }
    
    print("Summary Statistics:")
    for key, value in summary_stats.items():
        print(f"{key}: {value:.2f}")
    
    # 5. Export Examples to DataFrames for further analysis
    agent_contamination_df = pd.DataFrame(agent_contamination_examples)
    qualifying_language_df = pd.DataFrame(qualifying_examples)
    contrasting_df = pd.DataFrame(contrasting_examples)
    
    print(f"\nExported {len(agent_contamination_df)} agent contamination examples")
    print(f"Exported {len(qualifying_language_df)} qualifying language examples")
    print(f"Exported {len(contrasting_df)} contrasting examples")
    
    return agent_contamination_df, qualifying_language_df, contrasting_df, summary_stats
