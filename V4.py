def analyze_negation_contexts_debug(df, negation_candidates, max_words=10, max_rows=100):
    """
    Debug-safe version: Analyze negation contexts with progress logs and limited scope
    """
    print("\n2. ANALYZING NEGATION CONTEXTS [DEBUG MODE]")
    print("-" * 50)
    
    context_analysis = {}
    negation_words_subset = list(negation_candidates.keys())[:max_words]
    df_subset = df.head(max_rows)

    print(f"Processing up to {len(negation_words_subset)} negation candidates over {len(df_subset)} transcripts...\n")

    for idx, negation_word in enumerate(negation_words_subset):
        print(f"[{idx+1}/{len(negation_words_subset)}] Analyzing word: '{negation_word}'")
        
        context_data = {
            'total_occurrences': 0,
            'tp_occurrences': 0,
            'fp_occurrences': 0,
            'customer_occurrences': 0,
            'agent_occurrences': 0,
            'contexts_before': Counter(),
            'contexts_after': Counter(),
            'complaint_associations': Counter(),
            'information_associations': Counter()
        }

        for _, row in df_subset.iterrows():
            is_tp = row['Primary Marker'] == 'TP'
            
            for transcript_type, speaker in [('Customer Transcript', 'customer'), ('Agent Transcript', 'agent')]:
                text = str(row[transcript_type]).lower()
                words = text.split()
                
                for i, word in enumerate(words):
                    if negation_word in word:
                        context_data['total_occurrences'] += 1
                        if is_tp:
                            context_data['tp_occurrences'] += 1
                        else:
                            context_data['fp_occurrences'] += 1
                        if speaker == 'customer':
                            context_data['customer_occurrences'] += 1
                        else:
                            context_data['agent_occurrences'] += 1
                        
                        before_context = ' '.join(words[max(0, i-3):i])
                        after_context = ' '.join(words[i+1:min(len(words), i+4)])
                        full_context = ' '.join(words[max(0, i-5):min(len(words), i+6)])

                        context_data['contexts_before'][before_context] += 1
                        context_data['contexts_after'][after_context] += 1

                        if is_complaint_context(full_context):
                            context_data['complaint_associations'][full_context] += 1
                        else:
                            context_data['information_associations'][full_context] += 1
        
        # Log summary per word
        print(f"  Total Occurrences: {context_data['total_occurrences']}")
        print(f"    TP: {context_data['tp_occurrences']}, FP: {context_data['fp_occurrences']}")
        print(f"    Customer: {context_data['customer_occurrences']}, Agent: {context_data['agent_occurrences']}")
        print(f"    Complaint Contexts: {len(context_data['complaint_associations'])}")
        print(f"    Info Contexts: {len(context_data['information_associations'])}")
        print("-" * 50)
        
        context_analysis[negation_word] = context_data
    
    print("\nDebug-safe context analysis completed.")
    return context_analysis
