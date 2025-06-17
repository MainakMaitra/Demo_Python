def analyze_category_confusion(df_main):
    """Identify which categories are often confused"""
    
    # For transcripts with secondary validation where there's disagreement
    confusion_data = df_main[
        df_main['Has_Secondary_Validation'] & 
        df_main['Secondary L1'].notna() &
        df_main['Secondary L2'].notna()
    ].copy()
    
    if len(confusion_data) == 0:
        print("No secondary validation data available for confusion analysis")
        return pd.DataFrame()
    
    # Create confusion matrix for L1 categories
    print("L1 Category Confusion Matrix:")
    l1_confusion = pd.crosstab(
        confusion_data['Primary L1'].fillna('None'), 
        confusion_data['Secondary L1'].fillna('None')
    )
    print(l1_confusion)
    
    # Create confusion matrix for L2 categories
    print("\nL2 Category Confusion Matrix:")
    l2_confusion = pd.crosstab(
        confusion_data['Primary L2'].fillna('None'), 
        confusion_data['Secondary L2'].fillna('None')
    )
    
    # Find most confused L1 pairs
    confused_l1_pairs = []
    for primary in l1_confusion.index:
        for secondary in l1_confusion.columns:
            if primary != secondary and l1_confusion.loc[primary, secondary] > 0:
                confused_l1_pairs.append({
                    'Primary_L1': primary,
                    'Secondary_L1': secondary,
                    'Count': l1_confusion.loc[primary, secondary],
                    'Type': 'L1_Confusion'
                })
    
    # Find most confused L2 pairs
    confused_l2_pairs = []
    for primary in l2_confusion.index:
        for secondary in l2_confusion.columns:
            if primary != secondary and l2_confusion.loc[primary, secondary] > 0:
                confused_l2_pairs.append({
                    'Primary_L2': primary,
                    'Secondary_L2': secondary,
                    'Count': l2_confusion.loc[primary, secondary],
                    'Type': 'L2_Confusion'
                })
    
    # Combine and sort
    all_confused_pairs = pd.DataFrame(confused_l1_pairs + confused_l2_pairs)
    
    if len(all_confused_pairs) > 0:
        all_confused_pairs = all_confused_pairs.sort_values('Count', ascending=False)
        
        print("\nMost Confused Category Pairs:")
        print(all_confused_pairs.head(10))
        
        # Analyze disagreement patterns
        disagreement_analysis = confusion_data[
            (confusion_data['Primary L1'] != confusion_data['Secondary L1']) |
            (confusion_data['Primary L2'] != confusion_data['Secondary L2'])
        ]
        
        print(f"\nDisagreement Statistics:")
        print(f"Total records with secondary validation: {len(confusion_data)}")
        print(f"Records with disagreement: {len(disagreement_analysis)} ({len(disagreement_analysis)/len(confusion_data)*100:.1f}%)")
        
        # Show example disagreements
        if len(disagreement_analysis) > 0:
            print("\nExample Disagreements:")
            sample_disagreements = disagreement_analysis[
                ['variable5', 'Primary L1', 'Primary L2', 'Secondary L1', 'Secondary L2', 'Customer Transcript']
            ].head(5)
            
            for idx, row in sample_disagreements.iterrows():
                print(f"\nVariable5: {row['variable5']}")
                print(f"Primary: {row['Primary L1']} - {row['Primary L2']}")
                print(f"Secondary: {row['Secondary L1']} - {row['Secondary L2']}")
                print(f"Transcript snippet: {row['Customer Transcript'][:100]}...")
    
    return all_confused_pairs

# Call the corrected function
confused_categories = analyze_category_confusion(df_main)

def analyze_context_windows(df_main):
    """Analyze text surrounding complaint keywords"""
    
    complaint_keywords = ['problem', 'issue', 'wrong', 'error', 'frustrated']
    context_window = 50  # characters before/after
    
    # Helper function to extract common patterns
    def extract_common_patterns(contexts):
        """Extract common words/phrases from context list"""
        if not contexts:
            return []
        
        # Combine all contexts
        all_text = ' '.join(contexts).lower()
        
        # Remove the keyword itself and common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'this', 'that'}
        
        # Extract words
        words = re.findall(r'\b\w+\b', all_text)
        word_freq = {}
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Also look for common bigrams
        bigrams = []
        for i in range(len(words)-1):
            if words[i] not in stop_words and words[i+1] not in stop_words:
                bigrams.append(f"{words[i]} {words[i+1]}")
        
        bigram_freq = {}
        for bigram in bigrams:
            bigram_freq[bigram] = bigram_freq.get(bigram, 0) + 1
        
        common_bigrams = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'common_words': [word for word, count in common_words],
            'common_phrases': [phrase for phrase, count in common_bigrams]
        }
    
    context_analysis_results = []
    
    for keyword in complaint_keywords:
        tp_contexts = []
        fp_contexts = []
        
        # Analyze each row
        for _, row in df_main.iterrows():
            text = str(row['Customer Transcript']).lower()
            
            if keyword in text:
                # Find all occurrences of the keyword
                positions = [m.start() for m in re.finditer(r'\b' + keyword + r'\b', text)]
                
                for pos in positions:
                    # Extract context
                    start = max(0, pos - context_window)
                    end = min(len(text), pos + len(keyword) + context_window)
                    context = text[start:end]
                    
                    # Store context based on classification
                    if row['Primary Marker'] == 'TP':
                        tp_contexts.append(context)
                    else:
                        fp_contexts.append(context)
        
        # Analyze patterns
        tp_patterns = extract_common_patterns(tp_contexts)
        fp_patterns = extract_common_patterns(fp_contexts)
        
        print(f"\nKeyword: '{keyword}'")
        print(f"Found in {len(tp_contexts)} TP contexts and {len(fp_contexts)} FP contexts")
        
        if tp_patterns:
            print(f"TP contexts often include words: {tp_patterns['common_words']}")
            print(f"TP contexts common phrases: {tp_patterns['common_phrases']}")
        
        if fp_patterns:
            print(f"FP contexts often include words: {fp_patterns['common_words']}")
            print(f"FP contexts common phrases: {fp_patterns['common_phrases']}")
        
        # Store results
        context_analysis_results.append({
            'keyword': keyword,
            'tp_count': len(tp_contexts),
            'fp_count': len(fp_contexts),
            'tp_patterns': tp_patterns,
            'fp_patterns': fp_patterns
        })
        
        # Show example contexts
        if len(tp_contexts) > 0 and len(fp_contexts) > 0:
            print("\nExample TP context:")
            print(f"...{tp_contexts[0]}...")
            print("\nExample FP context:")
            print(f"...{fp_contexts[0]}...")
    
    # Identify distinguishing patterns
    print("\n" + "="*50)
    print("DISTINGUISHING PATTERNS SUMMARY")
    print("="*50)
    
    for result in context_analysis_results:
        keyword = result['keyword']
        tp_words = set(result['tp_patterns'].get('common_words', []))
        fp_words = set(result['fp_patterns'].get('common_words', []))
        
        # Words unique to TP contexts
        tp_unique = tp_words - fp_words
        # Words unique to FP contexts
        fp_unique = fp_words - tp_words
        
        if tp_unique or fp_unique:
            print(f"\n'{keyword}' distinguishing patterns:")
            if tp_unique:
                print(f"  TP indicators: {list(tp_unique)}")
            if fp_unique:
                print(f"  FP indicators: {list(fp_unique)}")
    
    return context_analysis_results

# Call the function
context_results = analyze_context_windows(df_main)

