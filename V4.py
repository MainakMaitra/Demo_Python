# Integration Guide: Replace TF-IDF Section with Advanced Keyword Analysis
# =========================================================================

def integrate_improved_keyword_analysis():
    """
    Complete integration guide for replacing Step 6 in your main function
    """
    
    integration_steps = '''
    INTEGRATION STEPS FOR IMPROVED KEYWORD ANALYSIS:
    ===============================================
    
    STEP 1: ADD the improved functions before your main function
    STEP 2: REPLACE Step 6 section in your main function
    STEP 3: UPDATE the export section to handle new results
    '''
    
    return integration_steps

# STEP 1: Add these functions before your main function
def create_domain_specific_stopwords():
    """
    Create comprehensive stopwords including conversational fillers
    (ADD THIS FUNCTION BEFORE bert_contrast_analysis)
    """
    
    # Standard English stopwords
    standard_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
        'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 
        'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
        'under', 'again', 'further', 'then', 'once'
    }
    
    # Customer service conversation fillers
    conversation_fillers = {
        'okay', 'ok', 'yes', 'yeah', 'yep', 'no', 'hmm', 'um', 'uh', 'ah',
        'well', 'so', 'like', 'you know', 'right', 'sure', 'exactly',
        'absolutely', 'definitely', 'certainly', 'of course', 'obviously'
    }
    
    # Politeness markers (context-specific stopwords)
    politeness_markers = {
        'please', 'thank', 'thanks', 'thank you', 'welcome', 'sorry',
        'excuse me', 'pardon', 'appreciate', 'grateful', 'kindly'
    }
    
    # Customer service specific terms
    service_stopwords = {
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'have a nice day', 'take care', 'bye', 'goodbye', 'see you',
        'help', 'assist', 'support', 'service', 'customer', 'sir', 'madam',
        'mam', 'mfm', 'call', 'phone', 'line', 'hold', 'wait'
    }
    
    # Combine all stopword categories
    comprehensive_stopwords = (standard_stopwords | 
                             conversation_fillers | 
                             politeness_markers | 
                             service_stopwords)
    
    return comprehensive_stopwords

def advanced_keyword_analysis(df, tp_mask, fp_mask):
    """
    Advanced keyword analysis focusing on meaningful content words
    (ADD THIS FUNCTION BEFORE bert_contrast_analysis)
    """
    
    print("\nStep 6: Advanced Keyword Analysis (Filtering Context-Specific Stopwords)...")
    
    # Get domain-specific stopwords
    domain_stopwords = create_domain_specific_stopwords()
    
    # Get clean texts
    all_clean_texts = df['Customer_Transcript_Clean'].values.tolist()
    
    # Additional cleaning for TF-IDF
    cleaned_texts = []
    valid_indices = []

    for i, text in enumerate(all_clean_texts):
        if text is not None and isinstance(text, str) and len(text.strip()) > 0:
            clean_text_str = str(text).strip()
            if len(clean_text_str.split()) >= 3:  # At least 3 words for meaningful analysis
                cleaned_texts.append(clean_text_str)
                valid_indices.append(i)

    print(f"Valid texts for advanced TF-IDF: {len(cleaned_texts)} out of {len(all_clean_texts)}")

    if len(cleaned_texts) < 20:
        print("Warning: Very few texts for keyword analysis. Results may not be reliable.")
        return create_fallback_keyword_analysis()

    # Update masks for valid indices only
    valid_tp_mask = tp_mask.iloc[valid_indices]
    valid_fp_mask = fp_mask.iloc[valid_indices]

    try:
        # Advanced TF-IDF with domain knowledge
        advanced_tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Include trigrams for context
            min_df=max(3, int(len(cleaned_texts) * 0.02)),  # Higher minimum frequency
            max_df=0.8,  # Remove very common terms
            stop_words=list(domain_stopwords),  # Use our comprehensive stopwords
            token_pattern=r'\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b',  # At least 3 characters
            sublinear_tf=True,  # Better handling of term frequencies
            norm='l2'  # L2 normalization
        )
        
        tfidf_matrix = advanced_tfidf.fit_transform(cleaned_texts)
        feature_names = advanced_tfidf.get_feature_names_out()
        
        print(f"Features after advanced filtering: {len(feature_names)}")
        
        # Calculate class-specific means
        tfidf_dense = tfidf_matrix.toarray()
        tp_mean = np.mean(tfidf_dense[valid_tp_mask], axis=0)
        fp_mean = np.mean(tfidf_dense[valid_fp_mask], axis=0)
        
        # Calculate discriminative power
        fp_inclination = fp_mean - tp_mean
        
        # Additional metrics for better analysis
        tp_std = np.std(tfidf_dense[valid_tp_mask], axis=0)
        fp_std = np.std(tfidf_dense[valid_fp_mask], axis=0)
        
        # Statistical significance (simple t-test approximation)
        pooled_std = np.sqrt((tp_std**2 + fp_std**2) / 2)
        t_statistic = fp_inclination / (pooled_std + 1e-8)  # Avoid division by zero
        
        # Create enhanced keyword analysis
        enhanced_analysis = pd.DataFrame({
            'keyword': feature_names,
            'tp_score': tp_mean,
            'fp_score': fp_mean,
            'fp_inclination': fp_inclination,
            'statistical_significance': np.abs(t_statistic),
            'tp_std': tp_std,
            'fp_std': fp_std,
            'discriminative_power': np.abs(fp_inclination) * np.abs(t_statistic)
        })
        
        # Sort by discriminative power (combines effect size and significance)
        enhanced_analysis = enhanced_analysis.sort_values(
            'discriminative_power', ascending=False
        )
        
        # Filter for meaningful differences
        meaningful_keywords = enhanced_analysis[
            (enhanced_analysis['statistical_significance'] > 1.0) &  # Some significance
            (np.abs(enhanced_analysis['fp_inclination']) > 0.001)     # Meaningful difference
        ]
        
        print(f"Meaningful discriminative keywords found: {len(meaningful_keywords)}")
        
        # Print top results
        print("\nTop 10 MEANINGFUL keywords most inclined toward False Positives:")
        for i, (_, row) in enumerate(meaningful_keywords.head(10).iterrows(), 1):
            print(f"   {i:2d}. '{row['keyword']}' | FP: {row['fp_score']:.4f} | TP: {row['tp_score']:.4f}")
            print(f"       Diff: {row['fp_inclination']:+.4f} | Significance: {row['statistical_significance']:.2f}")
        
        # Return both enhanced and meaningful analyses
        return {
            'enhanced_analysis': enhanced_analysis,
            'meaningful_keywords': meaningful_keywords,
            'stopwords_used': domain_stopwords,
            'analysis_type': 'advanced'
        }
        
    except Exception as e:
        print(f"Advanced keyword analysis failed: {e}. Using fallback...")
        return create_fallback_keyword_analysis()

def create_fallback_keyword_analysis():
    """
    Create fallback analysis if advanced method fails
    """
    
    print("Creating fallback keyword analysis...")
    
    fallback_analysis = pd.DataFrame({
        'keyword': ['analysis_failed'],
        'tp_score': [0.1],
        'fp_score': [0.1], 
        'fp_inclination': [0.0],
        'statistical_significance': [0.0],
        'discriminative_power': [0.0]
    })
    
    return {
        'enhanced_analysis': fallback_analysis,
        'meaningful_keywords': fallback_analysis,
        'stopwords_used': set(),
        'analysis_type': 'fallback'
    }



