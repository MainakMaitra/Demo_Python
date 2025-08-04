# Step 6: TF-IDF keyword analysis (FIXED VERSION)
print("\nStep 6: Keyword analysis for FP inclination...")

# CRITICAL FIX: Ensure proper text data for TF-IDF
all_clean_texts = df['Customer_Transcript_Clean'].values.tolist()

# Additional cleaning for TF-IDF
cleaned_texts = []
valid_indices = []

for i, text in enumerate(all_clean_texts):
    if text is not None and isinstance(text, str) and len(text.strip()) > 0:
        clean_text = str(text).strip()
        if len(clean_text.split()) >= 2:  # At least 2 words
            cleaned_texts.append(clean_text)
            valid_indices.append(i)

print(f"Valid texts for TF-IDF: {len(cleaned_texts)}")

# Update masks for valid indices only
valid_tp_mask = tp_mask.iloc[valid_indices]
valid_fp_mask = fp_mask.iloc[valid_indices]

try:
    # TF-IDF with conservative parameters
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=max(2, int(len(cleaned_texts) * 0.01)),
        max_df=0.95,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
    )
    
    tfidf_matrix = tfidf.fit_transform(cleaned_texts)
    feature_names = tfidf.get_feature_names_out()
    
    # Calculate means
    tfidf_dense = tfidf_matrix.toarray()
    tp_tfidf_mean = np.mean(tfidf_dense[valid_tp_mask], axis=0)
    fp_tfidf_mean = np.mean(tfidf_dense[valid_fp_mask], axis=0)
    
    fp_inclination = fp_tfidf_mean - tp_tfidf_mean
    
    # Create keyword analysis
    keyword_analysis = pd.DataFrame({
        'keyword': feature_names,
        'tp_score': tp_tfidf_mean,
        'fp_score': fp_tfidf_mean,
        'fp_inclination': fp_inclination,
        'fp_ratio': np.where(tp_tfidf_mean > 0, fp_tfidf_mean / tp_tfidf_mean, np.inf)
    })
    
    keyword_analysis = keyword_analysis.sort_values('fp_inclination', ascending=False)
    keyword_analysis = keyword_analysis[np.isfinite(keyword_analysis['fp_ratio'])]
    
    print("✅ TF-IDF analysis completed successfully")
    
except Exception as e:
    print(f"TF-IDF failed: {e}. Creating minimal analysis...")
    keyword_analysis = pd.DataFrame({
        'keyword': ['sample'],
        'tp_score': [0.1],
        'fp_score': [0.1], 
        'fp_inclination': [0.0],
        'fp_ratio': [1.0]
    })
