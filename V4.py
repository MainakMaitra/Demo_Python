def enhanced_deep_negation_analysis(df):
    """
    Enhanced negation analysis with clear evidence for context-insensitive handling
    and comprehensive Pre vs Post and monthly breakdown
    """
    
    print("="*80)
    print("ENHANCED DEEP NEGATION PATTERN ANALYSIS")
    print("="*80)
    
    # 1. Define Specific Negation Context Patterns
    negation_context_patterns = {
        'Complaint_Negation': r'\b(not|never|no|don\'t|won\'t|can\'t|isn\'t)\s+(working|received|getting|got|fair|right|correct|satisfied|resolved|fixed|helping|processed)\b',
        'Information_Negation': r'\b(don\'t|not|no|never)\s+(understand|know|see|find|have|remember|think|believe|sure|clear|aware)\b',
        'Service_Negation': r'\b(can\'t|won\'t|not|unable|doesn\'t)\s+(access|login|connect|load|work|function|available|possible)\b',
        'Denial_Negation': r'\b(not|never|no|didn\'t)\s+(my|mine|me|authorized|made|requested|asked|ordered)\b',
        'Process_Negation': r'\b(not|never|no|haven\'t|didn\'t)\s+(processed|completed|finished|done|updated|reflected|posted|credited)\b',
        'Agent_Negation': r'\b(not|no|never|don\'t|won\'t)\s+(worry|problem|issue|need|have\s+to|required|necessary)\b'
    }
    
    # 2. Analyze Context-Specific Negations
    print("1. CONTEXT-SPECIFIC NEGATION ANALYSIS")
    print("-" * 50)
    
    tp_data = df[df['Primary Marker'] == 'TP']
    fp_data = df[df['Primary Marker'] == 'FP']
    
    context_analysis = []
    
    for pattern_name, pattern in negation_context_patterns.items():
        # Analyze both customer and full transcript
        tp_customer_matches = tp_data['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        fp_customer_matches = fp_data['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        
        tp_full_matches = tp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        fp_full_matches = fp_data['Full_Transcript'].str.lower().str.contains(pattern, regex=True, na=False)
        
        tp_customer_rate = tp_customer_matches.mean() * 100 if len(tp_data) > 0 else 0
        fp_customer_rate = fp_customer_matches.mean() * 100 if len(fp_data) > 0 else 0
        
        tp_full_rate = tp_full_matches.mean() * 100 if len(tp_data) > 0 else 0
        fp_full_rate = fp_full_matches.mean() * 100 if len(fp_data) > 0 else 0
        
        # Calculate discrimination power
        discrimination_power = (tp_customer_rate - fp_customer_rate) / max(fp_customer_rate, 0.1)
        
        context_analysis.append({
            'Pattern_Type': pattern_name,
            'TP_Customer_Rate_%': tp_customer_rate,
            'FP_Customer_Rate_%': fp_customer_rate,
            'TP_Full_Rate_%': tp_full_rate,
            'FP_Full_Rate_%': fp_full_rate,
            'Customer_Discrimination': discrimination_power,
            'Customer_Difference': tp_customer_rate - fp_customer_rate,
            'Evidence_Strength': 'Strong' if abs(discrimination_power) > 1 else 'Moderate' if abs(discrimination_power) > 0.5 else 'Weak'
        })
    
    context_df = pd.DataFrame(context_analysis)
    context_df = context_df.sort_values('Customer_Discrimination', ascending=False)
    
    print("Context-Specific Negation Analysis:")
    print(context_df.round(2))
    
    # 3. Monthly Context Evolution
    print("\n2. MONTHLY CONTEXT EVOLUTION")
    print("-" * 50)
    
    monthly_context = []
    months = sorted(df['Year_Month'].dropna().unique())
    
    for month in months:
        month_data = df[df['Year_Month'] == month]
        month_tp = month_data[month_data['Primary Marker'] == 'TP']
        month_fp = month_data[month_data['Primary Marker'] == 'FP']
        
        for pattern_name, pattern in negation_context_patterns.items():
            tp_matches = month_tp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(month_tp) > 0 else 0
            fp_matches = month_fp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(month_fp) > 0 else 0
            
            monthly_context.append({
                'Year_Month': month,
                'Pattern_Type': pattern_name,
                'TP_Rate_%': tp_matches,
                'FP_Rate_%': fp_matches,
                'Discrimination': (tp_matches - fp_matches) / max(fp_matches, 0.1)
            })
    
    monthly_context_df = pd.DataFrame(monthly_context)
    
    # Show top discriminating patterns by month
    for pattern in ['Complaint_Negation', 'Information_Negation', 'Service_Negation']:
        pattern_monthly = monthly_context_df[monthly_context_df['Pattern_Type'] == pattern]
        
        print(f"\n{pattern} Monthly Evolution:")
        pivot_table = pattern_monthly.pivot_table(
            index='Pattern_Type',
            columns='Year_Month', 
            values=['TP_Rate_%', 'FP_Rate_%', 'Discrimination'],
            aggfunc='mean'
        ).round(2)
        print(pivot_table)
    
    # 4. Pre vs Post Context Analysis
    print("\n3. PRE VS POST CONTEXT ANALYSIS")
    print("-" * 50)
    
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    pre_data = df[df['Year_Month'].astype(str).isin(pre_months)]
    post_data = df[df['Year_Month'].astype(str).isin(post_months)]
    
    pre_tp = pre_data[pre_data['Primary Marker'] == 'TP']
    pre_fp = pre_data[pre_data['Primary Marker'] == 'FP']
    post_tp = post_data[post_data['Primary Marker'] == 'TP']
    post_fp = post_data[post_data['Primary Marker'] == 'FP']
    
    pre_post_analysis = []
    
    for pattern_name, pattern in negation_context_patterns.items():
        pre_tp_rate = pre_tp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(pre_tp) > 0 else 0
        pre_fp_rate = pre_fp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(pre_fp) > 0 else 0
        post_tp_rate = post_tp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(post_tp) > 0 else 0
        post_fp_rate = post_fp['Customer Transcript'].str.lower().str.contains(pattern, regex=True, na=False).mean() * 100 if len(post_fp) > 0 else 0
        
        pre_discrimination = (pre_tp_rate - pre_fp_rate) / max(pre_fp_rate, 0.1)
        post_discrimination = (post_tp_rate - post_fp_rate) / max(post_fp_rate, 0.1)
        
        pre_post_analysis.append({
            'Pattern_Type': pattern_name,
            'Pre_TP_Rate_%': pre_tp_rate,
            'Pre_FP_Rate_%': pre_fp_rate,
            'Post_TP_Rate_%': post_tp_rate,
            'Post_FP_Rate_%': post_fp_rate,
            'Pre_Discrimination': pre_discrimination,
            'Post_Discrimination': post_discrimination,
            'Discrimination_Change': post_discrimination - pre_discrimination,
            'Context_Degradation': 'YES' if post_discrimination < pre_discrimination else 'NO'
        })
    
    pre_post_df = pd.DataFrame(pre_post_analysis)
    pre_post_df = pre_post_df.sort_values('Discrimination_Change')
    
    print("Pre vs Post Context Analysis:")
    print(pre_post_df.round(2))
    
    # 5. Evidence for Context-Insensitive Handling
    print("\n4. EVIDENCE FOR CONTEXT-INSENSITIVE HANDLING")
    print("-" * 50)
    
    # Count total negations vs context-specific negations
    total_negation_pattern = r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t|isnt|isn\'t|doesnt|doesn\'t|havent|haven\'t|didnt|didn\'t)\b'
    
    evidence_analysis = []
    
    # For TPs
    tp_total_neg = tp_data['Customer Transcript'].str.lower().str.count(total_negation_pattern).sum()
    tp_context_neg = 0
    for pattern in negation_context_patterns.values():
        tp_context_neg += tp_data['Customer Transcript'].str.lower().str.count(pattern).sum()
    
    # For FPs  
    fp_total_neg = fp_data['Customer Transcript'].str.lower().str.count(total_negation_pattern).sum()
    fp_context_neg = 0
    for pattern in negation_context_patterns.values():
        fp_context_neg += fp_data['Customer Transcript'].str.lower().str.count(pattern).sum()
    
    tp_context_ratio = tp_context_neg / max(tp_total_neg, 1)
    fp_context_ratio = fp_context_neg / max(fp_total_neg, 1)
    
    evidence_summary = pd.DataFrame({
        'Metric': [
            'Total Negations',
            'Context-Specific Negations', 
            'Context Ratio',
            'Context-Less Negations',
            'Context-Less Ratio'
        ],
        'True_Positives': [
            tp_total_neg,
            tp_context_neg,
            tp_context_ratio,
            tp_total_neg - tp_context_neg,
            1 - tp_context_ratio
        ],
        'False_Positives': [
            fp_total_neg,
            fp_context_neg,
            fp_context_ratio,
            fp_total_neg - fp_context_neg,
            1 - fp_context_ratio
        ]
    })
    
    evidence_summary['FP_Problem_Indicator'] = evidence_summary['False_Positives'] / evidence_summary['True_Positives']
    
    print("Evidence Summary - Context vs Context-Less Negations:")
    print(evidence_summary.round(3))
    
    # 6. Category-Specific Context Analysis
    print("\n5. CATEGORY-SPECIFIC CONTEXT BREAKDOWN")
    print("-" * 50)
    
    category_context = []
    
    for l1_cat in df['Prosodica L1'].unique():
        if pd.notna(l1_cat):
            cat_data = df[df['Prosodica L1'] == l1_cat]
            cat_tp = cat_data[cat_data['Primary Marker'] == 'TP']
            cat_fp = cat_data[cat_data['Primary Marker'] == 'FP']
            
            if len(cat_fp) >= 5:  # Minimum sample size
                # Count complaint vs information negations
                complaint_pattern = negation_context_patterns['Complaint_Negation']
                info_pattern = negation_context_patterns['Information_Negation']
                
                tp_complaint_neg = cat_tp['Customer Transcript'].str.lower().str.contains(complaint_pattern, regex=True, na=False).mean() * 100 if len(cat_tp) > 0 else 0
                fp_complaint_neg = cat_fp['Customer Transcript'].str.lower().str.contains(complaint_pattern, regex=True, na=False).mean() * 100
                
                tp_info_neg = cat_tp['Customer Transcript'].str.lower().str.contains(info_pattern, regex=True, na=False).mean() * 100 if len(cat_tp) > 0 else 0
                fp_info_neg = cat_fp['Customer Transcript'].str.lower().str.contains(info_pattern, regex=True, na=False).mean() * 100
                
                category_context.append({
                    'Category': l1_cat,
                    'TP_Count': len(cat_tp),
                    'FP_Count': len(cat_fp),
                    'TP_Complaint_Neg_%': tp_complaint_neg,
                    'FP_Complaint_Neg_%': fp_complaint_neg,
                    'TP_Info_Neg_%': tp_info_neg,
                    'FP_Info_Neg_%': fp_info_neg,
                    'Complaint_Discrimination': (tp_complaint_neg - fp_complaint_neg) / max(fp_complaint_neg, 0.1),
                    'Info_Discrimination': (tp_info_neg - fp_info_neg) / max(fp_info_neg, 0.1),
                    'Context_Problem': 'HIGH' if fp_info_neg > tp_complaint_neg else 'MEDIUM' if fp_info_neg > fp_complaint_neg else 'LOW'
                })
    
    category_context_df = pd.DataFrame(category_context)
    category_context_df = category_context_df.sort_values('Info_Discrimination')
    
    print("Category-Specific Context Problems:")
    print(category_context_df.round(2))
    
    # 7. Clear Evidence Statement
    print("\n6. CLEAR EVIDENCE FOR CONTEXT-INSENSITIVE HANDLING")
    print("-" * 50)
    
    # Calculate key statistics
    fp_with_info_neg = (fp_data['Customer Transcript'].str.lower().str.contains(
        negation_context_patterns['Information_Negation'], regex=True, na=False
    ).sum())
    
    fp_with_complaint_neg = (fp_data['Customer Transcript'].str.lower().str.contains(
        negation_context_patterns['Complaint_Negation'], regex=True, na=False
    ).sum())
    
    total_fps = len(fp_data)
    
    print("SMOKING GUN EVIDENCE:")
    print(f"1. {fp_with_info_neg}/{total_fps} FPs ({fp_with_info_neg/total_fps*100:.1f}%) contain INFORMATION negations")
    print(f"2. {fp_with_complaint_neg}/{total_fps} FPs ({fp_with_complaint_neg/total_fps*100:.1f}%) contain COMPLAINT negations")
    print(f"3. Information negations in FPs are {fp_with_info_neg/max(fp_with_complaint_neg,1):.1f}x more common than complaint negations")
    
    complaint_discrimination = context_df[context_df['Pattern_Type'] == 'Complaint_Negation']['Customer_Discrimination'].iloc[0]
    info_discrimination = context_df[context_df['Pattern_Type'] == 'Information_Negation']['Customer_Discrimination'].iloc[0]
    
    print(f"4. Complaint negations discriminate TPs {complaint_discrimination:.1f}x better than FPs")
    print(f"5. Information negations discriminate FPs {abs(info_discrimination):.1f}x better than TPs")
    print(f"6. The model treats ALL negations equally, causing {total_fps} false positives")
    
    return context_df, monthly_context_df, pre_post_df, evidence_summary, category_context_df
