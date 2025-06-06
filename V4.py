def generate_comprehensive_findings_and_recommendations():
    """Synthesize all findings and generate actionable recommendations"""
    
    print("\n1. KEY FINDINGS SUMMARY")
    print("-" * 30)
    
    # Overall metrics with zero-division protection
    overall_precision = df_main['Is_TP'].mean()
    total_records = len(df_main)
    
    # FIX: Add protection against empty complaint_performance DataFrame
    if len(complaint_performance) > 0:
        categories_below_target = len(complaint_performance[complaint_performance['Precision'] < 0.70])
        total_categories = len(complaint_performance)
        category_percentage = categories_below_target/total_categories if total_categories > 0 else 0
    else:
        categories_below_target = 0
        total_categories = 0
        category_percentage = 0
    
    print(f"CURRENT STATE:")
    print(f"  Overall Precision: {overall_precision:.1%} (Target: 70%)")
    print(f"  Gap to Target: {0.70 - overall_precision:+.1%}")
    
    # FIX: Conditional formatting based on whether we have categories
    if total_categories > 0:
        print(f"  Categories Below Target: {categories_below_target}/{total_categories} ({category_percentage:.1%})")
    else:
        print(f"  Categories Below Target: No complaint categories found in data")
    
    print(f"  Total Records Analyzed: {total_records:,}")
    
    # Top findings from each analysis area
    print(f"\nMAJOR FINDINGS BY ANALYSIS AREA:")
    
    print(f"\nMacro Level Analysis:")
    # FIX: Check if top_drop_drivers exists and has data
    if 'top_drop_drivers' in globals() and len(top_drop_drivers) > 0:
        worst_category = top_drop_drivers.iloc[0]
        print(f"  - Worst performing category: {worst_category['L2_Category']} ({worst_category['Precision']:.1%} precision)")
    else:
        print(f"  - No specific worst performing category identified")
    
    # FIX: Check if period_comparison exists and has the expected structure
    if 'period_comparison' in globals() and len(period_comparison) >= 2:
        period_diff = period_comparison.loc[1, 'Precision'] - period_comparison.loc[0, 'Precision']
        print(f"  - {period_diff:+.1%} precision change (normal → problem periods)")
    else:
        print(f"  - Period comparison data not available")
    
    print(f"\nDeep Dive Analysis:")
    # FIX: Check if fp_reasons exists and has data
    if 'fp_reasons' in globals() and len(fp_reasons) > 0:
        top_fp_reason = fp_reasons.iloc[0]
        print(f"  - Primary FP cause: {top_fp_reason['FP_Reason']} ({top_fp_reason['Percentage']:.1f}% of FPs)")
    else:
        print(f"  - FP reason analysis not available")
    
    # FIX: Check validation_monthly exists and has data
    if 'validation_monthly' in globals() and validation_monthly is not None and len(validation_monthly) > 0:
        avg_agreement = validation_monthly['Agreement_Rate'].mean()
        print(f"  - Validation agreement rate: {avg_agreement:.1%}")
    else:
        print(f"  - Validation data not available")
    
    print(f"\nRoot Cause Analysis:")
    # FIX: Check if rule_performance exists and has data
    if 'rule_performance' in globals() and len(rule_performance) > 0:
        degrading_rules = sum(1 for perf in rule_performance.values() if perf.get('trend') == 'Degrading')
        print(f"  - Rules showing degradation: {degrading_rules}/{len(rule_performance)}")
    else:
        print(f"  - Rule performance analysis not available")
    
    # FIX: Check if pattern_results exists and has data
    if 'pattern_results' in globals() and len(pattern_results) > 0:
        high_risk_patterns = len(pattern_results[pattern_results['Risk_Factor'] > 2])
        print(f"  - High-risk content patterns: {high_risk_patterns}")
    else:
        print(f"  - Pattern analysis not available")
    
    print(f"\n2. ROOT CAUSE PRIORITIZATION")
    print("-" * 35)
    
    # Calculate impact scores for different root causes
    root_causes = []
    
    # FIX: Add checks before accessing analysis results
    # Negation handling issues
    if 'fp_reasons' in globals() and len(fp_reasons) > 0:
        context_issues_data = fp_reasons[fp_reasons['FP_Reason'] == 'Context Issues']
        context_issues_pct = context_issues_data['Percentage'].iloc[0] if len(context_issues_data) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Context-insensitive negation handling',
            'Impact_Score': context_issues_pct,
            'Implementation_Effort': 'Medium',
            'Time_to_Fix': '2-4 weeks',
            'Expected_Gain': min(0.15, context_issues_pct * 0.01)
        })
    
    # Agent explanation issues
    if 'fp_reasons' in globals() and len(fp_reasons) > 0:
        confusion_data = fp_reasons[fp_reasons['FP_Reason'] == 'Agent/Customer Confusion']
        confusion_pct = confusion_data['Percentage'].iloc[0] if len(confusion_data) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Agent explanations triggering rules',
            'Impact_Score': confusion_pct,
            'Implementation_Effort': 'Low',
            'Time_to_Fix': '1-2 weeks',
            'Expected_Gain': min(0.08, confusion_pct * 0.008)
        })
    
    # Overly broad rules
    if 'fp_reasons' in globals() and len(fp_reasons) > 0:
        broad_rules_data = fp_reasons[fp_reasons['FP_Reason'] == 'Overly Broad Rules']
        broad_rules_pct = broad_rules_data['Percentage'].iloc[0] if len(broad_rules_data) > 0 else 0
        root_causes.append({
            'Root_Cause': 'Overly broad query rules',
            'Impact_Score': broad_rules_pct,
            'Implementation_Effort': 'High',
            'Time_to_Fix': '6-12 weeks',
            'Expected_Gain': min(0.12, broad_rules_pct * 0.012)
        })
    
    # Validation inconsistency
    if 'validation_monthly' in globals() and validation_monthly is not None and len(validation_monthly) > 0:
        avg_agreement = validation_monthly['Agreement_Rate'].mean()
        validation_impact = (1 - avg_agreement) * 100 if avg_agreement < 0.85 else 0
        root_causes.append({
            'Root_Cause': 'Validation process inconsistency',
            'Impact_Score': validation_impact,
            'Implementation_Effort': 'Medium',
            'Time_to_Fix': '3-6 weeks',
            'Expected_Gain': min(0.05, validation_impact * 0.005)
        })
    
    # FIX: Handle case where no root causes were identified
    if len(root_causes) == 0:
        print("No specific root causes identified from available data")
        print("Recommend manual investigation of data quality and analysis pipeline")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Sort by expected gain
    root_causes_df = pd.DataFrame(root_causes).sort_values('Expected_Gain', ascending=False)
    
    print("Prioritized Root Causes:")
    print(f"{'Root Cause':<35} {'Impact':<8} {'Effort':<8} {'Time':<12} {'Expected Gain':<12}")
    print("-" * 85)
    
    for _, cause in root_causes_df.iterrows():
        print(f"{cause['Root_Cause']:<35} {cause['Impact_Score']:<8.1f} {cause['Implementation_Effort']:<8} "
              f"{cause['Time_to_Fix']:<12} {cause['Expected_Gain']:<12.1%}")
    
    print(f"\n3. IMMEDIATE ACTION PLAN")
    print("-" * 30)
    
    print("WEEK 1-2 (CRITICAL FIXES):")
    if len(root_causes_df) > 0:
        top_cause = root_causes_df.iloc[0]
        print(f"1. Address {top_cause['Root_Cause']}")
        
        if 'negation' in top_cause['Root_Cause'].lower():
            print("   - Add universal negation template: (query) AND NOT ((not|no|never) NEAR:3 (complain|complaint))")
        elif 'agent' in top_cause['Root_Cause'].lower():
            print("   - Add agent explanation filter: AND NOT ((explain|example|suppose) NEAR:5 (complaint))")
        elif 'broad' in top_cause['Root_Cause'].lower():
            print("   - Review and reduce OR clauses in top 5 worst-performing queries")
    else:
        print("1. Conduct data quality assessment")
        print("   - Verify data completeness and structure")
        print("   - Validate analysis pipeline")
    
    print("2. Fix top 3 worst-performing categories:")
    if 'top_drop_drivers' in globals() and len(top_drop_drivers) >= 3:
        for i, (_, category) in enumerate(top_drop_drivers.head(3).iterrows()):
            print(f"   - {category['L2_Category']}: Current {category['Precision']:.1%} → Target 70%")
    else:
        print("   - Category-specific improvements to be determined after data review")
    
    print("3. Implement daily monitoring dashboard")
    print("   - Real-time precision tracking")
    print("   - Category performance alerts")
    print("   - FP pattern detection")
    
    print(f"\nMONTH 1 (SYSTEMATIC IMPROVEMENTS):")
    print("1. Query optimization program:")
    print("   - Standardize negation handling across all queries")
    print("   - Optimize channel selection (customer vs both)")
    print("   - Reduce query complexity for poor performers")
    
    print("2. Enhanced validation process:")
    if 'validation_monthly' in globals() and validation_monthly is not None:
        print("   - Reviewer calibration sessions")
        print("   - Updated validation guidelines")
        print("   - Quality control sampling")
    else:
        print("   - Establish validation process framework")
        print("   - Implement secondary validation sampling")
    
    print("3. Pattern-based improvements:")
    if 'pattern_results' in globals() and len(pattern_results) > 0:
        high_risk = pattern_results[pattern_results['Risk_Factor'] > 2]
        if len(high_risk) > 0:
            print(f"   - Address high-risk patterns: {', '.join(high_risk['Pattern'].tolist())}")
        else:
            print("   - Develop pattern detection capabilities")
    else:
        print("   - Implement content pattern analysis")
    
    print(f"\nQUARTER 1 (STRATEGIC INITIATIVES):")
    print("1. Advanced analytics implementation:")
    print("   - ML-based FP prediction")
    print("   - Automated pattern detection")
    print("   - Dynamic threshold optimization")
    
    print("2. Platform enhancements:")
    print("   - Context-aware rule engine")
    print("   - Speaker role detection")
    print("   - Semantic understanding layer")
    
    print(f"\n4. SUCCESS METRICS AND MONITORING")
    print("-" * 40)
    
    # Calculate expected outcomes
    if len(root_causes_df) > 0:
        total_expected_gain = root_causes_df['Expected_Gain'].sum()
        final_precision = overall_precision + total_expected_gain
        
        print(f"EXPECTED OUTCOMES:")
        print(f"  Current Precision: {overall_precision:.1%}")
        print(f"  Expected Gain: +{total_expected_gain:.1%}")
        print(f"  Target Precision: {final_precision:.1%}")
        print(f"  Target Achievement: {'YES' if final_precision >= 0.70 else 'PARTIAL'}")
    else:
        print(f"EXPECTED OUTCOMES:")
        print(f"  Current Precision: {overall_precision:.1%}")
        print(f"  Expected Gain: To be determined after data review")
        print(f"  Target Precision: 70%")
        print(f"  Target Achievement: Requires further analysis")
    
    print(f"\nKEY PERFORMANCE INDICATORS:")
    print(f"  Primary: Overall precision ≥ 70%")
    print(f"  Secondary: All categories ≥ 60% precision")
    print(f"  Tertiary: Validation agreement ≥ 85%")
    
    print(f"\nMONITORING FRAMEWORK:")
    print(f"  Daily: Precision tracking, volume monitoring")
    print(f"  Weekly: Category performance review, FP pattern analysis")
    print(f"  Monthly: Validation assessment, rule effectiveness review")
    
    print(f"\n5. RISK MITIGATION")
    print("-" * 25)
    
    print(f"HIGH-RISK SCENARIOS:")
    print(f"  - Precision drops >15% month-over-month")
    print(f"  - New category launches without validation")
    print(f"  - Validation disagreement >25%")
    print(f"  - Rule changes without impact assessment")
    
    print(f"\nMITIGATION STRATEGIES:")
    print(f"  - Automated alerts for significant changes")
    print(f"  - Staged rollout for rule modifications")
    print(f"  - Regular backup validation sampling")
    print(f"  - Emergency response procedures")
    
    return root_causes_df
