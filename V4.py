def run_complete_analysis_step_by_step():
    """
    Run the complete contingency table analysis step by step with progress indicators
    This function executes all 34 analyses in sequence with clear output
    """
    
    import time
    
    print("STARTING COMPLETE PRECISION DROP ANALYSIS")
    print("="*100)
    print("This will generate 34 contingency tables covering all aspects of precision analysis")
    print("Estimated time: 3-5 minutes")
    print("="*100)
    
    start_time = time.time()
    
    # =============================================================================
    # STEP 1: DATA PREPARATION
    # =============================================================================
    
    print("\nSTEP 1: DATA PREPARATION")
    print("-" * 50)
    
    try:
        df_main, df_validation, df_rules_filtered = enhanced_data_preprocessing()
        
        if df_main is None:
            print("ERROR: Data loading failed. Please check your file paths.")
            return None, None
            
        print("SUCCESS: Data preparation completed successfully!")
        print(f"   Dataset: {len(df_main):,} records")
        print(f"   Time periods: {df_main['Year_Month'].nunique()} months")
        print(f"   Pre-period: {len(df_main[df_main['Period']=='Pre']):,} records")
        print(f"   Post-period: {len(df_main[df_main['Period']=='Post']):,} records")
        
    except Exception as e:
        print(f"ERROR in data preparation: {e}")
        return None, None
    
    # Store all results
    all_results = {}
    analysis_count = 0
    
    # =============================================================================
    # STEP 2: CORE HYPOTHESIS TESTING (4 ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 2: CORE HYPOTHESIS TESTING (4 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        hypothesis_results = comprehensive_hypothesis_testing(df_main)
        all_results['hypotheses'] = hypothesis_results
        analysis_count += len(hypothesis_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Hypothesis testing completed in {step_time:.1f} seconds")
        print(f"   Generated {len(hypothesis_results)} contingency tables")
        print(f"   Analyzed: Negation, Qualifying Language, Transcript Length, Agent Ratio")
        
    except Exception as e:
        print(f"ERROR in hypothesis testing: {e}")
        return None, None
    
    # =============================================================================
    # STEP 3: MACRO LEVEL ANALYSIS (4 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 3: MACRO LEVEL ANALYSIS (4 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        macro_results = macro_level_contingency_analysis(df_main)
        all_results['macro'] = macro_results
        analysis_count += len(macro_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Macro analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(macro_results)} contingency tables")
        print(f"   Analyzed: MoM Changes, Category Impact, Distribution Patterns, New Categories")
        
    except Exception as e:
        print(f"ERROR in macro analysis: {e}")
        # Continue with other analyses
        all_results['macro'] = {}
    
    # =============================================================================
    # STEP 4: PATTERN DETECTION (2 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 4: PATTERN DETECTION ANALYSIS (2 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        pattern_results = pattern_detection_contingency_analysis(df_main)
        all_results['pattern_detection'] = pattern_results
        analysis_count += len(pattern_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Pattern detection completed in {step_time:.1f} seconds")
        print(f"   Generated {len(pattern_results)} contingency tables")
        print(f"   Analyzed: Problem Periods, Drop Velocity Patterns")
        
    except Exception as e:
        print(f"ERROR in pattern detection: {e}")
        all_results['pattern_detection'] = {}
    
    # =============================================================================
    # STEP 5: FP ANALYSIS (3 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 5: FALSE POSITIVE ANALYSIS (3 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        fp_results = fp_analysis_contingency_tables(df_main)
        all_results['fp_analysis'] = fp_results
        analysis_count += len(fp_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: FP analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(fp_results)} contingency tables")
        print(f"   Analyzed: FP Distribution, SRSRWI Patterns, Root Causes")
        
    except Exception as e:
        print(f"ERROR in FP analysis: {e}")
        all_results['fp_analysis'] = {}
    
    # =============================================================================
    # STEP 6: VALIDATION ANALYSIS (2 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 6: VALIDATION ANALYSIS (2 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        validation_results = validation_analysis_contingency_tables(df_main)
        if validation_results:
            all_results['validation_new'] = validation_results
            analysis_count += len(validation_results)
            print(f"SUCCESS: Validation analysis completed in {time.time() - step_start:.1f} seconds")
            print(f"   Generated {len(validation_results)} contingency tables")
            print(f"   Analyzed: Consistency Trends, Process Changes")
        else:
            print("WARNING: No validation data available - skipping validation analysis")
            all_results['validation_new'] = {}
        
    except Exception as e:
        print(f"ERROR in validation analysis: {e}")
        all_results['validation_new'] = {}
    
    # =============================================================================
    # STEP 7: TEMPORAL ANALYSIS (1 NEW ANALYSIS)
    # =============================================================================
    
    print(f"\nSTEP 7: TEMPORAL ANALYSIS (1 analysis)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        temporal_results = temporal_analysis_contingency_tables(df_main)
        all_results['temporal_new'] = temporal_results
        analysis_count += len(temporal_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Temporal analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(temporal_results)} contingency tables")
        print(f"   Analyzed: Operational Change Patterns")
        
    except Exception as e:
        print(f"ERROR in temporal analysis: {e}")
        all_results['temporal_new'] = {}
    
    # =============================================================================
    # STEP 8: ROOT CAUSE ANALYSIS (3 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 8: ROOT CAUSE ANALYSIS (3 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        root_cause_results = root_cause_contingency_analysis(df_main, df_rules_filtered)
        all_results['root_cause'] = root_cause_results
        analysis_count += len(root_cause_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Root cause analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(root_cause_results)} contingency tables")
        print(f"   Analyzed: Rule Effectiveness, Degradation, Language Evolution")
        
    except Exception as e:
        print(f"ERROR in root cause analysis: {e}")
        all_results['root_cause'] = {}
    
    # =============================================================================
    # STEP 9: CROSS-CATEGORY ANALYSIS (1 NEW ANALYSIS)
    # =============================================================================
    
    print(f"\nSTEP 9: CROSS-CATEGORY ANALYSIS (1 analysis)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        cross_category_results = cross_category_contingency_analysis(df_main)
        all_results['cross_category'] = cross_category_results
        analysis_count += len(cross_category_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Cross-category analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(cross_category_results)} contingency tables")
        print(f"   Analyzed: Multi vs Single Category Performance")
        
    except Exception as e:
        print(f"ERROR in cross-category analysis: {e}")
        all_results['cross_category'] = {}
    
    # =============================================================================
    # STEP 10: ALL CATEGORY PRECISION (2 NEW ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 10: ALL CATEGORY PRECISION ANALYSIS (2 analyses)")
    print("-" * 50)
    
    try:
        step_start = time.time()
        all_precision_results = all_category_precision_contingency_analysis(df_main)
        all_results['all_precision'] = all_precision_results
        analysis_count += len(all_precision_results)
        step_time = time.time() - step_start
        
        print(f"SUCCESS: Category precision analysis completed in {step_time:.1f} seconds")
        print(f"   Generated {len(all_precision_results)} contingency tables")
        print(f"   Analyzed: All Category Performance, Drop Contributors")
        
    except Exception as e:
        print(f"ERROR in category precision analysis: {e}")
        all_results['all_precision'] = {}
    
    # =============================================================================
    # STEP 11: REMAINING ORIGINAL ANALYSES (16 ANALYSES)
    # =============================================================================
    
    print(f"\nSTEP 11: ORIGINAL ENHANCED ANALYSES (16 analyses)")
    print("-" * 50)
    
    try:
        # Volume & Performance Analysis
        print("   Running volume & performance analysis...")
        category_analysis, volume_results = advanced_volume_performance_analysis(df_main)
        all_results['volume'] = volume_results
        analysis_count += 1
        
        # Temporal Pattern Analysis  
        print("   Running temporal pattern analysis...")
        temporal_original = advanced_temporal_analysis(df_main)
        all_results['temporal'] = temporal_original
        analysis_count += len(temporal_original)
        
        # Validation Analysis
        print("   Running enhanced validation analysis...")
        validation_original = enhanced_validation_analysis(df_main)
        if validation_original:
            all_results['validation'] = validation_original
            analysis_count += len([k for k in validation_original.keys() if k != 'agreement_stats'])
        
        # FP Pattern Analysis
        print("   Running FP pattern analysis...")
        fp_pattern_results = advanced_fp_pattern_analysis(df_main)
        all_results['fp_patterns'] = fp_pattern_results
        analysis_count += len(fp_pattern_results)
        
        # Content & Context Analysis
        print("   Running content & context analysis...")
        content_results = advanced_content_context_analysis(df_main)
        all_results['content'] = content_results
        analysis_count += len(content_results)
        
        # Query Effectiveness Analysis
        print("   Running query effectiveness analysis...")
        query_results = comprehensive_query_effectiveness_analysis(df_main, df_rules_filtered)
        if query_results:
            all_results['query'] = query_results
            analysis_count += 1
        
        # Monthly Trends
        print("   Running monthly trends analysis...")
        monthly_trends, monthly_results = calculate_overall_monthly_trends(df_main)
        all_results['monthly'] = monthly_results
        analysis_count += 1
        
        print(f"SUCCESS: Original enhanced analyses completed")
        print(f"   Generated 16 additional contingency tables")
        
    except Exception as e:
        print(f"ERROR in original analyses: {e}")
    
    # =============================================================================
    # STEP 12: COMPREHENSIVE SUMMARY
    # =============================================================================
    
    print(f"\nSTEP 12: GENERATING COMPREHENSIVE SUMMARY")
    print("-" * 50)
    
    try:
        significant_changes = generate_complete_analysis_summary(all_results, df_main)
        
        total_time = time.time() - start_time
        
        print(f"\nANALYSIS COMPLETE!")
        print("="*100)
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Total contingency tables generated: {analysis_count}")
        print(f"Coverage: {analysis_count/32*100:.0f}% of original scope")
        print(f"Target achieved: {'YES' if analysis_count >= 32 else 'NO'}")
        print("="*100)
        
        return all_results, df_main
        
    except Exception as e:
        print(f"ERROR in summary generation: {e}")
        return all_results, df_main

# Convenience function to run just a quick test
def quick_test():
    """Run a quick test of the first few analyses"""
    
    print("QUICK TEST - Running first 3 analysis steps")
    print("="*60)
    
    # Data prep
    df_main, df_validation, df_rules_filtered = enhanced_data_preprocessing()
    if df_main is None:
        return None
    
    # Test 1: Hypothesis testing
    print("\n1. Testing hypothesis analysis...")
    hypothesis_results = comprehensive_hypothesis_testing(df_main)
    print(f"SUCCESS: Generated {len(hypothesis_results)} hypothesis contingency tables")
    
    # Test 2: Macro analysis  
    print("\n2. Testing macro analysis...")
    macro_results = macro_level_contingency_analysis(df_main)
    print(f"SUCCESS: Generated {len(macro_results)} macro contingency tables")
    
    # Test 3: Pattern detection
    print("\n3. Testing pattern detection...")
    pattern_results = pattern_detection_contingency_analysis(df_main)
    print(f"SUCCESS: Generated {len(pattern_results)} pattern contingency tables")
    
    print(f"\nQuick test completed successfully!")
    print(f"Total tables generated: {len(hypothesis_results) + len(macro_results) + len(pattern_results)}")
    
    return df_main
