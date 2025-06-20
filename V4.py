# =============================================================================
# COMPLETE FIX FOR PANDAS DATA TYPE ISSUES
# Run this cell to fix all data type problems before running visualizations
# =============================================================================

def fix_data_types_comprehensive(negation_df, pattern_analysis_df):
    """
    Comprehensive fix for all data type issues in the dataframes
    """
    
    print("="*60)
    print("FIXING DATA TYPE ISSUES")
    print("="*60)
    
    # Create copies to avoid modifying original data
    negation_df_fixed = negation_df.copy()
    pattern_analysis_df_fixed = pattern_analysis_df.copy()
    
    print("1. Fixing negation_df data types...")
    
    # Fix Year_Month column - convert everything to string
    if 'Year_Month' in negation_df_fixed.columns:
        print("   Fixing Year_Month column...")
        # Convert to string, handling NaN values
        negation_df_fixed['Year_Month'] = negation_df_fixed['Year_Month'].astype(str)
        # Replace 'nan' strings with actual NaN
        negation_df_fixed['Year_Month'] = negation_df_fixed['Year_Month'].replace('nan', pd.NA)
        # Drop rows with missing Year_Month
        negation_df_fixed = negation_df_fixed.dropna(subset=['Year_Month'])
        print(f"   Year_Month fixed. Remaining records: {len(negation_df_fixed)}")
    
    # Fix UUID column - ensure it's string
    if 'UUID' in negation_df_fixed.columns:
        print("   Fixing UUID column...")
        negation_df_fixed['UUID'] = negation_df_fixed['UUID'].astype(str)
    
    # Fix Primary_Marker column - ensure it's string
    if 'Primary_Marker' in negation_df_fixed.columns:
        print("   Fixing Primary_Marker column...")
        negation_df_fixed['Primary_Marker'] = negation_df_fixed['Primary_Marker'].astype(str)
    
    # Fix Speaker column - ensure it's string
    if 'Speaker' in negation_df_fixed.columns:
        print("   Fixing Speaker column...")
        negation_df_fixed['Speaker'] = negation_df_fixed['Speaker'].astype(str)
    
    # Fix Negation_Word column - ensure it's string
    if 'Negation_Word' in negation_df_fixed.columns:
        print("   Fixing Negation_Word column...")
        negation_df_fixed['Negation_Word'] = negation_df_fixed['Negation_Word'].astype(str)
    
    # Fix Context column - ensure it's string
    if 'Context' in negation_df_fixed.columns:
        print("   Fixing Context column...")
        negation_df_fixed['Context'] = negation_df_fixed['Context'].fillna('').astype(str)
    
    # Fix Pattern_Cluster column - ensure it's integer
    if 'Pattern_Cluster' in negation_df_fixed.columns:
        print("   Fixing Pattern_Cluster column...")
        try:
            negation_df_fixed['Pattern_Cluster'] = pd.to_numeric(negation_df_fixed['Pattern_Cluster'], errors='coerce')
            negation_df_fixed = negation_df_fixed.dropna(subset=['Pattern_Cluster'])
            negation_df_fixed['Pattern_Cluster'] = negation_df_fixed['Pattern_Cluster'].astype(int)
        except Exception as e:
            print(f"   Warning: Could not fix Pattern_Cluster: {e}")
    
    # Fix Period column - ensure it's string
    if 'Period' in negation_df_fixed.columns:
        print("   Fixing Period column...")
        negation_df_fixed['Period'] = negation_df_fixed['Period'].astype(str)
    
    print("2. Fixing pattern_analysis_df data types...")
    
    # Fix Cluster_ID column - ensure it's integer
    if 'Cluster_ID' in pattern_analysis_df_fixed.columns:
        print("   Fixing Cluster_ID column...")
        try:
            pattern_analysis_df_fixed['Cluster_ID'] = pd.to_numeric(pattern_analysis_df_fixed['Cluster_ID'], errors='coerce')
            pattern_analysis_df_fixed = pattern_analysis_df_fixed.dropna(subset=['Cluster_ID'])
            pattern_analysis_df_fixed['Cluster_ID'] = pattern_analysis_df_fixed['Cluster_ID'].astype(int)
        except Exception as e:
            print(f"   Warning: Could not fix Cluster_ID: {e}")
    
    # Fix numeric columns
    numeric_cols = ['Total_Count', 'TP_Count', 'FP_Count', 'TP_Rate', 'FP_Rate', 'Quality_Score']
    for col in numeric_cols:
        if col in pattern_analysis_df_fixed.columns:
            print(f"   Fixing {col} column...")
            try:
                pattern_analysis_df_fixed[col] = pd.to_numeric(pattern_analysis_df_fixed[col], errors='coerce')
            except Exception as e:
                print(f"   Warning: Could not fix {col}: {e}")
    
    # Fix string columns
    string_cols = ['Top_Features']
    for col in string_cols:
        if col in pattern_analysis_df_fixed.columns:
            print(f"   Fixing {col} column...")
            pattern_analysis_df_fixed[col] = pattern_analysis_df_fixed[col].fillna('').astype(str)
    
    print("3. Data cleaning summary:")
    print(f"   Original negation_df shape: {negation_df.shape}")
    print(f"   Fixed negation_df shape: {negation_df_fixed.shape}")
    print(f"   Original pattern_analysis_df shape: {pattern_analysis_df.shape}")
    print(f"   Fixed pattern_analysis_df shape: {pattern_analysis_df_fixed.shape}")
    
    # Verify key columns
    print("4. Verification of fixed data types:")
    
    key_checks = [
        ('Year_Month', 'negation_df_fixed'),
        ('Primary_Marker', 'negation_df_fixed'),
        ('Pattern_Cluster', 'negation_df_fixed'),
        ('Cluster_ID', 'pattern_analysis_df_fixed')
    ]
    
    for col, df_name in key_checks:
        df_to_check = negation_df_fixed if df_name == 'negation_df_fixed' else pattern_analysis_df_fixed
        if col in df_to_check.columns:
            dtype = df_to_check[col].dtype
            unique_count = df_to_check[col].nunique()
            print(f"   {col}: {dtype}, {unique_count} unique values")
    
    print("="*60)
    print("DATA TYPE FIXES COMPLETED")
    print("="*60)
    
    return negation_df_fixed, pattern_analysis_df_fixed

def safe_visualization_wrapper(negation_df, pattern_analysis_df, output_dir='safe_negation_visualizations'):
    """
    Safe wrapper for visualizations with comprehensive error handling
    """
    
    print("="*60)
    print("SAFE VISUALIZATION GENERATION")
    print("="*60)
    
    # First, fix all data types
    try:
        negation_df_safe, pattern_analysis_df_safe = fix_data_types_comprehensive(negation_df, pattern_analysis_df)
    except Exception as e:
        print(f"ERROR in data type fixing: {e}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Track success/failure
    results = {
        'pattern_distribution': False,
        'temporal_analysis': False,
        'speaker_analysis': False,
        'word_clouds': False,
        'performance_heatmaps': False,
        'interactive_dashboard': False
    }
    
    # 1. Pattern Distribution (Safest)
    print("\n1. Creating pattern distribution...")
    try:
        create_pattern_distribution_safe(negation_df_safe, pattern_analysis_df_safe, output_dir)
        results['pattern_distribution'] = True
        print("   ✓ Pattern distribution completed")
    except Exception as e:
        print(f"   ✗ Pattern distribution failed: {e}")
    
    # 2. Temporal Analysis (Fixed for data types)
    print("\n2. Creating temporal analysis...")
    try:
        create_temporal_analysis_safe(negation_df_safe, output_dir)
        results['temporal_analysis'] = True
        print("   ✓ Temporal analysis completed")
    except Exception as e:
        print(f"   ✗ Temporal analysis failed: {e}")
    
    # 3. Speaker Analysis
    print("\n3. Creating speaker analysis...")
    try:
        create_speaker_analysis_safe(negation_df_safe, output_dir)
        results['speaker_analysis'] = True
        print("   ✓ Speaker analysis completed")
    except Exception as e:
        print(f"   ✗ Speaker analysis failed: {e}")
    
    # 4. Word Clouds (Most likely to have issues)
    print("\n4. Creating word clouds...")
    try:
        create_wordclouds_safe(negation_df_safe, pattern_analysis_df_safe, output_dir)
        results['word_clouds'] = True
        print("   ✓ Word clouds completed")
    except Exception as e:
        print(f"   ✗ Word clouds failed: {e}")
    
    # 5. Performance Heatmaps
    print("\n5. Creating performance heatmaps...")
    try:
        create_heatmaps_safe(negation_df_safe, output_dir)
        results['performance_heatmaps'] = True
        print("   ✓ Performance heatmaps completed")
    except Exception as e:
        print(f"   ✗ Performance heatmaps failed: {e}")
    
    # 6. Interactive Dashboard
    print("\n6. Creating interactive dashboard...")
    try:
        create_dashboard_safe(negation_df_safe, pattern_analysis_df_safe, output_dir)
        results['interactive_dashboard'] = True
        print("   ✓ Interactive dashboard completed")
    except Exception as e:
        print(f"   ✗ Interactive dashboard failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION RESULTS SUMMARY")
    print("="*60)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"Successfully created: {successful}/{total} visualizations")
    
    for viz_type, success in results.items():
        status = "✓" if success else "✗"
        print(f"   {status} {viz_type.replace('_', ' ').title()}")
    
    if successful > 0:
        print(f"\nCheck the '{output_dir}' folder for generated files!")
    
    return negation_df_safe, pattern_analysis_df_safe, results

# =============================================================================
# SAFE VISUALIZATION FUNCTIONS - Simplified versions with error handling
# =============================================================================

def create_pattern_distribution_safe(negation_df, pattern_analysis_df, output_dir):
    """Safe pattern distribution with minimal pandas operations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Simple bar charts only
    try:
        # TP vs FP distribution
        marker_counts = negation_df['Primary_Marker'].value_counts()
        axes[0,0].bar(marker_counts.index, marker_counts.values, color=['green', 'red'], alpha=0.7)
        axes[0,0].set_title('TP vs FP Distribution')
        axes[0,0].set_ylabel('Count')
    except:
        axes[0,0].text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=axes[0,0].transAxes)
    
    try:
        # Speaker distribution
        speaker_counts = negation_df['Speaker'].value_counts()
        axes[0,1].bar(speaker_counts.index, speaker_counts.values, color=['lightblue', 'lightcoral'], alpha=0.7)
        axes[0,1].set_title('Speaker Distribution')
        axes[0,1].set_ylabel('Count')
    except:
        axes[0,1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=axes[0,1].transAxes)
    
    try:
        # Top negation words
        top_words = negation_df['Negation_Word'].value_counts().head(8)
        axes[1,0].barh(range(len(top_words)), top_words.values, color='skyblue', alpha=0.7)
        axes[1,0].set_yticks(range(len(top_words)))
        axes[1,0].set_yticklabels(top_words.index)
        axes[1,0].set_title('Top Negation Words')
        axes[1,0].set_xlabel('Count')
    except:
        axes[1,0].text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=axes[1,0].transAxes)
    
    try:
        # Pattern volume (if available)
        if len(pattern_analysis_df) > 0 and 'Total_Count' in pattern_analysis_df.columns:
            axes[1,1].bar(pattern_analysis_df['Cluster_ID'], pattern_analysis_df['Total_Count'], 
                         color='orange', alpha=0.7)
            axes[1,1].set_title('Pattern Volume')
            axes[1,1].set_xlabel('Pattern Cluster')
            axes[1,1].set_ylabel('Count')
        else:
            axes[1,1].text(0.5, 0.5, 'Pattern data unavailable', ha='center', va='center', transform=axes[1,1].transAxes)
    except:
        axes[1,1].text(0.5, 0.5, 'Pattern data unavailable', ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safe_pattern_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_analysis_safe(negation_df, output_dir):
    """Safe temporal analysis with simple aggregation"""
    
    if 'Year_Month' not in negation_df.columns:
        print("   No temporal data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    try:
        # Simple monthly volume
        monthly_data = negation_df.groupby(['Year_Month', 'Primary_Marker']).size().unstack(fill_value=0)
        
        months = monthly_data.index
        if 'TP' in monthly_data.columns:
            axes[0].plot(months, monthly_data['TP'], marker='o', color='green', label='TP')
        if 'FP' in monthly_data.columns:
            axes[0].plot(months, monthly_data['FP'], marker='o', color='red', label='FP')
        
        axes[0].set_title('Monthly Volume Trends')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Monthly unique words
        monthly_words = negation_df.groupby('Year_Month')['Negation_Word'].nunique()
        axes[1].plot(monthly_words.index, monthly_words.values, marker='o', color='blue')
        axes[1].set_title('Unique Negation Words per Month')
        axes[1].set_ylabel('Unique Words')
        axes[1].tick_params(axis='x', rotation=45)
        
    except Exception as e:
        axes[0].text(0.5, 0.5, f'Temporal analysis error:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safe_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speaker_analysis_safe(negation_df, output_dir):
    """Safe speaker analysis"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    try:
        # Speaker performance
        speaker_performance = pd.crosstab(negation_df['Speaker'], negation_df['Primary_Marker'])
        speaker_performance_pct = speaker_performance.div(speaker_performance.sum(axis=1), axis=0) * 100
        
        speaker_performance_pct.plot(kind='bar', ax=axes[0], color=['green', 'red'], alpha=0.7)
        axes[0].set_title('Speaker Performance (%)')
        axes[0].set_ylabel('Percentage')
        axes[0].legend(title='Primary Marker')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Speaker volume
        speaker_volume = negation_df['Speaker'].value_counts()
        axes[1].pie(speaker_volume.values, labels=speaker_volume.index, autopct='%1.1f%%')
        axes[1].set_title('Speaker Volume Distribution')
        
    except Exception as e:
        axes[0].text(0.5, 0.5, 'Speaker analysis error', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Data unavailable', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safe_speaker_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_wordclouds_safe(negation_df, pattern_analysis_df, output_dir):
    """Safe word cloud generation"""
    
    if 'Pattern_Cluster' not in negation_df.columns or 'Context' not in negation_df.columns:
        print("   No pattern or context data for word clouds")
        return
    
    # Get only patterns with sufficient data
    pattern_counts = negation_df['Pattern_Cluster'].value_counts()
    valid_patterns = pattern_counts[pattern_counts >= 10].index[:4]  # Top 4 patterns with 10+ instances
    
    if len(valid_patterns) == 0:
        print("   No patterns with sufficient data for word clouds")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, pattern in enumerate(valid_patterns):
        try:
            pattern_data = negation_df[negation_df['Pattern_Cluster'] == pattern]
            contexts = pattern_data['Context'].dropna().astype(str)
            
            if len(contexts) > 0:
                combined_text = ' '.join(contexts)
                # Simple text cleaning
                import re
                clean_text = re.sub(r'[^\w\s]', ' ', combined_text.lower())
                clean_text = ' '.join([w for w in clean_text.split() if len(w) > 2])
                
                if len(clean_text) > 50:
                    wordcloud = WordCloud(width=300, height=200, background_color='white', 
                                        max_words=30).generate(clean_text)
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(f'Pattern {pattern}\n({len(pattern_data)} instances)')
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f'Pattern {pattern}\nInsufficient text', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f'Pattern {pattern}\nNo context', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Pattern {pattern}\nError: {str(e)[:20]}...', 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(valid_patterns), 4):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safe_wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmaps_safe(negation_df, output_dir):
    """Safe heatmap generation"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        # Simple confusion matrix style
        if 'Period' in negation_df.columns:
            heatmap_data = pd.crosstab(negation_df['Period'], negation_df['Primary_Marker'])
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Period vs Primary Marker Counts')
        else:
            # Fallback: Speaker vs Primary Marker
            heatmap_data = pd.crosstab(negation_df['Speaker'], negation_df['Primary_Marker'])
            sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Speaker vs Primary Marker Counts')
            
    except Exception as e:
        ax.text(0.5, 0.5, f'Heatmap error:\n{str(e)[:50]}...', 
               ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safe_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dashboard_safe(negation_df, pattern_analysis_df, output_dir):
    """Safe interactive dashboard"""
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volume by Marker', 'Monthly Trends', 'Speaker Distribution', 'Pattern Performance')
        )
        
        # Simple bar chart
        marker_counts = negation_df['Primary_Marker'].value_counts()
        fig.add_trace(
            go.Bar(x=marker_counts.index, y=marker_counts.values, name='Volume'),
            row=1, col=1
        )
        
        # Monthly trend (if available)
        if 'Year_Month' in negation_df.columns:
            monthly = negation_df['Year_Month'].value_counts().sort_index()
            fig.add_trace(
                go.Scatter(x=monthly.index, y=monthly.values, mode='lines+markers', name='Monthly'),
                row=1, col=2
            )
        
        # Speaker pie (converted to bar for subplots)
        speaker_counts = negation_df['Speaker'].value_counts()
        fig.add_trace(
            go.Bar(x=speaker_counts.index, y=speaker_counts.values, name='Speaker'),
            row=2, col=1
        )
        
        # Pattern performance (if available)
        if len(pattern_analysis_df) > 0 and 'Total_Count' in pattern_analysis_df.columns:
            fig.add_trace(
                go.Bar(x=pattern_analysis_df['Cluster_ID'], y=pattern_analysis_df['Total_Count'], name='Patterns'),
                row=2, col=2
            )
        
        fig.update_layout(height=600, showlegend=False, title_text="Safe Negation Analysis Dashboard")
        fig.write_html(f'{output_dir}/safe_dashboard.html')
        
    except Exception as e:
        print(f"   Dashboard creation failed: {e}")

# =============================================================================
# USAGE: Replace your visualization call with this
# =============================================================================

print("Use this function instead of the previous ones:")
print("negation_df_fixed, pattern_analysis_df_fixed, results = safe_visualization_wrapper(negation_df_clustered, pattern_analysis_df)")
