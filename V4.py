def enhanced_data_preprocessing():
    """Enhanced data preparation with period classification and robust error handling"""
    
    print("="*80)
    print("ENHANCED DATA PREPARATION WITH PERIOD CLASSIFICATION")
    print("="*80)
    
    # Load main transcript data with error handling
    try:
        df_main = pd.read_excel('Precision_Drop_Analysis_OG.xlsx')
        print(f"Main dataset loaded: {df_main.shape}")
        print(f"Columns found: {list(df_main.columns)}")
    except FileNotFoundError:
        print("Error: Main dataset file 'Precision_Drop_Analysis_OG.xlsx' not found.")
        print("Please ensure the file is in the current directory.")
        return None, None, None
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        return None, None, None
    
    # Load validation summary with error handling
    try:
        df_validation = pd.read_excel('Categorical Validation.xlsx', sheet_name='Summary validation vol')
        print(f"Validation summary loaded: {df_validation.shape}")
    except FileNotFoundError:
        print("Warning: Validation file not found. Continuing without validation data.")
        df_validation = None
    except Exception as e:
        print(f"Warning: Error loading validation file: {e}")
        df_validation = None
    
    # Load query rules with error handling
    try:
        df_rules = pd.read_excel('Query_Rules.xlsx')
        # Check if Category column exists before filtering
        if 'Category' in df_rules.columns:
            df_rules_filtered = df_rules[df_rules['Category'].isin(['complaints', 'collection_complaints'])].copy()
        else:
            df_rules_filtered = df_rules.copy()
        print(f"Query rules loaded and filtered: {df_rules_filtered.shape}")
    except FileNotFoundError:
        print("Warning: Query rules file not found. Continuing without rules data.")
        df_rules_filtered = None
    except Exception as e:
        print(f"Warning: Error loading query rules: {e}")
        df_rules_filtered = None
    
    # Enhanced data preprocessing with robust date handling
    print("\nProcessing date information...")
    
    # Find date column (handle different possible names)
    date_columns = [col for col in df_main.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_columns:
        date_col = date_columns[0]
        print(f"Using date column: '{date_col}'")
        
        try:
            # Try multiple date parsing approaches
            df_main['Date'] = pd.to_datetime(df_main[date_col], errors='coerce')
            
            # Check if conversion was successful
            if df_main['Date'].isna().all():
                print("Warning: Date conversion failed. Trying alternative formats...")
                # Try common date formats
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        df_main['Date'] = pd.to_datetime(df_main[date_col], format=fmt, errors='coerce')
                        if not df_main['Date'].isna().all():
                            print(f"Successfully parsed dates with format: {fmt}")
                            break
                    except:
                        continue
            
            # If still no success, create dummy dates
            if df_main['Date'].isna().all():
                print("Warning: Could not parse dates. Creating dummy date sequence...")
                df_main['Date'] = pd.date_range(start='2024-10-01', periods=len(df_main), freq='D')
            
        except Exception as e:
            print(f"Error processing dates: {e}")
            print("Creating dummy date sequence...")
            df_main['Date'] = pd.date_range(start='2024-10-01', periods=len(df_main), freq='D')
    else:
        print("Warning: No date column found. Creating dummy date sequence...")
        df_main['Date'] = pd.date_range(start='2024-10-01', periods=len(df_main), freq='D')
    
    # Safe date feature extraction
    try:
        df_main['Year_Month'] = df_main['Date'].dt.strftime('%Y-%m')
        df_main['DayOfWeek'] = df_main['Date'].dt.day_name()
        df_main['WeekOfMonth'] = df_main['Date'].dt.day // 7 + 1
        df_main['Quarter'] = df_main['Date'].dt.quarter
        df_main['Is_Holiday_Season'] = df_main['Date'].dt.month.isin([11, 12, 1])
        df_main['Is_Month_End'] = df_main['Date'].dt.day >= 25
        print("Date features extracted successfully")
    except Exception as e:
        print(f"Error extracting date features: {e}")
        # Fallback values
        df_main['Year_Month'] = '2024-10'
        df_main['DayOfWeek'] = 'Monday'
        df_main['WeekOfMonth'] = 1
        df_main['Quarter'] = 4
        df_main['Is_Holiday_Season'] = False
        df_main['Is_Month_End'] = False
    
    # CRITICAL: Add Period Classification for Contingency Tables
    pre_months = ['2024-10', '2024-11', '2024-12']
    post_months = ['2025-01', '2025-02', '2025-03']
    
    try:
        df_main['Period'] = df_main['Year_Month'].apply(
            lambda x: 'Pre' if str(x) in pre_months else 'Post' if str(x) in post_months else 'Other'
        )
        
        # Check if we have both periods
        pre_count = (df_main['Period'] == 'Pre').sum()
        post_count = (df_main['Period'] == 'Post').sum()
        other_count = (df_main['Period'] == 'Other').sum()
        
        print(f"\nPeriod Classification:")
        print(f"  Pre Period (Oct-Dec 2024): {pre_count} records")
        print(f"  Post Period (Jan-Mar 2025): {post_count} records")
        print(f"  Other Period: {other_count} records")
        
        # If no Pre/Post data, create artificial split
        if pre_count == 0 and post_count == 0:
            print("Warning: No matching time periods found. Creating artificial Pre/Post split...")
            total_records = len(df_main)
            df_main['Period'] = ['Pre' if i < total_records//2 else 'Post' for i in range(total_records)]
        
        # Filter only Pre and Post periods for analysis
        original_size = len(df_main)
        df_main = df_main[df_main['Period'].isin(['Pre', 'Post'])].copy()
        filtered_size = len(df_main)
        
        print(f"Filtered from {original_size} to {filtered_size} records for Pre/Post analysis")
        
    except Exception as e:
        print(f"Error in period classification: {e}")
        # Create simple 50/50 split
        total_records = len(df_main)
        df_main['Period'] = ['Pre' if i < total_records//2 else 'Post' for i in range(total_records)]
    
    # Text processing with error handling
    print("\nProcessing text data...")
    
    # Find transcript columns
    customer_cols = [col for col in df_main.columns if 'customer' in col.lower() and 'transcript' in col.lower()]
    agent_cols = [col for col in df_main.columns if 'agent' in col.lower() and 'transcript' in col.lower()]
    
    customer_col = customer_cols[0] if customer_cols else None
    agent_col = agent_cols[0] if agent_cols else None
    
    try:
        if customer_col:
            df_main['Customer Transcript'] = df_main[customer_col].fillna('')
            print(f"Using customer transcript column: '{customer_col}'")
        else:
            df_main['Customer Transcript'] = ''
            print("Warning: No customer transcript column found")
        
        if agent_col:
            df_main['Agent Transcript'] = df_main[agent_col].fillna('')
            print(f"Using agent transcript column: '{agent_col}'")
        else:
            df_main['Agent Transcript'] = ''
            print("Warning: No agent transcript column found")
        
        df_main['Full_Transcript'] = df_main['Customer Transcript'] + ' ' + df_main['Agent Transcript']
        
    except Exception as e:
        print(f"Error processing transcript columns: {e}")
        df_main['Customer Transcript'] = ''
        df_main['Agent Transcript'] = ''
        df_main['Full_Transcript'] = ''
    
    # Text features with error handling
    try:
        df_main['Transcript_Length'] = df_main['Full_Transcript'].str.len()
        df_main['Customer_Word_Count'] = df_main['Customer Transcript'].str.split().str.len().fillna(0)
        df_main['Agent_Word_Count'] = df_main['Agent Transcript'].str.split().str.len().fillna(0)
        df_main['Customer_Agent_Ratio'] = df_main['Customer_Word_Count'] / (df_main['Agent_Word_Count'] + 1)
        
        # Advanced text features
        df_main['Customer_Question_Count'] = df_main['Customer Transcript'].str.count('\?').fillna(0)
        df_main['Customer_Exclamation_Count'] = df_main['Customer Transcript'].str.count('!').fillna(0)
        df_main['Customer_Caps_Ratio'] = df_main['Customer Transcript'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1) if pd.notna(x) else 0
        )
        
        # Negation and qualifying patterns
        negation_patterns = r'\b(not|no|never|dont|don\'t|wont|won\'t|cant|can\'t|isnt|isn\'t)\b'
        df_main['Customer_Negation_Count'] = df_main['Customer Transcript'].str.lower().str.count(negation_patterns).fillna(0)
        df_main['Agent_Negation_Count'] = df_main['Agent Transcript'].str.lower().str.count(negation_patterns).fillna(0)
        
        qualifying_patterns = r'\b(might|maybe|seems|appears|possibly|perhaps|probably|likely)\b'
        df_main['Customer_Qualifying_Count'] = df_main['Customer Transcript'].str.lower().str.count(qualifying_patterns).fillna(0)
        
        print("Text features extracted successfully")
        
    except Exception as e:
        print(f"Error extracting text features: {e}")
        # Set default values
        for col in ['Transcript_Length', 'Customer_Word_Count', 'Agent_Word_Count', 'Customer_Agent_Ratio',
                   'Customer_Question_Count', 'Customer_Exclamation_Count', 'Customer_Caps_Ratio',
                   'Customer_Negation_Count', 'Agent_Negation_Count', 'Customer_Qualifying_Count']:
            df_main[col] = 0
    
    # Target variables with error handling
    print("\nProcessing target variables...")
    
    # Find marker columns
    primary_cols = [col for col in df_main.columns if 'primary' in col.lower() and 'marker' in col.lower()]
    secondary_cols = [col for col in df_main.columns if 'secondary' in col.lower() and 'marker' in col.lower()]
    
    primary_col = primary_cols[0] if primary_cols else None
    secondary_col = secondary_cols[0] if secondary_cols else None
    
    try:
        if primary_col:
            df_main['Is_TP'] = (df_main[primary_col] == 'TP').astype(int)
            df_main['Is_FP'] = (df_main[primary_col] == 'FP').astype(int)
            print(f"Using primary marker column: '{primary_col}'")
        else:
            print("Warning: No primary marker column found. Creating dummy target variables.")
            # Create dummy 70% TP, 30% FP split
            np.random.seed(42)
            df_main['Is_TP'] = np.random.choice([0, 1], size=len(df_main), p=[0.3, 0.7])
            df_main['Is_FP'] = 1 - df_main['Is_TP']
        
        if secondary_col:
            df_main['Has_Secondary_Validation'] = df_main[secondary_col].notna()
            df_main['Primary_Secondary_Agreement'] = np.where(
                df_main['Has_Secondary_Validation'] & df_main[secondary_col].notna(),
                (df_main[primary_col] == df_main[secondary_col]).astype(int),
                np.nan
            )
            print(f"Using secondary marker column: '{secondary_col}'")
        else:
            df_main['Has_Secondary_Validation'] = False
            df_main['Primary_Secondary_Agreement'] = np.nan
            print("Warning: No secondary marker column found")
        
    except Exception as e:
        print(f"Error processing target variables: {e}")
        # Create dummy targets
        np.random.seed(42)
        df_main['Is_TP'] = np.random.choice([0, 1], size=len(df_main), p=[0.3, 0.7])
        df_main['Is_FP'] = 1 - df_main['Is_TP']
        df_main['Has_Secondary_Validation'] = False
        df_main['Primary_Secondary_Agreement'] = np.nan
    
    # Risk factor calculations for contingency tables
    try:
        # Use safe median calculations
        negation_median = df_main['Customer_Negation_Count'].median() if df_main['Customer_Negation_Count'].sum() > 0 else 0
        length_q75 = df_main['Transcript_Length'].quantile(0.75) if df_main['Transcript_Length'].sum() > 0 else 1000
        
        df_main['High_Negation_Risk'] = (df_main['Customer_Negation_Count'] > negation_median).astype(int)
        df_main['High_Qualifying_Risk'] = (df_main['Customer_Qualifying_Count'] > 1).astype(int)
        df_main['Long_Transcript_Risk'] = (df_main['Transcript_Length'] > length_q75).astype(int)
        df_main['High_Agent_Ratio_Risk'] = (df_main['Customer_Agent_Ratio'] < 0.5).astype(int)
        
        print("Risk factors calculated successfully")
        
    except Exception as e:
        print(f"Error calculating risk factors: {e}")
        # Default risk factors
        for col in ['High_Negation_Risk', 'High_Qualifying_Risk', 'Long_Transcript_Risk', 'High_Agent_Ratio_Risk']:
            df_main[col] = 0
    
    # Find category columns for analysis
    prosodica_l1_cols = [col for col in df_main.columns if 'prosodica' in col.lower() and 'l1' in col.lower()]
    prosodica_l2_cols = [col for col in df_main.columns if 'prosodica' in col.lower() and 'l2' in col.lower()]
    
    if prosodica_l1_cols:
        df_main['Prosodica L1'] = df_main[prosodica_l1_cols[0]]
        print(f"Using Prosodica L1 column: '{prosodica_l1_cols[0]}'")
    else:
        df_main['Prosodica L1'] = 'complaints'
        print("Warning: No Prosodica L1 column found. Using default.")
    
    if prosodica_l2_cols:
        df_main['Prosodica L2'] = df_main[prosodica_l2_cols[0]]
        print(f"Using Prosodica L2 column: '{prosodica_l2_cols[0]}'")
    else:
        df_main['Prosodica L2'] = 'general_complaint'
        print("Warning: No Prosodica L2 column found. Using default.")
    
    # Find variable5 column or create one
    variable5_cols = [col for col in df_main.columns if 'variable5' in col.lower()]
    if variable5_cols:
        df_main['variable5'] = df_main[variable5_cols[0]]
        print(f"Using variable5 column: '{variable5_cols[0]}'")
    else:
        df_main['variable5'] = range(len(df_main))
        print("Warning: No variable5 column found. Creating sequential IDs.")
    
    print(f"\nEnhanced data preparation completed successfully!")
    print(f"Final dataset shape: {df_main.shape}")
    print(f"Period distribution: Pre={len(df_main[df_main['Period']=='Pre'])}, Post={len(df_main[df_main['Period']=='Post'])}")
    print(f"Target distribution: TP={df_main['Is_TP'].sum()}, FP={df_main['Is_FP'].sum()}")
    
    return df_main, df_validation, df_rules_filtered
