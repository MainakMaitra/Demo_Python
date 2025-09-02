# PQC Data Analysis - Past 3 Months (June, July, August 2025) - MONTHLY BREAKDOWN
# This code answers the questions from your Excel requirements sheet with monthly splits

import warnings
import oracledb
import configparser
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Database Connection Setup
def connect_to_database():
    """Connect to Oracle database using existing config"""
    config = configparser.ConfigParser(interpolation=None)
    config.read('/var/run/secrets/user_credentials/PQC_CONFIG')
    
    # Get credentials
    hostname = config.get('pub_conn', 'hostname')
    port = config.get('pub_conn', 'port')
    servicename = config.get('pub_conn', 'servicename')
    username = config.get('pub_conn', 'username')
    password = config.get('pub_conn', 'pwd')
    
    # Connect
    conn_str = f"{hostname}:{port}/{servicename}"
    connection = oracledb.connect(user=username, password=password, dsn=conn_str)
    return connection

# Connect to database
print("Connecting to database...")
conn = connect_to_database()
print("Connected successfully!")

# Define date ranges for each month
date_ranges = {
    'June': ('2025-06-01', '2025-06-30'),
    'July': ('2025-07-01', '2025-07-31'), 
    'August': ('2025-08-01', '2025-08-31')
}

print(f"Analyzing data by month: {list(date_ranges.keys())}")

# =============================================================================
# DATA LOADING BY MONTH
# =============================================================================

def load_monthly_data(month_name, start_date, end_date):
    """Load data for a specific month"""
    
    # Load case assignments data
    query_assignments = f"""
    SELECT * FROM pqc_case_assignments 
    WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
    AND PQC_ASSIGNMENT_DATE <= '{end_date}'
    """
    df_assignments = pd.read_sql(query_assignments, conn)
    
    # Load case questions aggregated data
    query_aggr = f"""
    SELECT * FROM pqc_case_questions_aggr 
    WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
    AND PQC_ASSIGNMENT_DATE <= '{end_date}'
    """
    df_aggr = pd.read_sql(query_aggr, conn)
    
    # Load case closures data
    query_closures = f"""
    SELECT * FROM pqc_case_closures 
    WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
    AND PQC_ASSIGNMENT_DATE <= '{end_date}'
    """
    df_closures = pd.read_sql(query_closures, conn)
    
    # Clean column names
    df_assignments.columns = df_assignments.columns.str.lower().str.strip()
    df_aggr.columns = df_aggr.columns.str.lower().str.strip()
    df_closures.columns = df_closures.columns.str.lower().str.strip()
    
    # Convert dates to datetime
    df_assignments['pqc_assignment_date'] = pd.to_datetime(df_assignments['pqc_assignment_date'])
    df_aggr['pqc_assignment_date'] = pd.to_datetime(df_aggr['pqc_assignment_date'])
    df_closures['pqc_assignment_date'] = pd.to_datetime(df_closures['pqc_assignment_date'])
    
    print(f"{month_name} - Assignments: {len(df_assignments)}, Aggregated: {len(df_aggr)}, Closures: {len(df_closures)}")
    
    return df_assignments, df_aggr, df_closures

# Load data for all months
monthly_data = {}
for month_name, (start_date, end_date) in date_ranges.items():
    monthly_data[month_name] = load_monthly_data(month_name, start_date, end_date)

# =============================================================================
# MONTHLY ANALYSIS FUNCTION
# =============================================================================

def analyze_month(month_name, df_assignments, df_aggr, df_closures):
    """Analyze data for a specific month"""
    
    results = {'month': month_name}
    
    # Calculate total days in month
    start_date, end_date = date_ranges[month_name]
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    
    # =============================================================================
    # QUALITY REVIEW STATS
    # =============================================================================
    
    # Row 4: Average number of cases worked per day per PQC manager
    if len(df_assignments) > 0:
        daily_cases = df_assignments.groupby(['pqc_sso', 'pqc_assignment_date']).size().reset_index(name='daily_cases')
        avg_cases_per_day_per_manager = daily_cases.groupby('pqc_sso')['daily_cases'].mean()
        overall_avg_cases_per_day = avg_cases_per_day_per_manager.mean()
        total_managers = len(avg_cases_per_day_per_manager)
        
        # Workload distribution
        managers_less_than_5 = (avg_cases_per_day_per_manager < 5).sum()
        managers_5_to_8 = ((avg_cases_per_day_per_manager >= 5) & (avg_cases_per_day_per_manager <= 8)).sum()
        managers_9_plus = (avg_cases_per_day_per_manager > 8).sum()
        
        percent_less_than_5 = (managers_less_than_5 / total_managers) * 100 if total_managers > 0 else 0
        percent_5_to_8 = (managers_5_to_8 / total_managers) * 100 if total_managers > 0 else 0
        percent_9_plus = (managers_9_plus / total_managers) * 100 if total_managers > 0 else 0
    else:
        overall_avg_cases_per_day = 0
        total_managers = 0
        percent_less_than_5 = percent_5_to_8 = percent_9_plus = 0
    
    results['avg_cases_per_day_per_manager'] = overall_avg_cases_per_day
    results['percent_less_than_5'] = percent_less_than_5
    results['percent_5_to_8'] = percent_5_to_8
    results['percent_9_plus'] = percent_9_plus
    results['total_managers'] = total_managers
    
    # Performance rates by workload groups
    if len(df_aggr) > 0 and len(df_assignments) > 0:
        # Calculate sendback and correction rates for workload groups
        manager_sendbacks = df_aggr.groupby('pqc_sso').agg({
            'case_key_lvl_sendback_flag': lambda x: (x == 'True').sum(),
            'case_key': 'count'
        }).reset_index()
        manager_sendbacks['sendback_rate'] = manager_sendbacks['case_key_lvl_sendback_flag'] / manager_sendbacks['case_key'] * 100
        
        manager_corrections = df_aggr.groupby('pqc_sso').agg({
            'case_key_lvl_defect_flag': lambda x: (x == 'True').sum(),
            'case_key': 'count'
        }).reset_index()
        manager_corrections['correction_rate'] = manager_corrections['case_key_lvl_defect_flag'] / manager_corrections['case_key'] * 100
        
        # Merge with workload data
        manager_stats = avg_cases_per_day_per_manager.reset_index()
        manager_stats.columns = ['pqc_sso', 'avg_daily_cases']
        manager_stats = manager_stats.merge(manager_sendbacks[['pqc_sso', 'sendback_rate']], on='pqc_sso', how='left')
        manager_stats = manager_stats.merge(manager_corrections[['pqc_sso', 'correction_rate']], on='pqc_sso', how='left')
        
        # Calculate rates by groups
        less_than_5_sendback = manager_stats[manager_stats['avg_daily_cases'] < 5]['sendback_rate'].mean() if len(manager_stats[manager_stats['avg_daily_cases'] < 5]) > 0 else 0
        five_to_8_sendback = manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]['sendback_rate'].mean() if len(manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]) > 0 else 0
        nine_plus_sendback = manager_stats[manager_stats['avg_daily_cases'] > 8]['sendback_rate'].mean() if len(manager_stats[manager_stats['avg_daily_cases'] > 8]) > 0 else 0
        
        less_than_5_correction = manager_stats[manager_stats['avg_daily_cases'] < 5]['correction_rate'].mean() if len(manager_stats[manager_stats['avg_daily_cases'] < 5]) > 0 else 0
        five_to_8_correction = manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]['correction_rate'].mean() if len(manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]) > 0 else 0
        nine_plus_correction = manager_stats[manager_stats['avg_daily_cases'] > 8]['correction_rate'].mean() if len(manager_stats[manager_stats['avg_daily_cases'] > 8]) > 0 else 0
    else:
        less_than_5_sendback = five_to_8_sendback = nine_plus_sendback = 0
        less_than_5_correction = five_to_8_correction = nine_plus_correction = 0
    
    results.update({
        'less_than_5_sendback': less_than_5_sendback,
        'five_to_8_sendback': five_to_8_sendback,
        'nine_plus_sendback': nine_plus_sendback,
        'less_than_5_correction': less_than_5_correction,
        'five_to_8_correction': five_to_8_correction,
        'nine_plus_correction': nine_plus_correction,
        'less_than_5_approval': 100 - less_than_5_correction,
        'five_to_8_approval': 100 - five_to_8_correction,
        'nine_plus_approval': 100 - nine_plus_correction
    })
    
    # =============================================================================
    # GEOGRAPHY ANALYSIS
    # =============================================================================
    
    india_managers = ['Cicily Raj', 'Savithry Kandregula', 'Ayesha Ahmed', 'Pavani Shelly', 'Rupa Talla', 'Rupa T']
    
    if len(df_assignments) > 0:
        india_assignments = df_assignments[df_assignments['pqc_name'].isin(india_managers)]
        domestic_assignments = df_assignments[~df_assignments['pqc_name'].isin(india_managers)]
        
        # Daily case counts
        india_daily_cases = india_assignments.groupby('pqc_assignment_date').size()
        domestic_daily_cases = domestic_assignments.groupby('pqc_assignment_date').size()
        
        avg_india_daily = india_daily_cases.mean() if len(india_daily_cases) > 0 else 0
        avg_domestic_daily = domestic_daily_cases.mean() if len(domestic_daily_cases) > 0 else 0
        
        # Geography performance rates
        if len(df_aggr) > 0:
            india_case_keys = india_assignments['case_key'].unique()
            domestic_case_keys = domestic_assignments['case_key'].unique()
            
            india_aggr = df_aggr[df_aggr['case_key'].isin(india_case_keys)]
            domestic_aggr = df_aggr[df_aggr['case_key'].isin(domestic_case_keys)]
            
            india_sendback_rate = (india_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0
            domestic_sendback_rate = (domestic_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0
            
            india_correction_rate = (india_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0
            domestic_correction_rate = (domestic_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0
        else:
            india_sendback_rate = domestic_sendback_rate = 0
            india_correction_rate = domestic_correction_rate = 0
    else:
        avg_india_daily = avg_domestic_daily = 0
        india_sendback_rate = domestic_sendback_rate = 0
        india_correction_rate = domestic_correction_rate = 0
    
    results.update({
        'avg_india_daily': avg_india_daily,
        'avg_domestic_daily': avg_domestic_daily,
        'india_sendback_rate': india_sendback_rate,
        'domestic_sendback_rate': domestic_sendback_rate,
        'india_correction_rate': india_correction_rate,
        'domestic_correction_rate': domestic_correction_rate
    })
    
    # =============================================================================
    # GENERAL INFORMATION
    # =============================================================================
    
    total_cases = len(df_assignments)
    avg_outflow_per_day = total_cases / total_days if total_days > 0 else 0
    avg_inflow_per_day = avg_outflow_per_day  # Steady state assumption
    
    if len(df_aggr) > 0:
        cases_with_corrections = (df_aggr['case_key_lvl_defect_flag'] == 'True').sum()
        cases_sent_back = (df_aggr['case_key_lvl_sendback_flag'] == 'True').sum()
        cases_fixed_only = ((df_aggr['case_key_lvl_defect_flag'] == 'True') & 
                           (df_aggr['case_key_lvl_sendback_flag'] != 'True')).sum()
        
        rate_fixed_only = (cases_fixed_only / len(df_aggr)) * 100
        rate_sent_back = (cases_sent_back / len(df_aggr)) * 100
    else:
        cases_with_corrections = cases_sent_back = cases_fixed_only = 0
        rate_fixed_only = rate_sent_back = 0
    
    results.update({
        'avg_outflow_per_day': avg_outflow_per_day,
        'avg_inflow_per_day': avg_inflow_per_day,
        'cases_with_corrections': cases_with_corrections,
        'cases_sent_back': cases_sent_back,
        'rate_fixed_only': rate_fixed_only,
        'cases_requiring_pqc_fix': cases_with_corrections,
        'rate_sent_back': rate_sent_back
    })
    
    return results

# =============================================================================
# ANALYZE ALL MONTHS
# =============================================================================

print("\n" + "="*80)
print("ANALYZING DATA BY MONTH")
print("="*80)

monthly_results = {}
for month_name, (df_assignments, df_aggr, df_closures) in monthly_data.items():
    monthly_results[month_name] = analyze_month(month_name, df_assignments, df_aggr, df_closures)

# =============================================================================
# SECTION 1: QUALITY REVIEW STATS (Rows 3-7)
# =============================================================================

print("\n" + "="*60)
print("QUALITY REVIEW STATS - FOR PAST 3 MONTHS")
print("="*60)

# Row 4: Average number of cases worked per day per PQC manager
daily_cases = df_assignments.groupby(['pqc_sso', 'pqc_assignment_date']).size().reset_index(name='daily_cases')
avg_cases_per_day_per_manager = daily_cases.groupby('pqc_sso')['daily_cases'].mean()
overall_avg_cases_per_day = avg_cases_per_day_per_manager.mean()

print(f"\n4. Average number of cases worked per day per PQC manager: {overall_avg_cases_per_day:.2f}")

# Row 5: Percent of PQC managers working less than 5 per day on average
managers_less_than_5 = (avg_cases_per_day_per_manager < 5).sum()
total_managers = len(avg_cases_per_day_per_manager)
percent_less_than_5 = (managers_less_than_5 / total_managers) * 100

print(f"5. Percent of PQC managers working less than 5 per day on average: {percent_less_than_5:.1f}%")

# Row 6: Percent of PQC managers working 5 to 8 per day on average
managers_5_to_8 = ((avg_cases_per_day_per_manager >= 5) & (avg_cases_per_day_per_manager <= 8)).sum()
percent_5_to_8 = (managers_5_to_8 / total_managers) * 100

print(f"6. Percent of PQC managers working 5 to 8 per day on average: {percent_5_to_8:.1f}%")

# Row 7: Percent of PQC managers working 9 or more per day on average
managers_9_plus = (avg_cases_per_day_per_manager > 8).sum()
percent_9_plus = (managers_9_plus / total_managers) * 100

print(f"7. Percent of PQC managers working 9 or more per day on average: {percent_9_plus:.1f}%")

# Row 9: Sendback rates by each group above (rows 5-7)
print(f"\n9. Sendback rates by each group above:")

# Calculate sendback rates for each manager
manager_sendbacks = df_aggr.groupby('pqc_sso').agg({
    'case_key_lvl_sendback_flag': lambda x: (x == 'True').sum(),
    'case_key': 'count'
}).reset_index()
manager_sendbacks['sendback_rate'] = manager_sendbacks['case_key_lvl_sendback_flag'] / manager_sendbacks['case_key'] * 100

# Merge with average cases per day
manager_stats = avg_cases_per_day_per_manager.reset_index()
manager_stats.columns = ['pqc_sso', 'avg_daily_cases']
manager_stats = manager_stats.merge(manager_sendbacks, on='pqc_sso', how='left')

# Calculate sendback rates by groups
less_than_5_sendback = manager_stats[manager_stats['avg_daily_cases'] < 5]['sendback_rate'].mean()
five_to_8_sendback = manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]['sendback_rate'].mean()
nine_plus_sendback = manager_stats[manager_stats['avg_daily_cases'] > 8]['sendback_rate'].mean()

print(f"   - Less than 5 cases/day group: {less_than_5_sendback:.1f}% sendback rate")
print(f"   - 5 to 8 cases/day group: {five_to_8_sendback:.1f}% sendback rate") 
print(f"   - 9+ cases/day group: {nine_plus_sendback:.1f}% sendback rate")

# Row 10: Correction rates by each group above
print(f"\n10. Correction rates by each group above:")

# Calculate correction rates (defect rates)
manager_corrections = df_aggr.groupby('pqc_sso').agg({
    'case_key_lvl_defect_flag': lambda x: (x == 'True').sum(),
    'case_key': 'count'
}).reset_index()
manager_corrections['correction_rate'] = manager_corrections['case_key_lvl_defect_flag'] / manager_corrections['case_key'] * 100

# Merge with manager stats
manager_stats = manager_stats.merge(manager_corrections[['pqc_sso', 'correction_rate']], on='pqc_sso', how='left')

# Calculate correction rates by groups
less_than_5_correction = manager_stats[manager_stats['avg_daily_cases'] < 5]['correction_rate'].mean()
five_to_8_correction = manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]['correction_rate'].mean()
nine_plus_correction = manager_stats[manager_stats['avg_daily_cases'] > 8]['correction_rate'].mean()

print(f"   - Less than 5 cases/day group: {less_than_5_correction:.1f}% correction rate")
print(f"   - 5 to 8 cases/day group: {five_to_8_correction:.1f}% correction rate")
print(f"   - 9+ cases/day group: {nine_plus_correction:.1f}% correction rate")

# Row 11: Approval rates by each group above
print(f"\n11. Approval rates by each group above:")

# Calculate approval rates (opposite of defect rates)
manager_stats['approval_rate'] = 100 - manager_stats['correction_rate']

less_than_5_approval = manager_stats[manager_stats['avg_daily_cases'] < 5]['approval_rate'].mean()
five_to_8_approval = manager_stats[(manager_stats['avg_daily_cases'] >= 5) & (manager_stats['avg_daily_cases'] <= 8)]['approval_rate'].mean()
nine_plus_approval = manager_stats[manager_stats['avg_daily_cases'] > 8]['approval_rate'].mean()

print(f"   - Less than 5 cases/day group: {less_than_5_approval:.1f}% approval rate")
print(f"   - 5 to 8 cases/day group: {five_to_8_approval:.1f}% approval rate")
print(f"   - 9+ cases/day group: {nine_plus_approval:.1f}% approval rate")

# =============================================================================
# SECTION 2: GEOGRAPHY ANALYSIS (Rows 13-16)
# =============================================================================

print("\n" + "="*60)
print("GEOGRAPHY (INDIA VS DOMESTIC)")
print("="*60)

# Define India and Domestic managers based on manager feedback
india_managers = ['Cicily', 'Savithry', 'Ayesha', 'Pavani', 'Rupa']

# Row 14: Number of cases worked daily
# Using manager names to classify geography
india_assignments = df_assignments[df_assignments['pqc_name'].isin(india_managers)]
domestic_assignments = df_assignments[~df_assignments['pqc_name'].isin(india_managers)]

# Calculate daily case counts by geography
india_daily_cases = india_assignments.groupby('pqc_assignment_date').size()
domestic_daily_cases = domestic_assignments.groupby('pqc_assignment_date').size()

avg_india_daily = india_daily_cases.mean() if len(india_daily_cases) > 0 else 0
avg_domestic_daily = domestic_daily_cases.mean() if len(domestic_daily_cases) > 0 else 0

print(f"\n14. Number of cases worked daily:")
print(f"    - India: {avg_india_daily:.1f} cases/day")
print(f"    - Domestic: {avg_domestic_daily:.1f} cases/day")

# Row 15: Sendback rates by geography
# Get case keys for each geography
india_case_keys = india_assignments['case_key'].unique()
domestic_case_keys = domestic_assignments['case_key'].unique()

# Filter aggregated data by geography
india_aggr = df_aggr[df_aggr['case_key'].isin(india_case_keys)]
domestic_aggr = df_aggr[df_aggr['case_key'].isin(domestic_case_keys)]

# Calculate sendback rates
india_sendback_rate = (india_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0
domestic_sendback_rate = (domestic_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0

print(f"\n15. Sendback rates by geography:")
print(f"    - India: {india_sendback_rate:.1f}%")
print(f"    - Domestic: {domestic_sendback_rate:.1f}%")

# Row 16: Correction rates by geography
india_correction_rate = (india_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0
domestic_correction_rate = (domestic_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0

print(f"\n16. Correction rates by geography:")
print(f"    - India: {india_correction_rate:.1f}%")
print(f"    - Domestic: {domestic_correction_rate:.1f}%")

# Additional insight: Show manager distribution
print(f"\nManager Distribution:")
print(f"    - India Managers: {len(india_managers)} ({', '.join(india_managers)})")
print(f"    - Domestic Managers: {len(domestic_assignments['pqc_name'].unique())} total")
print(f"    - India Cases: {len(india_assignments):,}")
print(f"    - Domestic Cases: {len(domestic_assignments):,}")

# =============================================================================
# SECTION 3: GENERAL INFORMATION (Rows 18-27)
# =============================================================================

print("\n" + "="*60)
print("GENERAL INFORMATION")
print("="*60)

# Row 19: Average number of cases outflow per day
total_cases = len(df_assignments)
total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
avg_outflow_per_day = total_cases / total_days

print(f"\n19. Average number of cases outflow per day: {avg_outflow_per_day:.1f}")

# Row 20: Average number inflowing each day (assuming same as outflow for steady state)
avg_inflow_per_day = avg_outflow_per_day  # Assumption: steady state
print(f"20. Average number inflowing each day: {avg_inflow_per_day:.1f}")

# Row 22: Number of cases fixed
cases_with_corrections = (df_aggr['case_key_lvl_defect_flag'] == 'True').sum()
print(f"\n22. Number of cases fixed: {cases_with_corrections}")

# Row 23: Number of cases sent back
cases_sent_back = (df_aggr['case_key_lvl_sendback_flag'] == 'True').sum()
print(f"23. Number of cases sent back: {cases_sent_back}")

# Row 25: Rate of cases fixed only
total_processed_cases = len(df_aggr)
cases_fixed_only = ((df_aggr['case_key_lvl_defect_flag'] == 'True') & 
                   (df_aggr['case_key_lvl_sendback_flag'] != 'True')).sum()
rate_fixed_only = (cases_fixed_only / total_processed_cases) * 100

print(f"\n25. Rate of cases fixed only: {rate_fixed_only:.1f}%")

# Row 26: Any case where PQC has to fix the response
cases_requiring_pqc_fix = (df_aggr['case_key_lvl_defect_flag'] == 'True').sum()
print(f"26. Cases where PQC has to fix the response: {cases_requiring_pqc_fix}")

# Row 27: Rate of cases sent back
rate_sent_back = (cases_sent_back / total_processed_cases) * 100
print(f"27. Rate of cases sent back: {rate_sent_back:.1f}%")

# =============================================================================
# SECTION 4: OPERATIONS (Rows 30-31)
# =============================================================================

print("\n" + "="*60)
print("OPERATIONS")
print("="*60)

# Note: pqc_case_questions_aggr doesn't have ccr_investigator_sso
# We need to join with assignments table to get CCR manager info

# Create a mapping of case_key to ccr_investigator_sso from assignments
ccr_mapping = df_assignments[['case_key', 'ccr_investigator_sso']].drop_duplicates()

# Merge aggr data with CCR manager info
df_aggr_with_ccr = df_aggr.merge(ccr_mapping, on='case_key', how='left')

# Row 30: Defect rate by Operations CCR Manager
ccr_manager_defects = df_aggr_with_ccr.groupby('ccr_investigator_sso').agg({
    'case_key_lvl_defect_flag': lambda x: (x == 'True').mean() * 100,
    'case_key': 'count'
}).reset_index()
ccr_manager_defects.columns = ['ccr_manager', 'defect_rate', 'total_cases']
ccr_manager_defects = ccr_manager_defects[ccr_manager_defects['total_cases'] >= 5]  # Filter for managers with at least 5 cases

print(f"\n30. Defect rate by Operations CCR Manager:")
for _, row in ccr_manager_defects.iterrows():
    print(f"    - {row['ccr_manager']}: {row['defect_rate']:.1f}% ({row['total_cases']} cases)")

# Row 31: Sendback rates by Operations CCR Manager
ccr_manager_sendbacks = df_aggr_with_ccr.groupby('ccr_investigator_sso').agg({
    'case_key_lvl_sendback_flag': lambda x: (x == 'True').mean() * 100,
    'case_key': 'count'
}).reset_index()
ccr_manager_sendbacks.columns = ['ccr_manager', 'sendback_rate', 'total_cases']
ccr_manager_sendbacks = ccr_manager_sendbacks[ccr_manager_sendbacks['total_cases'] >= 5]  # Filter for managers with at least 5 cases

print(f"\n31. Sendback rates by Operations CCR Manager:")
for _, row in ccr_manager_sendbacks.iterrows():
    print(f"    - {row['ccr_manager']}: {row['sendback_rate']:.1f}% ({row['total_cases']} cases)")

# =============================================================================
# MONTHLY RESULTS DISPLAY - ALL ANSWERS BY MONTH
# =============================================================================

print("\n" + "="*100)
print("PQC DATA ANALYSIS - MONTHLY BREAKDOWN RESULTS")
print("Data Period: June, July, August 2025")
print("="*100)

# Create comparison table for easy viewing
months = ['June', 'July', 'August']

print("\n📊 QUALITY REVIEW STATS - MONTHLY COMPARISON")
print("-" * 80)
print(f"{'Metric':<50} {'June':<15} {'July':<15} {'August':<15}")
print("-" * 80)

# Row 4: Average cases per day per PQC manager
print(f"{'4. Avg cases/day/manager:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['avg_cases_per_day_per_manager']:.2f}"[:14].ljust(15), end="")
print()

# Rows 5-7: Manager workload distribution
print(f"{'5. Managers <5 cases/day (%):':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['percent_less_than_5']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'6. Managers 5-8 cases/day (%):':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['percent_5_to_8']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'7. Managers 9+ cases/day (%):':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['percent_9_plus']:.1f}%"[:14].ljust(15), end="")
print()

# Rows 9-11: Performance rates by workload groups
print(f"\n{'SENDBACK RATES BY WORKLOAD GROUP:':<50}")
print(f"{'9. <5 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['less_than_5_sendback']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'   5-8 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['five_to_8_sendback']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'   9+ cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['nine_plus_sendback']:.1f}%"[:14].ljust(15), end="")
print()

print(f"\n{'CORRECTION RATES BY WORKLOAD GROUP:':<50}")
print(f"{'10. <5 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['less_than_5_correction']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    5-8 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['five_to_8_correction']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    9+ cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['nine_plus_correction']:.1f}%"[:14].ljust(15), end="")
print()

print(f"\n{'APPROVAL RATES BY WORKLOAD GROUP:':<50}")
print(f"{'11. <5 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['less_than_5_approval']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    5-8 cases/day group:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['five_to_8_approval']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    9+ cases/day: {nine_plus_approval:.1f}%"[:14].ljust(15), end="")
print()

print("\n🌍 GEOGRAPHY (INDIA VS DOMESTIC) - MONTHLY COMPARISON")
print("-" * 80)
print(f"{'Metric':<50} {'June':<15} {'July':<15} {'August':<15}")
print("-" * 80)

print(f"{'14. India cases/day:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['avg_india_daily']:.1f}"[:14].ljust(15), end="")
print()

print(f"{'    Domestic cases/day:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['avg_domestic_daily']:.1f}"[:14].ljust(15), end="")
print()

print(f"{'15. India sendback rate:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['india_sendback_rate']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    Domestic sendback rate:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['domestic_sendback_rate']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'16. India correction rate:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['india_correction_rate']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'    Domestic correction rate:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['domestic_correction_rate']:.1f}%"[:14].ljust(15), end="")
print()

print("\n📈 GENERAL INFORMATION - MONTHLY COMPARISON")
print("-" * 80)
print(f"{'Metric':<50} {'June':<15} {'July':<15} {'August':<15}")
print("-" * 80)

print(f"{'19. Avg outflow/day:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['avg_outflow_per_day']:.1f}"[:14].ljust(15), end="")
print()

print(f"{'20. Avg inflow/day:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['avg_inflow_per_day']:.1f}"[:14].ljust(15), end="")
print()

print(f"{'22. Cases fixed:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['cases_with_corrections']:,}"[:14].ljust(15), end="")
print()

print(f"{'23. Cases sent back:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['cases_sent_back']:,}"[:14].ljust(15), end="")
print()

print(f"{'25. Rate of cases fixed only:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['rate_fixed_only']:.1f}%"[:14].ljust(15), end="")
print()

print(f"{'26. Cases requiring PQC fix:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['cases_requiring_pqc_fix']:,}"[:14].ljust(15), end="")
print()

print(f"{'27. Rate of cases sent back:':<50} ", end="")
for month in months:
    print(f"{monthly_results[month]['rate_sent_back']:.1f}%"[:14].ljust(15), end="")
print()

print("\n📋 MONTHLY SUMMARY STATISTICS")
print("-" * 80)
for month in months:
    result = monthly_results[month]
    print(f"\n{month.upper()} SUMMARY:")
    print(f"  • Total PQC Managers: {result['total_managers']}")
    print(f"  • Total Cases: {result['cases_with_corrections'] + result['cases_sent_back']:,}")
    print(f"  • India vs Domestic: {result['avg_india_daily']:.1f} vs {result['avg_domestic_daily']:.1f} cases/day")

print("\n" + "="*100)
print("MONTHLY ANALYSIS COMPLETE")
print("="*100)

# Close database connection
conn.close()
print("Database connection closed.")
: {nine_plus_approval:.1f}%")

print("\n🌍 GEOGRAPHY (INDIA VS DOMESTIC)")
print("-" * 50)
print(f"14. Daily case volume:")
print(f"   • India: {avg_india_daily:.1f} cases/day")
print(f"   • Domestic: {avg_domestic_daily:.1f} cases/day")
print(f"\n15. Sendback rates:")
print(f"   • India: {india_sendback_rate:.1f}%")
print(f"   • Domestic: {domestic_sendback_rate:.1f}%")
print(f"\n16. Correction rates:")
print(f"   • India: {india_correction_rate:.1f}%")
print(f"   • Domestic: {domestic_correction_rate:.1f}%")

print("\n📈 GENERAL INFORMATION")
print("-" * 50)
print(f"19. Average outflow per day: {avg_outflow_per_day:.1f} cases")
print(f"20. Average inflow per day: {avg_inflow_per_day:.1f} cases")
print(f"22. Cases fixed by PQC: {cases_with_corrections:,}")
print(f"23. Cases sent back: {cases_sent_back:,}")
print(f"25. Rate of cases fixed only: {rate_fixed_only:.1f}%")
print(f"26. Cases requiring PQC response fix: {cases_requiring_pqc_fix:,}")
print(f"27. Rate of cases sent back: {rate_sent_back:.1f}%")

print("\n👥 OPERATIONS - TOP CCR MANAGERS")
print("-" * 50)
print("30. Defect rates by CCR Manager (min 5 cases):")
top_defect_managers = ccr_manager_defects.nlargest(5, 'defect_rate')
for _, row in top_defect_managers.iterrows():
    print(f"   • {row['ccr_manager']}: {row['defect_rate']:.1f}% ({int(row['total_cases'])} cases)")

print(f"\n31. Sendback rates by CCR Manager (min 5 cases):")
top_sendback_managers = ccr_manager_sendbacks.nlargest(5, 'sendback_rate')
for _, row in top_sendback_managers.iterrows():
    print(f"   • {row['ccr_manager']}: {row['sendback_rate']:.1f}% ({int(row['total_cases'])} cases)")

print("\n📋 KEY METRICS SUMMARY")
print("-" * 50)
overall_defect_rate = (df_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100
overall_sendback_rate = (df_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100
overall_approval_rate = 100 - overall_defect_rate

print(f"• Total cases analyzed: {len(df_aggr):,}")
print(f"• Total PQC managers: {total_managers}")
print(f"• Overall defect rate: {overall_defect_rate:.1f}%")
print(f"• Overall sendback rate: {overall_sendback_rate:.1f}%")
print(f"• Overall approval rate: {overall_approval_rate:.1f}%")
print(f"• India managers: {len(india_managers)} ({', '.join(india_managers)})")
print(f"• India vs Domestic case split: {len(india_assignments):,} vs {len(domestic_assignments):,}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE ✅")
print("="*80)

# Close database connection
conn.close()
print(f"\nAnalysis complete. Database connection closed.")
