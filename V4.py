# PQC Data Analysis - Past 3 Months (June, July, August 2025)
# This code answers the questions from your Excel requirements sheet

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

# Define date range - Past 3 months (June, July, August 2025)
start_date = '2025-06-01'
end_date = '2025-08-31'
print(f"Analyzing data from {start_date} to {end_date}")

# =============================================================================
# DATA LOADING - Load all necessary data
# =============================================================================

print("\nLoading data from tables...")

# Load case assignments data
query_assignments = f"""
SELECT * FROM pqc_case_assignments 
WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
AND PQC_ASSIGNMENT_DATE <= '{end_date}'
"""
df_assignments = pd.read_sql(query_assignments, conn)
print(f"Assignments loaded: {len(df_assignments)} records")

# Load case questions aggregated data
query_aggr = f"""
SELECT * FROM pqc_case_questions_aggr 
WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
AND PQC_ASSIGNMENT_DATE <= '{end_date}'
"""
df_aggr = pd.read_sql(query_aggr, conn)
print(f"Aggregated questions loaded: {len(df_aggr)} records")

# Load case closures data
query_closures = f"""
SELECT * FROM pqc_case_closures 
WHERE PQC_ASSIGNMENT_DATE >= '{start_date}' 
AND PQC_ASSIGNMENT_DATE <= '{end_date}'
"""
df_closures = pd.read_sql(query_closures, conn)
print(f"Closures loaded: {len(df_closures)} records")

# Clean column names - convert to lowercase and strip spaces
df_assignments.columns = df_assignments.columns.str.lower().str.strip()
df_aggr.columns = df_aggr.columns.str.lower().str.strip()
df_closures.columns = df_closures.columns.str.lower().str.strip()

# Debug: Print column names to verify
print("Assignments columns:", df_assignments.columns.tolist())
print("Aggr columns:", df_aggr.columns.tolist())
print("Closures columns:", df_closures.columns.tolist())

# Convert dates to datetime
df_assignments['pqc_assignment_date'] = pd.to_datetime(df_assignments['pqc_assignment_date'])
df_aggr['pqc_assignment_date'] = pd.to_datetime(df_aggr['pqc_assignment_date'])
df_closures['pqc_assignment_date'] = pd.to_datetime(df_closures['pqc_assignment_date'])

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

# Row 14: Number of cases worked daily
# Using is_spanish_case as proxy - assuming Spanish = Domestic, Non-Spanish = India
domestic_cases = df_assignments[df_assignments['is_spanish_case'] == 'Yes'].groupby('pqc_assignment_date').size()
india_cases = df_assignments[df_assignments['is_spanish_case'] != 'Yes'].groupby('pqc_assignment_date').size()

avg_domestic_daily = domestic_cases.mean() if len(domestic_cases) > 0 else 0
avg_india_daily = india_cases.mean() if len(india_cases) > 0 else 0

print(f"\n14. Number of cases worked daily:")
print(f"    - Domestic: {avg_domestic_daily:.1f} cases/day")
print(f"    - India: {avg_india_daily:.1f} cases/day")

# Row 15: Sendback rates by geography
domestic_keys = df_assignments[df_assignments['is_spanish_case'] == 'Yes']['case_key'].unique()
india_keys = df_assignments[df_assignments['is_spanish_case'] != 'Yes']['case_key'].unique()

domestic_aggr = df_aggr[df_aggr['case_key'].isin(domestic_keys)]
india_aggr = df_aggr[df_aggr['case_key'].isin(india_keys)]

domestic_sendback_rate = (domestic_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0
india_sendback_rate = (india_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0

print(f"\n15. Sendback rates by geography:")
print(f"    - Domestic: {domestic_sendback_rate:.1f}%")
print(f"    - India: {india_sendback_rate:.1f}%")

# Row 16: Correction rates by geography
domestic_correction_rate = (domestic_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(domestic_aggr) > 0 else 0
india_correction_rate = (india_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100 if len(india_aggr) > 0 else 0

print(f"\n16. Correction rates by geography:")
print(f"    - Domestic: {domestic_correction_rate:.1f}%")
print(f"    - India: {india_correction_rate:.1f}%")

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
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print(f"\nData Period: {start_date} to {end_date}")
print(f"Total Assignments: {len(df_assignments):,}")
print(f"Total Cases Processed: {len(df_aggr):,}")
print(f"Total Cases Closed: {len(df_closures):,}")
print(f"Total PQC Managers: {total_managers}")
print(f"Total CCR Managers: {len(ccr_manager_defects)}")

print(f"\nOverall Metrics:")
print(f"- Overall Defect Rate: {(df_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100:.1f}%")
print(f"- Overall Sendback Rate: {(df_aggr['case_key_lvl_sendback_flag'] == 'True').mean() * 100:.1f}%")
print(f"- Overall Approval Rate: {100 - (df_aggr['case_key_lvl_defect_flag'] == 'True').mean() * 100:.1f}%")

# Close database connection
conn.close()
print(f"\nAnalysis complete. Database connection closed.")
