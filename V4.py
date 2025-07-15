PQC Data Ingestion Pipeline - Detailed Code Explanation
Overview
This Python script implements a comprehensive data ingestion pipeline for Process Quality Control (PQC) data. It processes Excel files from SharePoint Online (SPO), transforms the data, and loads it into an Oracle database for analysis and reporting.
Memory Map & System Architecture
┌─────────────────────────────────────────────────────────────────┐
│                     PQC DATA INGESTION PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   SharePoint    │    │   Local Data    │    │   Oracle     │ │
│  │     Online      │───▶│   Processing    │───▶│   Database   │ │
│  │                 │    │                 │    │     (PUB)    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                        DATA FLOW STAGES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. CONFIGURATION & AUTHENTICATION                             │
│     ├── Read config file                                       │
│     ├── Setup logging                                          │
│     └── Connect to SharePoint                                  │
│                                                                 │
│  2. DATA EXTRACTION                                            │
│     ├── Download old format files (RQC_*.xlsx)                │
│     ├── Download new format files (PQC_*.xlsx)                │
│     └── Extract SPO list data (defects & sendbacks)           │
│                                                                 │
│  3. DATA TRANSFORMATION                                        │
│     ├── Process old format Excel files                        │
│     ├── Process new format Excel files                        │
│     ├── Merge datasets                                         │
│     └── Create aggregations                                    │
│                                                                 │
│  4. DATA LOADING                                               │
│     ├── Backup existing tables                                │
│     ├── Truncate target tables                                │
│     └── Insert transformed data                               │
│                                                                 │
│  5. CLEANUP & MONITORING                                       │
│     ├── Remove local files                                     │
│     ├── Send success/failure notifications                     │
│     └── Log completion status                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
Detailed Code Breakdown
1. Imports and Configuration Setup
pythonimport io, warnings, oracledb, configparser, logging, os, sys, pytz, datetime, time, traceback, smtplib, pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from office365.sharepoint.client_context import ClientContext, ClientCredential
Purpose: The script begins by importing essential libraries for:

Database connectivity: oracledb for Oracle database operations
SharePoint integration: office365 for accessing SharePoint Online
Data processing: pandas for data manipulation and analysis
File operations: Standard Python libraries for file handling
Communication: Email libraries for notification sending
Logging: Comprehensive logging for monitoring and debugging

2. Logging Configuration
pythonloggingConfig = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        }
    },
    "handlers": {
        "console": {"level": "INFO", "class": "logging.StreamHandler"},
        "error_file": {"class": "logging.handlers.RotatingFileHandler"},
        "access_file": {"class": "logging.handlers.TimedRotatingFileHandler"}
    }
}
Purpose: Establishes a robust logging system with:

Console output: Real-time monitoring during execution
Error file logging: Rotating log files for error tracking (10MB max, 10 backups)
Access file logging: Daily rotating logs for audit trails
Standardized formatting: Consistent timestamp and module identification

3. Core Utility Functions
Email Notification System
pythondef send_email(body_str, recipient_list, senders_email, subject):
    msg_body = MIMEText(body_str, 'plain')
    msg = MIMEMultipart()
    msg.attach(msg_body)
    msg['Subject'] = subject
    msg['From'] = senders_email
    server = smtplib.SMTP('relay.app.syfbank.com:25')
    server.sendmail(senders_email, recipient_list, msg.as_string())
Purpose: Automated notification system for:

Job completion alerts: Success/failure notifications
Error reporting: Detailed error information to administrators
System monitoring: Operational status updates

Timezone Conversion
pythondef convert_utc_str_to_est(utc_ts_str):
    dt_obj = datetime.datetime.fromisoformat(utc_ts_str)
    est = pytz.timezone('US/Eastern')
    res_dt_obj = dt_obj.astimezone(est)
    return res_dt_obj
Purpose: Handles timezone conversions for SharePoint timestamps from UTC to Eastern Time, ensuring consistent time representation across the system.
4. SharePoint Integration Functions
Large List Query Optimization
pythondef query_large_list(target_list):
    paged_items = target_list.items.paged(500, page_loaded=print_progress).get().execute_query()
    data = []
    for index, item in enumerate(paged_items):
        if (index + 1) % 5000 == 0:
            logger.info(f"{index + 1} records read")
        data.append(item.properties)
    return data
Purpose: Efficiently processes large SharePoint lists by:

Pagination: Retrieving data in chunks of 500 items
Progress tracking: Logging every 5000 records processed
Memory management: Preventing memory overflow with large datasets
Performance optimization: Reducing API call overhead

File Download Management
pythondef download_spo_pqc_files(context, file_path, file_fmt_type):
    # Process old format files (RQC_*)
    if file_fmt_type == "old_fmt":
        pqc_data_folder = context.web.get_folder_by_server_relative_url(file_path)
        old_files = pqc_data_folder.expand(["Files", "Folders"]).get().execute_query()
        for file in old_files.files:
            if "RQC_" in str(file):
                files_list.append(file)
    
    # Process new format files (PQC_*)
    elif file_fmt_type == "new_fmt":
        for new_folder in file_path:
            files_list = []
            pqc_data_folder = context.web.get_folder_by_server_relative_url(new_folder)
            new_files = pqc_data_folder.expand(["Files", "Folders"]).get().execute_query()
            for file in new_files.files:
                if str(file)[0:4] == "PQC_":
                    files_list.append(file)
Purpose: Downloads Excel files from SharePoint with:

Format differentiation: Separate handling for old (RQC_) and new (PQC_) formats
Selective downloading: Only processes relevant PQC files
Local storage: Saves files to structured local directories
File organization: Maintains separate folders for different formats

5. Database Operations
Table Management
pythondef trunc_table(table_name, conn_str, username, password):
    connection = oracledb.connect(user=username, password=password, dsn=conn_str)
    cur = connection.cursor()
    sql_stmt = '''TRUNCATE TABLE {0}'''.format(table_name)
    try:
        cur.execute(sql_stmt)
        connection.commit()
        logger.info("***************{0} is truncated ***************".format(table_name))
    except Exception as e:
        logger.info(e, exc_info=True)
        # Send error notification
    finally:
        cur.close()
        connection.close()
Purpose: Provides database table management with:

Data cleanup: Truncates tables before fresh data load
Backup functionality: Creates backup tables before truncation
Error handling: Comprehensive exception management
Connection management: Proper resource cleanup

Bulk Data Insertion
pythondef insert_data_to_pub(table_name, sql_stmt, data, conn_str, username, password):
    connection = oracledb.connect(user=PUB_USERNAME, password=PUB_PWD, dsn=conn_str)
    cur = connection.cursor()
    chunk_size = 10000
    
    # Special handling for sendback data
    if table_name == "pqc_sendbacks_reported":
        for i in range(0, len(data)):
            data_set = [data[i]]
            cur.setinputsizes(int, str, str, str, oracledb.TIMESTAMP, ...)
            cur.execute(sql_stmt, data_set)
    else:
        # Batch processing for other tables
        for i in range(0, len(data), chunk_size):
            data_set = data[i:i + chunk_size]
            cur.executemany(sql_stmt, data_set)
Purpose: Optimized data insertion with:

Batch processing: 10,000 records per batch for performance
Data type handling: Proper type mapping for Oracle database
Special case handling: Different processing for timestamp-sensitive data
Performance optimization: Bulk insert operations

6. Data Transformation Functions
Quality Assurance Check Transformation
pythondef get_qa_check_data(row):
    defect_check_str = row["defect_check_src"].upper().strip()
    
    if defect_check_str == "PASS/NO DEFECT":
        row["defect_check"] = "PASS"
        row["actionable_check"] = "NA"
        row["sendback_check"] = "NO"
    elif defect_check_str == "ACTIONABLE/DEFECT":
        row["defect_check"] = "DEFECT"
        row["actionable_check"] = "YES"
        row["sendback_check"] = "NO"
    # ... additional conditions
Purpose: Transforms quality check data by:

Standardization: Converting various input formats to consistent values
Business logic application: Applying PQC-specific transformation rules
Data normalization: Ensuring consistent data representation
Multi-field population: Setting multiple related fields based on single input

Case Assignment Integration
pythondef get_indexing_ccr_details(row):
    case_key = row["key"]
    case_sno = row["s_no"]
    ca_df = new_fmt_case_assignment_res_df[
        (new_fmt_case_assignment_res_df["key"] == case_key) & 
        (new_fmt_case_assignment_res_df["s_no"] == case_sno)
    ]
    
    case_indexing_specialist_sso = str(ca_df["indexing_specialist_sso"].iloc[0]).strip()
    case_owner_sso = str(ca_df["ccr_investigator_sso"].iloc[0]).strip()
    
    row["indexing_specialist_sso"] = case_indexing_specialist_sso
    row["ccr_investigator_sso"] = case_owner_sso
    return row
Purpose: Enriches question data with assignment information by:

Data linking: Connecting questions to their corresponding assignments
Key-based lookup: Using composite keys for accurate matching
Data enrichment: Adding specialist and investigator information
Data integrity: Handling cases with multiple matches

7. Main Processing Pipeline
Old Format Data Processing
The pipeline processes legacy Excel files (RQC_*) with this structure:
Memory Map for Old Format Processing:
Old Format File (RQC_*.xlsx)
├── Case Assignment Sheet
│   ├── PQC Details (rows 1-4)
│   └── Case Data (rows 8+)
├── Call Listening Sheet
├── Intake Sheet
├── Investigation Sheet
├── Alleged Violations Sheet
├── AIT Sheet
├── Resolution Sheet
├── Response Sheet
└── Case Closure Sheet

Processing Flow:
1. Extract PQC SSO and Name from Assignment sheet
2. Read case numbers from Assignment sheet
3. For each case, iterate through all question sheets
4. Map questions from template to actual responses
5. Transform and normalize data
6. Combine into unified dataset
New Format Data Processing
The pipeline processes current Excel files (PQC_*) with this structure:
Memory Map for New Format Processing:
New Format File (PQC_*.xlsx)
├── Case_Assignment Sheet
│   ├── PQC Details (rows 1-4)
│   └── Assignment Data (rows 9+)
├── Case_Testing Sheet
│   └── Questions and Responses
└── Case_Closure Sheet
    └── Closure Information

Processing Flow:
1. Extract PQC details from Assignment sheet
2. Process assignment data
3. Process testing (questions) data
4. Apply transformation functions
5. Enrich with assignment details
6. Process closure data
7. Combine into unified dataset
8. Data Aggregation and Analysis
Defect Analysis
python# Generate defect flags
mgd_case_questions_agg_tmp["defect_check_y_flag"] = mgd_case_questions_agg_tmp["defect_check"].isin(defect_chk_y_list)

# Create case-level summaries
defect_summary = mgd_case_questions_agg_tmp.groupby([
    "s_no", "key", "pqc_assignment_date", "pqc_sso", "pqc_name", "case_number"
]).agg(
    defect_y_question_ct=("defect_check_y_flag", sum),
    defect_na_question_ct=("defect_check_na_flag", sum),
    defect_pass_question_ct=("defect_check_pass_flag", sum),
    defect_blank_question_ct=("defect_check_blank_flag", sum)
)
Purpose: Creates comprehensive analytics by:

Flag generation: Boolean indicators for different defect types
Aggregation: Case-level summaries from question-level data
Business metrics: Compliance vs. procedural defect categorization
Performance indicators: Pass/fail rates and completion metrics

9. SharePoint List Integration
Defects Reported Processing
pythonspo_list_name = "RQC Defect Reporting"
defects_list = context.web.lists.get_by_title(spo_list_name)
defects_data_list = query_large_list(defects_list)
defects_data_spo_df = pd.DataFrame(defects_data_list)

# Column mapping and transformation
defects_spo_rename_cols_dict = {
    'ID': 'id', 
    'Title': 'defect_clfb_num', 
    'SpecialistSSO': 'specialist_sso',
    # ... additional mappings
}
Purpose: Integrates external defect reporting data by:

List extraction: Reading SharePoint list data
Column standardization: Mapping SharePoint columns to database schema
Timezone handling: Converting UTC timestamps to local time
Data enrichment: Adding external defect information to pipeline

10. Final Database Loading
Target Tables and Data Flow
Database Tables Created/Updated:
├── pqc_case_questions (22 columns)
│   └── Individual question responses and outcomes
├── pqc_case_assignments (19 columns)
│   └── Case assignment details and metadata
├── pqc_case_closures (35 columns)
│   └── Case closure information and regulatory details
├── pqc_case_questions_aggr (23 columns)
│   └── Aggregated metrics and KPIs
├── pqc_defects_reported (16 columns)
│   └── External defect reporting data
└── pqc_sendbacks_reported (17 columns)
    └── Sendback case escalation data
Each table follows this loading pattern:

Backup existing data to _bkp tables
Truncate target table for fresh load
Insert new data in optimized batches
Log success/failure with record counts

11. Error Handling and Monitoring
The pipeline implements comprehensive error handling:

Exception catching: Try-catch blocks around critical operations
Email notifications: Automatic alerts for failures
Detailed logging: Complete audit trail of operations
Resource cleanup: Guaranteed cleanup of temporary files and connections
Transaction management: Proper commit/rollback for database operations

12. Performance Optimizations

Batch processing: 10,000 record chunks for database operations
Pagination: 500-item pages for SharePoint queries
Memory management: Strategic dataframe cleanup
Connection pooling: Efficient database connection handling
Parallel processing: Simultaneous old/new format processing where possible

Summary
This PQC data ingestion pipeline is a sophisticated ETL (Extract, Transform, Load) system that:

Extracts data from multiple SharePoint sources (files and lists)
Transforms data through complex business logic and standardization
Loads processed data into Oracle database tables for analytics
Monitors operations through comprehensive logging and notifications
Maintains data quality through validation and error handling

The system supports both legacy (RQC) and current (PQC) data formats, handles large datasets efficiently, and provides robust error recovery and notification mechanisms. It's designed for automated, scheduled execution in an enterprise environment with full audit capabilities.
