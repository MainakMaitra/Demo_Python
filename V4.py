import oracledb
import pandas as pd
import configparser
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_conn_str(hostname, port, servicename):
    """
    Returns oracle connection string.
    """
    return f"{hostname}:{port}/{servicename}"

def analyze_table_structure(connection, table_name):
    """
    Analyze table structure including columns, data types, and sample data
    """
    cursor = connection.cursor()
    
    print(f"\n{'='*80}")
    print(f"ANALYZING TABLE: {table_name.upper()}")
    print(f"{'='*80}")
    
    # Get table structure
    structure_query = """
    SELECT column_name, data_type, data_length, nullable, data_default
    FROM user_tab_columns 
    WHERE table_name = :table_name
    ORDER BY column_id
    """
    
    try:
        cursor.execute(structure_query, {'table_name': table_name.upper()})
        columns_info = cursor.fetchall()
        
        if not columns_info:
            print(f"Table {table_name} not found or no access.")
            return
            
        # Display column information
        print("\nCOLUMN STRUCTURE:")
        print("-" * 80)
        print(f"{'Column Name':<30} {'Data Type':<20} {'Length':<10} {'Nullable':<10} {'Default':<15}")
        print("-" * 80)
        
        for col_info in columns_info:
            col_name, data_type, data_length, nullable, default_val = col_info
            length_str = str(data_length) if data_length else 'N/A'
            default_str = str(default_val)[:15] if default_val else 'None'
            print(f"{col_name:<30} {data_type:<20} {length_str:<10} {nullable:<10} {default_str:<15}")
        
        # Get row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(count_query)
        row_count = cursor.fetchone()[0]
        print(f"\nTOTAL ROWS: {row_count:,}")
        
        # Get sample data (first 5 rows)
        if row_count > 0:
            sample_query = f"SELECT * FROM {table_name} WHERE ROWNUM <= 5"
            cursor.execute(sample_query)
            sample_data = cursor.fetchall()
            
            # Get column names for header
            column_names = [desc[0] for desc in cursor.description]
            
            print(f"\nSAMPLE DATA (First 5 rows):")
            print("-" * 120)
            
            # Create DataFrame for better display
            if sample_data:
                df_sample = pd.DataFrame(sample_data, columns=column_names)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_colwidth', 50)
                print(df_sample.to_string(index=False))
            else:
                print("No sample data available")
        
        # Get constraints information
        constraints_query = """
        SELECT constraint_name, constraint_type, search_condition
        FROM user_constraints 
        WHERE table_name = :table_name
        """
        cursor.execute(constraints_query, {'table_name': table_name.upper()})
        constraints = cursor.fetchall()
        
        if constraints:
            print(f"\nCONSTRAINTS:")
            print("-" * 80)
            print(f"{'Constraint Name':<30} {'Type':<10} {'Condition':<40}")
            print("-" * 80)
            for constraint in constraints:
                const_name, const_type, condition = constraint
                condition_str = str(condition)[:40] if condition else 'N/A'
                type_map = {'P': 'PRIMARY', 'R': 'FOREIGN', 'C': 'CHECK', 'U': 'UNIQUE'}
                type_str = type_map.get(const_type, const_type)
                print(f"{const_name:<30} {type_str:<10} {condition_str:<40}")
        
    except Exception as e:
        print(f"Error analyzing table {table_name}: {str(e)}")
    
    finally:
        cursor.close()

def main():
    """
    Main function to analyze PQC tables
    """
    try:
        # Read configuration - adjust path as needed
        config = configparser.ConfigParser(interpolation=None)
        
        # Try different possible config paths
        config_paths = [
            "/var/mt/suncrest/user_credentials/PQC_CONFIG",
            "./PQC_CONFIG",  # Local path
            "../config/PQC_CONFIG"  # Alternative path
        ]
        
        config_loaded = False
        for config_path in config_paths:
            try:
                config.read(config_path)
                if config.sections():  # Check if config was loaded
                    print(f"Configuration loaded from: {config_path}")
                    config_loaded = True
                    break
            except:
                continue
        
        if not config_loaded:
            print("Configuration file not found. Please provide database connection details manually.")
            print("Update the script with your database connection parameters.")
            
            # Manual configuration - UPDATE THESE VALUES
            hostname = "your_hostname"
            port = "your_port"
            servicename = "your_servicename"
            username = "your_username"
            password = "your_password"
        else:
            # Get DB credentials from config
            hostname = config.get("pub_conn", "hostname")
            port = config.get("pub_conn", "port")
            servicename = config.get("pub_conn", "servicename")
            username = config.get("pub_conn", "username")
            password = config.get("pub_conn", "password")
        
        # Create connection string
        conn_str = get_conn_str(hostname, port, servicename)
        
        print("Connecting to Oracle database...")
        print(f"Connection string: {hostname}:{port}/{servicename}")
        
        # Connect to database
        connection = oracledb.connect(user=username, password=password, dsn=conn_str)
        
        print(f"Successfully connected to database at {datetime.now()}")
        
        # Tables to analyze
        tables_to_analyze = [
            "pqc_case_closures",
            "pqc_case_questions", 
            "pqc_case_questions_aggr"
        ]
        
        # Analyze each table
        for table_name in tables_to_analyze:
            try:
                analyze_table_structure(connection, table_name)
            except Exception as e:
                print(f"Error analyzing {table_name}: {str(e)}")
                continue
        
        print(f"\n{'='*80}")
        print("TABLE ANALYSIS COMPLETED")
        print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        print(f"\nConnection failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if the Oracle client is installed")
        print("2. Verify database connection parameters")
        print("3. Ensure you have proper database permissions")
        print("4. Check if the tables exist in your schema")
        
    finally:
        try:
            if 'connection' in locals():
                connection.close()
                print("Database connection closed.")
        except:
            pass

if __name__ == "__main__":
    print("PQC Table Structure Analyzer")
    print("=" * 50)
    main()
