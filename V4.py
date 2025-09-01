from io import BytesIO
import warnings
import oracledb
import configparser
import pandas as pd
import logging

# Filter warnings like in your original code
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_conn_str(hostname, port, servicename):
    """
    Returns oracle connection string.
    """
    res = hostname + ':' + port + '/' + servicename
    return res

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
        # Read configuration following your existing pattern
        config = configparser.ConfigParser(interpolation=None)
        config.read('/var/run/secrets/user_credentials/PQC_CONFIG')
        
        # Print available sections and options for debugging
        print("Available config sections:", config.sections())
        if config.has_section('pub_conn'):
            print("Available options in pub_conn:", config.options('pub_conn'))
        
        # Get DB credentials with error handling
        try:
            PUB_HOSTNAME = config.get('pub_conn', 'hostname')
            PUB_PORT = config.get('pub_conn', 'port')
            PUB_SERVICENAME = config.get('pub_conn', 'servicename')
            PUB_USERNAME = config.get('pub_conn', 'username')
            
            # Try different possible password field names
            password_fields = ['password', 'pwd', 'PUB_PWD', 'PASSWORD']
            PUB_PWD = None
            
            for pwd_field in password_fields:
                if config.has_option('pub_conn', pwd_field):
                    PUB_PWD = config.get('pub_conn', pwd_field)
                    print(f"Found password field: {pwd_field}")
                    break
            
            if PUB_PWD is None:
                print("Password field not found in config. Available options:")
                for option in config.options('pub_conn'):
                    print(f"  - {option}")
                raise Exception("Password configuration not found")
                
        except Exception as config_error:
            print(f"Configuration error: {config_error}")
            print("\nPlease check your config file structure.")
            print("Expected structure:")
            print("[pub_conn]")
            print("hostname = your_host")
            print("port = your_port") 
            print("servicename = your_service")
            print("username = your_user")
            print("password = your_password")
            return
        
        # Create connection string using your function
        conn_str = get_conn_str(PUB_HOSTNAME, PUB_PORT, PUB_SERVICENAME)
        
        print("Connecting to Oracle database...")
        print(f"Connection string: {conn_str}")
        print(f"Username: {PUB_USERNAME}")
        
        # Connect to database using your pattern
        connection = oracledb.connect(user=PUB_USERNAME, password=PUB_PWD, dsn=conn_str)
        
        # Remove datetime.now() call to avoid import issues
        print("Successfully connected to database")
        
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
