import warnings
import oracledb
import configparser
import pandas as pd

warnings.filterwarnings("ignore")

def get_conn_str(hostname, port, servicename):
    res = hostname + ':' + port + '/' + servicename
    return res

# Read config
config = configparser.ConfigParser(interpolation=None)
config.read('/var/run/secrets/user_credentials/PQC_CONFIG')

# Get credentials
PUB_HOSTNAME = config.get('pub_conn', 'hostname')
PUB_PORT = config.get('pub_conn', 'port')
PUB_SERVICENAME = config.get('pub_conn', 'servicename')
PUB_USERNAME = config.get('pub_conn', 'username')
PUB_PWD = config.get('pub_conn', 'pwd')

# Connect
conn_str = get_conn_str(PUB_HOSTNAME, PUB_PORT, PUB_SERVICENAME)
connection = oracledb.connect(user=PUB_USERNAME, password=PUB_PWD, dsn=conn_str)

print("Connected successfully!")

# Query pqc_case_closures
print("\n" + "="*50)
print("PQC_CASE_CLOSURES")
print("="*50)
query1 = "SELECT * FROM pqc_case_closures WHERE ROWNUM <= 5"
df_closures = pd.read_sql(query1, connection)
print(f"Shape: {df_closures.shape}")
print(df_closures.head())

# Query pqc_case_questions
print("\n" + "="*50)
print("PQC_CASE_QUESTIONS")
print("="*50)
query2 = "SELECT * FROM pqc_case_questions WHERE ROWNUM <= 5"
df_questions = pd.read_sql(query2, connection)
print(f"Shape: {df_questions.shape}")
print(df_questions.head())

# Query pqc_case_questions_aggr
print("\n" + "="*50)
print("PQC_CASE_QUESTIONS_AGGR")
print("="*50)
query3 = "SELECT * FROM pqc_case_questions_aggr WHERE ROWNUM <= 5"
df_aggr = pd.read_sql(query3, connection)
print(f"Shape: {df_aggr.shape}")
print(df_aggr.head())

# Close connection
connection.close()
print("\nConnection closed.")
