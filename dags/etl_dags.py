from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Función que será llamada por el DAG
def hello_world():
    print("Hello World")

# Definición del DAG
with DAG(
    dag_id='hello_world_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,  # Ejecutar manualmente
    catchup=False,
    tags=['not example']
) as dag:

    task_hello = PythonOperator(
        task_id='print_hello',
        python_callable=hello_world
    )



# import logging
# import pandas as pd

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.operators.postgres import PostgresOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook

# from src.cleaner import df_cleaner, split_data
# from src.encoder import encode_categorical_columns
# from src.robust_scaler import scale_numerical_columns, transform_with_scaler
# from src.db_helpers import SQLConnection


# from config.logger_setup import setup_logging
# setup_logging()
# logger = logging.getLogger(__name__)

# def main():
    
    
#     test_data = pd.read_csv("data/raw_data/fraudTest.csv")
#     cleaned_data = df_cleaner(test_data,logger)
#     cleaned_data = encode_categorical_columns(cleaned_data,None,"models/transformers/encoders/ordinal_encoder.pkl",logger)
#     encode_categorical_columns(cleaned_data,"models/transformers/encoders/ordinal_encoder.pkl",None,logger)

#     X, Y = split_data(cleaned_data,"is_fraud",logger)
#     X = scale_numerical_columns(X,None,"models/transformers/scalers/robust_scaler.pk", logger)

