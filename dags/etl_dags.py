from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import pandas as pd
import os
from dotenv import load_dotenv
from airflow.models import Variable
from src.db_helpers import SQLConnection
from src.cleaner import df_cleaner, split_data
from src.encoder import encode_categorical_columns
from src.robust_scaler import scale_numerical_columns, transform_with_scaler

from config.logger_setup import setup_logging

# Configuración de logging
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Cargando variables de entorno")
load_dotenv(dotenv_path='config/credentials/.env')

# Configuración de variables por defecto
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Función para extraer datos desde la base de datos
def extract_data_from_db(**kwargs):
    logger.info("Iniciando extracción de datos desde la base de datos")
    try:
        # Cargar variables de entorno
        
        # logger.info(f"Variables de entorno cargadas correctamente- TEST{os.getenv('HOST')}")
        
        # Establecer conexión con la base de datos
        logger.info("Estableciendo conexión con la base de datos")
        db_connection = SQLConnection(Variable.get("HOST"),Variable.get("PORT"),Variable.get("DATABASE"),Variable.get("USER"),Variable.get("PASSWORD"), logger)
        db_connection.connect()
        logger.info("Conexión establecida")

        dag_folder = os.path.dirname(__file__)
        query_path = os.path.join(dag_folder, 'src', 'utils', 'testquery.sql')

        with open(query_path, 'r') as f:
            query = f.read()
        logger.info(f"Consulta SQL leída correctamente {query}")

        df = db_connection.execute_query_to_pd(query)
        logger.info("Consulta SQL ejecutada y datos extraídos")

        output_path = os.path.join(dag_folder, 'data', 'raw_data', 'testquery.csv')

        df.to_csv(output_path, index=False)
        
        # Cerrar la conexión
        db_connection.close_connection()
        
        logger.info(f"Datos extraídos y guardados en {output_path}")
        
        return
    except Exception as e:
        logger.error(f"Error durante la extracción de datos: {e}")
        raise

# Definición del DAG
with DAG(
    dag_id='etl_pipeline_dag',
    default_args=default_args,
    description='ETL pipeline para extracción y procesamiento de datos',
    schedule=None,  # Ejecutar manualmente
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['etl', 'data_pipeline'],
) as dag:

    # Tarea para extraer datos
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data_from_db
    )    
    # Definir el orden de las tareas
    extract_task 

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import logging
# import pandas as pd

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.operators.postgres import PostgresOperator
# from airflow.providers.postgres.hooks.postgres import PostgresHook

# from src.db_helpers import SQLConnection
# from src.cleaner import df_cleaner, split_data
# from src.encoder import encode_categorical_columns
# from src.robust_scaler import scale_numerical_columns, transform_with_scaler



# from config.logger_setup import setup_logging
# setup_logging()
# logger = logging.getLogger(__name__)

# # Función que será llamada por el DAG
# def hello_world():
#     print("Hello World")

# # Definición del DAG
# with DAG(
#     dag_id='hello_world_dag',
#     start_date=datetime(2023, 1, 1),
#     schedule=None,  # Ejecutar manualmente
#     catchup=False,
#     tags=['not example']
# ) as dag:

#     task_hello = PythonOperator(
#         task_id='print_hello',
#         python_callable=hello_world
#     )



# def main():
    
    
#     test_data = pd.read_csv("data/raw_data/fraudTest.csv")
#     cleaned_data = df_cleaner(test_data,logger)
#     cleaned_data = encode_categorical_columns(cleaned_data,None,"models/transformers/encoders/ordinal_encoder.pkl",logger)
#     encode_categorical_columns(cleaned_data,"models/transformers/encoders/ordinal_encoder.pkl",None,logger)

#     X, Y = split_data(cleaned_data,"is_fraud",logger)
#     X = scale_numerical_columns(X,None,"models/transformers/scalers/robust_scaler.pk", logger)

