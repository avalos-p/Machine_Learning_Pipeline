import os
import logging
import pandas as pd

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from dotenv import load_dotenv
from src.db_helpers import SQLConnection
from src.cleaner import df_cleaner, split_data
from src.encoder import encode_categorical_columns
from src.robust_scaler import scale_numerical_columns
from src.model2 import AdvancedModelTrainer

from config.logger_setup import setup_logging

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)

logger.info("Loading environment variables")
load_dotenv(dotenv_path='config/credentials/.env')

# Default variable configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Function to extract data from the database
def extract_data_from_db(**kwargs):
    logger.info("Starting data extraction from the database")
    try:
        logger.info("Establishing connection with the database")
        db_connection = SQLConnection(Variable.get("HOST"),Variable.get("PORT"),Variable.get("DATABASE"),Variable.get("USER"),Variable.get("PASSWORD"), logger)
        db_connection.connect()
        logger.info("Connection established")

        dag_folder = os.path.dirname(__file__)
        query_path = os.path.join(dag_folder, 'src', 'utils', 'get_data.sql')

        with open(query_path, 'r') as f:
            query = f.read()
        logger.info(f"SQL query read correctly {query}")

        df = db_connection.execute_query_to_pd(query)
        logger.info("SQL query executed and data extracted")

        # Define output path within dag_folder structure for consistency
        output_dir = os.path.join(dag_folder, 'data', 'raw_data')
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        output_path = os.path.join(output_dir, 'fraud_database.csv')

        df.to_csv(output_path, index=False)
        
        db_connection.close_connection()
        
        logger.info(f"Data extracted and saved to {output_path}")
        
        return # Explicitly return None or avoid return if nothing is passed via XComs
    except Exception as e:
        logger.error(f"Error during data extraction: {e}", exc_info=True)
        raise

def clean_and_encode_data(**kwargs):
    logger.info("Starting data cleaning and encoding")
    try:
        dag_folder = os.path.dirname(__file__)
        raw_data_path = os.path.join(dag_folder, 'data', 'raw_data', 'fraud_database.csv')
        
        db = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded from {raw_data_path}")
        
        db_cleaned = df_cleaner(db, logger)
        logger.info("Data cleaning completed")

        encoder_dir = os.path.join(dag_folder, 'models', 'transformers', 'encoders')
        os.makedirs(encoder_dir, exist_ok=True)
        encoder_path = os.path.join(encoder_dir, 'ordinal_encoder.pkl')
        
        db_encoded = encode_categorical_columns(db_cleaned, encoder_path, None, logger)
        logger.info(f"Categorical column encoding completed. Encoder saved to {encoder_path}")

        cleaned_data_dir = os.path.join(dag_folder, 'data', 'cleaned_data')
        os.makedirs(cleaned_data_dir, exist_ok=True)
        cleaned_data_path = os.path.join(cleaned_data_dir, 'fraud_database_cleaned.csv')
        
        db_encoded.to_csv(cleaned_data_path, index=False)
        logger.info(f"Cleaned and encoded data saved to {cleaned_data_path}")
    except Exception as e:
        logger.error(f"Error during data cleaning and encoding: {e}", exc_info=True)
        raise

def preprocess_for_model(**kwargs):
    logger.info("Starting preprocessing for the advanced model")
    try:
        dag_folder = os.path.dirname(__file__)
        cleaned_data_path = os.path.join(dag_folder, 'data', 'cleaned_data', 'fraud_database_cleaned.csv')
        db = pd.read_csv(cleaned_data_path)
        logger.info(f"Cleaned data loaded from {cleaned_data_path}")

        X, Y = split_data(db, "is_fraud", logger)
        logger.info("Data split into features (X) and target (Y)")

        scaler_dir = os.path.join(dag_folder, 'models', 'transformers', 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = os.path.join(scaler_dir, 'robust_scaler.pkl')
        
        X_scaled = scale_numerical_columns(X, scaler_path, None, logger)
        logger.info(f"Numerical column scaling completed. Scaler saved to {scaler_path}")

        # Recombine X_scaled and Y for AdvancedModelTrainer, which expects a single DataFrame with the target column
        processed_df = X_scaled.copy()
        processed_df['is_fraud'] = Y # Assumes Y is a pandas Series with a compatible index

        processed_data_dir = os.path.join(dag_folder, 'data', 'cleaned_data')
        os.makedirs(processed_data_dir, exist_ok=True)
        processed_data_path = os.path.join(processed_data_dir, 'fraud_database_cleaned.csv')
        
        processed_df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data for the advanced model saved to {processed_data_path}")
    except Exception as e:
        logger.error(f"Error during preprocessing for the advanced model: {e}", exc_info=True)
        raise

def run_advanced_model_training(**kwargs):
    logger.info("Starting advanced model training")
    try:
        dag_folder = os.path.dirname(__file__)
        processed_data_path = os.path.join(dag_folder, 'data', 'cleaned_data', 'fraud_database_cleaned.csv')
        config_file_path = os.path.join(dag_folder,'src', 'utils', 'model_config.json')

        logger.info(f"Using model configuration file: {config_file_path}")
        logger.info(f"Using preprocessed data from: {processed_data_path}")

        # Construct absolute paths based on dag_folder for outputs
        # Note: AdvancedModelTrainer already handles os.makedirs for these paths internally if given relative paths from its execution context (which is this script's location)
        # However, it's good practice to ensure they are based on the DAG's root if paths in config are relative.
        # For this example, assuming paths in config are relative to the DAG folder or will be handled by the model trainer.

        trainer = AdvancedModelTrainer(config_file_path, logger=logger)
        trainer.load_data(filepath=processed_data_path)
        trained_model, metrics = trainer.train(target_col='is_fraud')

        trainer.save_model(os.path.join(dag_folder,'models','trained_models', 'first_model.joblib'))
        trainer.save_metrics(os.path.join(dag_folder,'models','trained_models', 'first_model_metrics.json'))
        trainer.save_plots(os.path.join(dag_folder,'plots'))
        
        logger.info(f"Advanced model training completed. Model: {trained_model}")
        logger.info(f"Advanced model metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error during advanced model training: {e}", exc_info=True)
        raise

# DAG Definition
with DAG(
    dag_id='fraud_detection_etl_and_training_pipeline', # Renamed DAG ID for clarity
    default_args=default_args,
    description='ETL pipeline for extraction, processing, and training of fraud detection model',
    schedule=None,  # Run manually
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['etl', 'data_pipeline', 'fraud_detection', 'model_training'],
) as dag:

    extract_task = PythonOperator(
        task_id='extract_data_from_database',
        python_callable=extract_data_from_db
    )

    clean_encode_task = PythonOperator(
        task_id='clean_and_encode_data',
        python_callable=clean_and_encode_data
    )

    preprocess_model_task = PythonOperator(
        task_id='preprocess_data_for_model',
        python_callable=preprocess_for_model
    )

    train_advanced_model_task = PythonOperator(
        task_id='train_advanced_decision_tree_model',
        python_callable=run_advanced_model_training
    )

    # Define task order
    extract_task >> clean_encode_task >> preprocess_model_task >> train_advanced_model_task


