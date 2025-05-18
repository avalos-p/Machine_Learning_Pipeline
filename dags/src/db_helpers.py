import os
import time
import logging
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('config/credentials/.env')
load_dotenv(dotenv_path=dotenv_path)

class SQLConnection:
    def __init__(self,host,port,database,user,password, logger=None):
        
        self.logger = logger if logger else logging.getLogger(__name__)

        self.engine = None
        self.max_retries = 3
        self.retry_delay = 5 

        self.db_host = host
        self.db_port = port
        self.db_database = database
        self.db_user = user
        self.db_pass = password

        self.db_url = self._build_db_url()

    def _build_db_url(self):
        """
        Formatting the database URL
        """
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_database}"

    def connect(self):
        """
        Try to connect to the database
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                self.logger.info("Trying to connect to the database...")
                self.engine = create_engine(self.db_url)
                
                self.engine.connect()
                self.logger.info("Connected to the database successfully.")
                return self.engine
            except OperationalError as e:
                self.logger.error(f"Conection error: {e}, retrying")
                attempts += 1
                time.sleep(self.retry_delay)
        self.logger.critical("Could not connect to the database.")
        return None

    def close_connection(self):
        """
        Close the connection to the database
        """
        if self.engine:
            self.logger.info("Ending connection to the database.")
            self.engine.dispose()
            self.logger.info("Connection closed.")
        else:
            self.logger.warning("There is no connection to close.")

            
    def execute_query(self, query):
        """
        This method executes a query and returns the results as a list of dictionaries
        """
        try:
            with self.engine.connect() as connection:
                self.logger.info(f"Executing query")
                result = connection.execute(text(query))
                return [row for row in result]
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            return None
    
    def execute_query_to_pd(self, query):
        """
        This method executes a query and returns the results as a pandas DataFrame.
        """
        try:
            self.logger.info(f"Executing query")
            # Usar el método execute de SQLAlchemy y convertir los resultados a DataFrame
            from sqlalchemy import text
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall())
                if result.keys():  # Si hay nombres de columnas disponibles
                    df.columns = result.keys()
                return df
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
        return None