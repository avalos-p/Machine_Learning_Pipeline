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
    def __init__(self, logger=None):
        
        self.logger = logger if logger else logging.getLogger(__name__)

        self.engine = None
        self.max_retries = 3
        self.retry_delay = 5 

        self.db_host = os.getenv('HOST')
        self.db_port = os.getenv('PORT')
        self.db_database = os.getenv('DATABASE')
        self.db_user = os.getenv('USER')
        self.db_pass = os.getenv('PASS')

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
            # Execute the query and return the results as a pandas DataFrame.
            self.logger.info(f"Executing query")
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
        return None