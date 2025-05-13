import pandas as pd
import logging
import math
from datetime import datetime

def df_cleaner(df, logger=None):
     """
       Cleans the DataFrame by converting date columns to datetime, calculating age 
       from date of birth, and dropping unnecessary columns."""
     
     assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
     if logger is None:
       logger = logging.getLogger(__name__)

     current_date = datetime.now()
     try:
       df['dob'] = pd.to_datetime(df['dob'])
       df['age_in_days'] = (current_date - df['dob']).dt.days
       df['age'] = df['age_in_days'] / 365
       df['age'] = df['age'].apply(math.floor)
       df['age'] = df['age'].astype(int) # Age of person
       df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time']) # Transform to date
       df['hour'] = df['trans_date_trans_time'].dt.hour # Gets hour
       df.drop(columns=['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last', 'street', 'city', 'zip',
                        'lat', 'long', 'job','trans_num', 'unix_time', 'merch_lat', 'merch_long','dob',
                        'age_in_days','trans_date_trans_time'],inplace=True) # Removing not generalized
       df.rename(columns={'amt' : 'amount','city_pop':'city_population'}, inplace=True)
       return df
          
     except (TypeError, ValueError) as err:
       # Handle specific error types for better debugging
       logger.error(f"Error cleaning DataFrame: {str(err)}")
       return None
       

def split_data(df, y, logger=None):
     """
       Splits the DataFrame into features and target variable.
     """
     assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
     if logger is None:
       logger = logging.getLogger(__name__)
     if y not in df.columns:
       raise ValueError(f"El dataframe debe tener una columna {y}")
     Y = df[y]  # Target
     X = df.drop(y, axis=1)  # Features
     return X, Y

