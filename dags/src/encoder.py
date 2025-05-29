from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import pickle
import os
import logging

def encode_categorical_columns(df, save_path=None, encoder_path=None, logger=None):
    """
    Encodes categorical columns of a DataFrame and optionally saves the encoder
    or loads it from a file and encodes the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame to encode.
    save_path (str): Path to save the encoder.
    encoder_path (str): Path to load the encoder.   
    logger (logging.Logger): Logger for logging messages.
    """

    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    if logger is None:
        logger = logging.getLogger(__name__)
        
    encoded_df = df.copy()
    if encoder_path is None:
        encoder = OrdinalEncoder()
    else:
        try:
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
                logger.info(f"Encoder loaded from {encoder_path}")
        except Exception as e:
            logger.error(f"Error loading encoder: {str(e)}")
            return None
    
    # Select categorical columns
    categorical_columns = encoded_df.select_dtypes(include=['object']).columns
    
    if len(categorical_columns) == 0:
        logger.warning("Columns to encode not found.")
        return encoded_df, encoder
    
    encoded_df[categorical_columns] = encoder.fit_transform(encoded_df[categorical_columns])
    
    if save_path:
        try:
            directory = os.path.dirname(save_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)           
            with open(save_path, 'wb') as f:
                pickle.dump(encoder, f)
            logger.info(f"Encoder successfully saved in {save_path}")
        except Exception as e:
            logger.error(f"Error saving encoder: {str(e)}")
    
    return encoded_df