from sklearn.preprocessing import RobustScaler
import pandas as pd
import pickle
import os
import logging

def scale_numerical_columns(df, save_path=None, scaler_path=None, logger=None):
    """
    Scale numerical columns of a DataFrame using RobustScaler.
    
    Args:
        df: DataFrame with numerical columns to scale.
        save_path: Path to save the scaler (pickle file).
        scaler_path: Path to load an existing scaler.
        logger: Optional logger object for logging information.
        
    Returns:
        tuple: (DataFrame transformed, RobustScaler)
    """
    
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    if logger is None:
        logger = logging.getLogger(__name__)
        
    scaled_df = df.copy()
    
    # Select numerical columns
    numerical_columns = scaled_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    
    if len(numerical_columns) == 0:
        logger.warning("No numerical columns found to scale.")
        return None, None
    
    # CASE 1: Load existing scaler
    if scaler_path is not None:
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                logger.info(f"Scaler loaded successfully from {scaler_path}")
            
            # ONLY TRANSFORM - NO RE-TRAIN
            scaled_df[numerical_columns] = scaler.transform(scaled_df[numerical_columns])
            
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return None, None
    
    # CASE 2: Create new scaler (TRAINING)
    else:
        scaler = RobustScaler()
        logger.info("Creating new RobustScaler")
        
        try:
            scaled_df[numerical_columns] = scaler.fit_transform(scaled_df[numerical_columns])
            logger.info(f"Successfully scaled {len(numerical_columns)} numerical columns")
        except Exception as e:
            logger.error(f"Error during scaling: {str(e)}")
            return None, None
        
        # Save scaler if path is specified
        if save_path:
            try:
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)           
                with open(save_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Scaler saved successfully to {save_path}")
            except Exception as e:
                logger.error(f"Error saving scaler: {str(e)}")
    
    return scaled_df, scaler