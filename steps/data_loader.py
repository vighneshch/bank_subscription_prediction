# Loading the required libraries
import pandas as pd
import logging
from zenml import step
from typing import Optional
from sklearn.datasets import load_iris

logger = logging.getLogger("__name__")

@step
def load_data(filepath:Optional[str] = None) -> pd.DataFrame:

    """
    Loads data from csv file and returns a pandas dataframe

    """
    if filepath:
        df = pd.read_csv(filepath)
        logger.info("Loading the file from the local file path")
    else:
        X,y = load_iris(return_X_y=True,as_frame=True)
        df = pd.concat([X,y],axis=1)
        logger.info("Loading iris dataset from sklearn")
    
    return df
        

    