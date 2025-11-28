# Loading the required libraries
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from zenml import step

@step
def preprocessor(X_train:pd.DataFrame,y_train):
    """
    Apply preprocessing and returns the preprocessor
    """
    num_cols = X_train.select_dtypes(["int","float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
       transformers=[
           ("num_scaler",StandardScaler(),num_cols)
       ]
   )
    return preprocessor