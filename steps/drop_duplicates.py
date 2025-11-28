# Loading libraries
import pandas as pd
from zenml import step

@step
def drop_duplicates(df:pd.DataFrame) -> pd.DataFrame:
    """
    drops the duplicates and returns the dataframe

    """
    df.drop_duplicates(inplace=True)

    return df