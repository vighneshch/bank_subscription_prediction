# Loading the libraries
import pandas as pd
from zenml import step
from sklearn.model_selection import train_test_split

@step
def data_splitter(df:pd.DataFrame) -> pd.DataFrame:

    """
    Splits the data into train and test dataset
    
    Parameters:
    df : A pandas dataframe

    """
    X = df.drop('target',axis=1,inplace=False)
    y = df['target']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    return X_train,X_test,y_train,y_test