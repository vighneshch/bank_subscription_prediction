# Loading the required libraries
from typing import Optional
from zenml import pipeline
from steps.data_loader import load_data
from steps.data_splitter import data_splitter
from steps.drop_duplicates import drop_duplicates
from steps.preprocesser import preprocessor
from steps.train_model import train_model
from steps.evaluate_model import evaluation

@pipeline
def training_pipeline(filepath:Optional[str],target_name:str):

    df = load_data()

    df = drop_duplicates(df)
    
    X_train,X_test,y_train,y_test = data_splitter(df)
    
    preprocessor = preprocessor(X_train,y_train)

    pipeline = train_model(preprocessor,X_train,X_test)

    test_accuracy = evaluation(pipeline,X_test,y_test)
    