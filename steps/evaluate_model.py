# Loading libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from zenml import step

@step
def evaluation(pipeline,X_test,y_test):
    y_test_pred = pipeline.predict(X_test)

    test_accuracy = accuracy_score(y_test,y_test_pred)

    return test_accuracy