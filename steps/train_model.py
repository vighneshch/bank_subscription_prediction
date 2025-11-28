# Loading the required libaries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from zenml import step

@step
def train_model(preprocessor,X_train,y_train):
    model =  KNeighborsClassifier(n_neighbors=9,weights="uniform")

    pipeline = Pipeline(
        steps = [("preprocessor",preprocessor),
                 ("knn",model)]
    )

    pipeline.fit(X_train,y_train)

    return pipeline
