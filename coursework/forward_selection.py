import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class ForwardSelection():
    def __init__(self, estimator, n_features_to_select:int) -> None:
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.selected_features = []

    def _train_model(self, X:pd.DataFrame, y:pd.DataFrame, val_X:pd.DataFrame = None, val_Y:pd.DataFrame = None):
        self.estimator.fit(
            np.array(X), 
            np.array(y), 
            np.array(val_X), 
            np.array(val_Y)
        )

        return self.estimator.predict(val_X)
            

    def fit(self, X:pd.DataFrame, y:pd.DataFrame, val_X:pd.DataFrame = None, val_Y:pd.DataFrame = None):

        features = X.columns.values.tolist()

        for _ in range(0, self.n_features_to_select):
            best_acc = 0
            best_feature = ""
            for feature in features:



                predictions = self._train_model(
                    X[[feature]+self.selected_features],
                    y,
                    val_X[[feature]+self.selected_features],
                    val_Y
                )

                predictions = np.where( predictions > 0.5, 1, 0)
                acc = accuracy_score(val_Y.ravel(), predictions.ravel())

                if best_acc < acc:
                    best_acc = acc
                    print("here")
                    best_feature = feature
            
            self.selected_features.append(best_feature)
            features.remove(best_feature)
            print(best_acc)

        return self
