import src.fisher_func as f
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split

iris_data = f.get_irises()