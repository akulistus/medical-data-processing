import src.fisher_func as fisher_func
import matplotlib.pyplot as plt
from sklearn.decomposition._pca import PCA

PCA_obj = PCA(n_components=2)

data = fisher_func.get_irises()

data_setosa = data["setosa"]
data_versicolor = data["virginica"]

reduced_data_1 = PCA_obj.fit_transform(data_setosa)
reduced_data_2 = PCA_obj.fit_transform(data_versicolor)

plt.scatter(reduced_data_1, reduced_data_2)
plt.show()