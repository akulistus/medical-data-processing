import numpy as np
import src.fisher_func as fisher_func
import matplotlib.pyplot as plt
from sklearn.decomposition._pca import PCA

PCA_obj = PCA(n_components=2)

data = fisher_func.get_irises()

data_setosa = data["setosa"]
data_versicolor = data["virginica"]
data = np.vstack((data_setosa, data_versicolor))

reduced_data = PCA_obj.fit_transform(data)

plt.scatter(reduced_data[:,0],reduced_data[:,1])
plt.show()