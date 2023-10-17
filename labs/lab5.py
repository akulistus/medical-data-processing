import numpy as np
import src.fisher_func as fisher_func
import matplotlib.pyplot as plt
from sklearn.decomposition._pca import PCA

PCA_obj = PCA()

data, Y = fisher_func.get_irises()

data_setosa = data["setosa"]
data_virginica = data["virginica"]
data_versicolor = data["versicolor"]

data_svi = np.vstack((data_setosa, data_virginica))
data_sve = np.vstack((data_setosa, data_versicolor))
data_vevi = np.vstack((data_versicolor, data_virginica))
reduced_data_svi = PCA_obj.fit_transform(data_svi)
print(reduced_data_svi)
# reduced_data_sve = PCA_obj.fit_transform(data_sve)
# reduced_data_vevi = PCA_obj.fit_transform(data_vevi)

# plt.scatter(reduced_data_svi[:,0],reduced_data_svi[:,1])
# plt.xlabel('Первая главная компонента')
# plt.ylabel('Вторая главная компонента')
# plt.title('setosa и virginica в пространстве полученных главных компонент')
# plt.show()

# plt.scatter(reduced_data_sve[:,0],reduced_data_sve[:,1])
# plt.xlabel('Первая главная компонента')
# plt.ylabel('Вторая главная компонента')
# plt.title('setosa и versicolor в пространстве полученных главных компонент')
# plt.show()

# plt.scatter(reduced_data_vevi[0:49,0],reduced_data_vevi[0:49,1], marker='o')
# plt.scatter(reduced_data_vevi[50:99,0],reduced_data_vevi[50:99,1], marker='*')
# plt.xlabel('Первая главная компонента')
# plt.ylabel('Вторая главная компонента')
# plt.title('versicolor и virginica в пространстве полученных главных компонент')
# plt.show()
