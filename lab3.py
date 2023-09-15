import numpy as np
import fisher_func
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


#Пример
# data = load_iris()
# X = data.data
# Y = data.target
# Y_str = data.target_names

# setosa_inds = Y == np.where(Y_str == "setosa")
# setosa_data = X[np.ravel(setosa_inds), :]

# plt.scatter(setosa_data[:, 0], setosa_data[:, 1])
# plt.show()

#Задние
data = load_iris()
X = data.data
Y = data.target
Y_str = data.target_names

prty_1 = 2
prty_2 = 3

iris_data = fisher_func.get_irises()

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.scatter(iris_data["setosa"][:,prty_1],iris_data["setosa"][:,prty_2])
ax2.scatter(iris_data["versicolor"][:,prty_1],iris_data["versicolor"][:,prty_2])
ax3.scatter(iris_data["virginica"][:,prty_1],iris_data["virginica"][:,prty_2])
plt.show()

flowers_info = dict()
for name in Y_str:
    flower_mean = (np.mean(iris_data[name][:,prty_1]), np.mean(iris_data[name][:,prty_2]))
    flower_std = (np.std(iris_data[name][:,prty_1]), np.std(iris_data[name][:,prty_2]))
    flower_xcor = np.corrcoef(iris_data[name][:,prty_1], iris_data[name][:,prty_2])
    flowers_info[name] = [flower_mean,flower_std,flower_xcor]

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.hist(iris_data["setosa"][:,prty_1])
ax1.hist(iris_data["setosa"][:,prty_2])
data1 = iris_data["setosa"][:,prty_1]
data2 = iris_data["setosa"][:,prty_2]
ax2.boxplot([data1,data2])
plt.show()

# print(flowers_info["virginica"])