import numpy as np
import fisher_func
import matplotlib.pyplot as plt


# Пример
# data = load_iris()
# X = data.data
# Y = data.target
# Y_str = data.target_names

# setosa_inds = Y == np.where(Y_str == "setosa")
# setosa_data = X[np.ravel(setosa_inds), :]

# plt.scatter(setosa_data[:, 0], setosa_data[:, 1])
# plt.show()

# Задние

prty_1 = 2
prty_2 = 3

iris_data = fisher_func.get_irises()
Y_str = iris_data.keys()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
scatter_data_1 = iris_data["setosa"][:, prty_1]
scatter_data_2 = iris_data["setosa"][:, prty_2]
ax1.scatter(scatter_data_1, scatter_data_2)

scatter_data_1 = iris_data["versicolor"][:, prty_1]
scatter_data_2 = iris_data["versicolor"][:, prty_2]
ax2.scatter(scatter_data_1, scatter_data_2)

scatter_data_1 = iris_data["virginica"][:, prty_1]
scatter_data_2 = iris_data["virginica"][:, prty_2]
ax3.scatter(scatter_data_1, scatter_data_2)
plt.show()

flowers_info = dict()
for name in Y_str:
    data_1 = iris_data[name][:, prty_1]
    data_2 = iris_data[name][:, prty_2]

    flower_mean = (np.mean(data_1), np.mean(data_2))
    flower_std = (np.std(data_1), np.std(data_2))
    flower_xcor = np.corrcoef(data_1, data_1)
    flowers_info[name] = [flower_mean, flower_std, flower_xcor]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(iris_data["setosa"][:, prty_1])
ax1.hist(iris_data["setosa"][:, prty_2])
data1 = iris_data["setosa"][:, prty_1]
data2 = iris_data["setosa"][:, prty_2]
ax2.boxplot([data1, data2])
plt.show()
