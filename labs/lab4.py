import numpy as np
import fisher_func

iris_data = fisher_func.get_irises()

class1 = "versicolor"
class2 = "virginica"

data_class1 = iris_data[class1]
data_class2 = iris_data[class2]

diff_mean = np.mean(data_class1, axis=0) - np.mean(data_class2, axis=0)
sum_covariance = np.cov(data_class1, rowvar=0) + np.cov(data_class2, rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
w = W/np.linalg.norm(W)

proj_class1 = np.matmul(data_class1, w)
proj_class2 = np.matmul(data_class2, w)