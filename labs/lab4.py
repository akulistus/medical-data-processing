import numpy as np
import src.fisher_func as fisher_func
import matplotlib.pyplot as plt

iris_data = fisher_func.get_irises()

class1 = "versicolor"
class2 = "setosa"

data_class1 = iris_data[class1]
data_class2 = iris_data[class2]
test = np.vstack([data_class1, data_class2])
print(test)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(test[:,1], test[:,2], test[:,3])
plt.show()

diff_mean = np.mean(data_class1, axis=0) - np.mean(data_class2, axis=0)
sum_covariance = np.cov(data_class1, rowvar=0) + np.cov(data_class2, rowvar=0)
W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
print(W)
w = W/np.linalg.norm(W)

proj_class1 = np.matmul(data_class1, w)
proj_class2 = np.matmul(data_class2, w)

print(proj_class1.reshape(1,50))
hist_data = np.hstack((proj_class1.reshape(50,1), proj_class2.reshape(50,1)))
print(hist_data)
plt.hist(hist_data)
plt.show()