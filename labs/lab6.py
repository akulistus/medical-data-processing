import numpy as np
import copy as cp
import matplotlib.pyplot as plt

gr1 = np.array([[0.7,   0.3,   1.2],
[0.5,   0.7,   1.0],
[0.4,   1.0,   0.4],
[0.7,   0.7,   1.0],
[0.6,   0.6,   1.5],
[0.6,   0.6,   1.2],
[0.6,   0.5,   1.0],
[0.4,   0.9,   0.6],
[0.5,   0.6,   1.1],
[0.8,   0.3,   1.2]])

gr2 = np.array([[0.4,   0.2,   0.8],
[0.2,   0.2,   0.7],
[0.9,   0.3,   0.5],
[0.8,   0.3,   0.6],
[0.5,   0.6,   0.4],
[0.6,   0.5,   0.7],
[0.4,   0.4,   1.2],
[0.6,   0.3,   1.0],
[0.3,   0.2,   0.6],
[0.5,   0.5,   0.8]])

# gr2 = gr2*-1

# diff_mean = np.mean(gr1, axis=0) - np.mean(gr2, axis=0)
# sum_covariance = np.cov(gr1, rowvar=0) + np.cov(gr2, rowvar=0)
# W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
# print(W)
# w = W/np.linalg.norm(W)


# proj_class1 = np.matmul(data, w)

# plt.hist(proj_class1)
# plt.show()

data = np.vstack((gr1, gr2*-1))
# Set learning rate to 0.1
learn_coeff = 0.1
projections = np.zeros(20)
projections[1] = -1
w = [18,52,21]

while min(projections) < 0:
    # Test all data and tweak the weights
    for ind, data_sample in enumerate(data):
        projections[ind] = np.matmul(data_sample, w)
        if projections[ind] < 0:
            w = w + data_sample * learn_coeff

    print(min(projections))
    # Recalculate projections based on last value of w for the current iteration
    projections = np.matmul(data, w)

plt.hist(projections)
plt.show()