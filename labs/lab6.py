import numpy as np
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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(gr1[:,0], gr1[:,1], gr1[:,2])
ax.scatter(gr2[:,0], gr2[:,1], gr2[:,2])
plt.show()

ones1 = np.ones((10,1))
ones2 = np.ones((10,1))
gr1_1 = np.hstack((gr1, ones1))
gr2_2 = np.hstack((gr2, ones2))
gr2_2 = gr2_2*-1

diff_mean = np.mean(gr1_1, axis=0) - np.mean(gr2_2, axis=0)
sum_covariance = np.cov(gr1_1, rowvar=0) + np.cov(gr2_2, rowvar=0)
W = np.matmul(np.linalg.pinv(sum_covariance), diff_mean)
print(W)
w = W/np.linalg.norm(W)

data = np.vstack((gr1_1,gr2_2))
projections = np.matmul(data, w)

learn_coeff = 0.1
while min(projections) < 0:
    for ind, data_sample in enumerate(data):
        projections[ind] = np.matmul(data_sample, w)
        if projections[ind] < 0:
            w = w + data_sample * learn_coeff
    projections = np.matmul(data, w)

test = np.matmul(gr1_1, w)
gr2_2 = gr2_2*-1
test1 = np.matmul(gr2_2, w)

plt.hist(test)
plt.hist(test1)
plt.xlabel("Проекции")
plt.ylabel("N")
plt.title("Гистограмма проекций двух классов")
plt.show()
