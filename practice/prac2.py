import matplotlib.pyplot as plt
import numpy as np
from model import LogitRegression
from sklearn.preprocessing import normalize

N = 1000
M0 = [1, 2, 3, 5]
M1 = [5, 6, 10, 18]
c0 = [1, 2, 3, 4]
c1 = [1, 2, 3, 4]
A = np.random.normal(M0[0],c0[0], (N,1))
B = np.random.normal(M1[0],c1[0], (N,1))
for i in range(1,len(M0)):
   A = np.hstack((A, np.random.normal(M0[i],c0[i], (N,1))))
   B = np.hstack((B, np.random.normal(M1[i],c1[i], (N,1))))

fig, ax = plt.subplots(3, 2)
ax[0,0].scatter(A[:,0], A[:,1])
ax[0,0].scatter(B[:,0], B[:,1])
ax[0,1].scatter(A[:,0], A[:,2])
ax[0,1].scatter(B[:,0], B[:,2])
ax[1,0].scatter(A[:,0], A[:,3])
ax[1,0].scatter(B[:,0], B[:,3])
ax[1,1].scatter(A[:,1], A[:,2])
ax[1,1].scatter(B[:,1], B[:,2])
ax[2,0].scatter(A[:,1], A[:,3])
ax[2,0].scatter(B[:,1], B[:,3])
ax[2,1].scatter(A[:,2], A[:,3])
ax[2,1].scatter(B[:,2], B[:,3])
plt.show()
Y0 = np.zeros((N,1))
Y1 = np.ones((N,1))
Y = np.vstack((Y0,Y1))
X = np.vstack((A,B))
data = np.hstack((X,Y))
data_copy = data[:]

def split(data):
    data_copy = data[:]
    np.random.shuffle(data_copy)
    percent = int(len(data_copy)*0.8)
    train = data_copy[0:percent,:]
    test = data_copy[percent:,:]

    train_X = train[:,0:4]
    train_Y = np.where(train[:,4] == 1, 1, 0)
    test_X = test[:,0:4]
    test_Y = np.where(test[:,4] == 1, 1, 0)

    return train_X, train_Y, test_X, test_Y



train_X, train_Y, test_X, test_Y = split(data_copy)

model = LogitRegression(0.6, 1000)
print(train_Y)
model.fit(normalize(train_X), train_Y)

pred = model.predict(normalize(test_X))

# fig, ax = plt.subplots(1, 2)
# ax[0].hist(pred.T)
# ax[1].hist(test_Y)
plt.plot(model.loss)
plt.show()