import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from model import LogitRegression
from src.funcs import split, normalize
from sklearn.metrics import accuracy_score, recall_score, precision_score

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
ax[0,0].title.set_text("params = 1,2")
ax[0,1].scatter(A[:,0], A[:,2])
ax[0,1].scatter(B[:,0], B[:,2])
ax[0,1].title.set_text("params = 1,3")
ax[1,0].scatter(A[:,0], A[:,3])
ax[1,0].scatter(B[:,0], B[:,3])
ax[1,0].title.set_text("params = 1,4")
ax[1,1].scatter(A[:,1], A[:,2])
ax[1,1].scatter(B[:,1], B[:,2])
ax[1,1].title.set_text("params = 2,3")
ax[2,0].scatter(A[:,1], A[:,3])
ax[2,0].scatter(B[:,1], B[:,3])
ax[2,0].title.set_text("params = 2,4")
ax[2,1].scatter(A[:,2], A[:,3])
ax[2,1].scatter(B[:,2], B[:,3])
ax[2,1].title.set_text("params = 3,4")
plt.show()
Y0 = np.zeros((N,1))
Y1 = np.ones((N,1))
Y = np.vstack((Y0,Y1))
X = np.vstack((A,B))
data = np.hstack((X,Y))
data_copy = data[:]

train_X, train_Y, test_X, test_Y = split(data_copy)

model = LogitRegression(0.01, 10000)
model.fit(normalize(train_X), train_Y, normalize(test_X), test_Y)

train_pred = model.predict(normalize(train_X))
test_pred = model.predict(normalize(test_X))

plt.plot(model.loss, label = "Train")
plt.plot(model.val_loss, label = "Test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.plot(model.acc_loss, label = "Train")
plt.plot(model.acc_val_loss, label = "Test")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show()
class_0 = train_pred.ravel()[np.where(train_Y == 0)[0]]
class_1 = train_pred.ravel()[np.where(train_Y == 1)[0]]

sns.histplot(class_0, legend=False, bins=25)
sns.histplot(class_1, legend=False, bins=25)
plt.show()

print(accuracy_score(train_Y, np.where( train_pred.ravel() > 0.5, 1, 0)), recall_score(train_Y, np.where( train_pred.ravel() > 0.5, 1, 0)), precision_score(train_Y, np.where( train_pred.ravel() > 0.5, 1, 0)), "\n")
print(accuracy_score(test_Y, np.where( test_pred.ravel() > 0.5, 1, 0)), recall_score(test_Y, np.where( test_pred.ravel() > 0.5, 1, 0)), precision_score(test_Y, np.where( test_pred.ravel() > 0.5, 1, 0)))