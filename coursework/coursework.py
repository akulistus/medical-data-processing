import funcs as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fisher import Fisher
from normalizer import Normalizer
from log_reg import LogitRegression
from sklearn.decomposition._pca import PCA
from sklearn.metrics import accuracy_score, recall_score
from forward_selection import ForwardSelection
from sklearn.model_selection import train_test_split

#Create header row
header = []
for i in range(0, 16):
    header.append(f"param_{i}")
header.append("class")

#Read data
data = pd.read_csv('./coursework/var_14.txt', sep=" ", names=header)
train, test = train_test_split(data, train_size=0.8, shuffle=True, stratify=data["class"])
train_X, train_Y = f.split_result(train)
test_X, test_Y = f.split_result(test)

#Normalize data
norm = Normalizer()
train_normalized = norm.normalize(train_X)
test_normalized = (test_X-norm.mean)/norm.std

#Remove outliers from test using same quantils?
Q1 = train_normalized.quantile(0.25, axis=0)
Q3 = train_normalized.quantile(0.75, axis=0)
IQR = Q3.subtract(Q1)

upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR

train_normalized_base = train_normalized[(train_normalized < upper_bound).all(axis=1)]
train_normalized_base = train_normalized[(train_normalized > lower_bound).all(axis=1)]
train_Y_normalized = train_Y[train_normalized_base.index]

test_normalized_base = test_normalized[(test_normalized < upper_bound).all(axis=1)]
test_normalized_base = test_normalized[(test_normalized > lower_bound).all(axis=1)]
test_Y_normalized = test_Y[test_normalized_base.index]

#Remove columns with corr > 0.9
train_normalized_no_corr, test_normalized_no_corr = f.delete_corr(train_normalized_base, test_normalized_base, 0.9)

#PCA
PCA_obj = PCA(n_components=3)
PCA_obj.fit(train_X)
class_1 = train_X[train_Y == 1]
class_2 = train_X[train_Y == 0]
class_1_PCA = pd.DataFrame(PCA_obj.transform(class_1))
class_2_PCA = pd.DataFrame(PCA_obj.transform(class_2))
train_PCA = pd.DataFrame(PCA_obj.transform(train_X))
train_PCA.index = train_Y.index
#cause np.exp can not handle large numbers
train_PCA = train_PCA.divide(100)
test_PCA = pd.DataFrame(PCA_obj.transform(test_X))
test_PCA.index = test_Y.index
test_PCA = test_PCA.divide(100)

#Plot 3D scatter for three devidable feachers
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(class_1_PCA[0], class_1_PCA[1], class_1_PCA[2], label = "class_1")
ax.scatter(class_2_PCA[0], class_2_PCA[1], class_2_PCA[2], label = "class_2")
ax.set_title("PCA")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
class_1 = train_normalized_no_corr[train_Y_normalized == 1]
class_2 = train_normalized_no_corr[train_Y_normalized == 0]
ax.scatter(class_1["param_10"],
                class_1["param_14"],
                class_1["param_15"], label = "class_1")
ax.scatter(class_2["param_10"],
                class_2["param_14"],
                class_2["param_15"], label = "class_2")
ax.set_title("train_normalized_no_corr")
ax.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
class_1 = train_normalized_base[train_Y_normalized == 1]
class_2 = train_normalized_base[train_Y_normalized == 0]
ax.scatter(class_1["param_10"],
                class_1["param_14"],
                class_1["param_15"], label = "class_1")
ax.scatter(class_2["param_10"],
                class_2["param_14"],
                class_2["param_15"], label = "class_2")
ax.set_title("train_normalized_base")
ax.legend()
plt.show()

#Histograms
f.plot_hist(train_normalized_no_corr, train_Y_normalized)
params = ["param_5", "param_11", "param_10","param_12","param_13"]
train_normalized_base.drop(params, axis=1, inplace=True)
test_normalized_base.drop(params, axis=1, inplace=True)

# Log_reg
LogReg = LogitRegression(learning_rate=0.01, iterations=10000)
fig, ax = plt.subplots(3,3)
LogReg.fit(np.array(train_PCA), np.array(train_Y), np.array(test_PCA), np.array(test_Y))
ax[0,0].plot(LogReg.acc_loss, label = "Train")
ax[0,0].plot(LogReg.acc_val_loss, label = "Test")
ax[0,0].set_xlabel("Epochs")
ax[0,0].set_ylabel("Acc")
ax[0,0].set_title("PCA")
ax[0,0].legend()
pred_train = LogReg.predict(np.array(train_PCA))
pred_test = LogReg.predict(np.array(test_PCA))
print(accuracy_score(train_Y.ravel(), pred_train.ravel()), recall_score(train_Y.ravel(), pred_train.ravel()), recall_score(train_Y.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y.ravel(), pred_test.ravel()), recall_score(test_Y.ravel(), pred_test.ravel()), recall_score(test_Y.ravel(), pred_test.ravel(), pos_label=0),"\n")


ax[1,0].plot(LogReg.loss, label = "Train")
ax[1,0].plot(LogReg.val_loss, label = "Test")
ax[1,0].set_xlabel("Epochs")
ax[1,0].set_ylabel("Loss")
ax[1,0].legend()

pred_test = LogReg.predict(np.array(test_PCA), False)
class_0 = pred_test.ravel()[np.where(test_Y == 0)[0]]
class_1 = pred_test.ravel()[np.where(test_Y == 1)[0]]
ax[2,0].hist(class_0, label = "class_0")
ax[2,0].hist(class_1, label = "class_1")
ax[2,0].set_ylabel("Quantity")
ax[2,0].legend()


LogReg.fit(np.array(train_normalized_base), np.array(train_Y_normalized), np.array(test_normalized_base), np.array(test_Y_normalized))
ax[0,1].plot(LogReg.acc_loss, label = "Train")
ax[0,1].plot(LogReg.acc_val_loss, label = "Test")
ax[0,1].set_xlabel("Epochs")
ax[0,1].set_title("base")
ax[0,1].legend()
pred_train = LogReg.predict(np.array(train_normalized_base))
pred_test = LogReg.predict(np.array(test_normalized_base))
print(accuracy_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel(), pos_label=0),"\n")

ax[1,1].plot(LogReg.loss, label = "Train")
ax[1,1].plot(LogReg.val_loss, label = "Test")
ax[1,1].set_xlabel("Epochs")
ax[1,1].legend()

pred_test = LogReg.predict(np.array(test_normalized_base), False)
class_0 = pred_test.ravel()[np.where(test_Y_normalized == 0)[0]]
class_1 = pred_test.ravel()[np.where(test_Y_normalized == 1)[0]]
ax[2,1].hist(class_0, label = "class_0")
ax[2,1].hist(class_1, label = "class_1")
ax[2,1].legend()

LogReg.fit(np.array(train_normalized_no_corr), np.array(train_Y_normalized), np.array(test_normalized_no_corr), np.array(test_Y_normalized))
ax[0,2].plot(LogReg.acc_loss, label = "Train")
ax[0,2].plot(LogReg.acc_val_loss, label = "Test")
ax[0,2].set_xlabel("Epochs")
ax[0,2].set_title("no_corr")
ax[0,2].legend()
pred_train = LogReg.predict(np.array(train_normalized_no_corr))
pred_test = LogReg.predict(np.array(test_normalized_no_corr))
print(accuracy_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel(), pos_label=0),"\n")

ax[1,2].plot(LogReg.loss, label = "Train")
ax[1,2].plot(LogReg.val_loss, label = "Test")
ax[1,2].set_xlabel("Epochs")
ax[1,2].legend()

pred_test = LogReg.predict(np.array(test_normalized_base), False)
class_0 = pred_test.ravel()[np.where(test_Y_normalized == 0)[0]]
class_1 = pred_test.ravel()[np.where(test_Y_normalized == 1)[0]]
ax[2,2].hist(class_0, label = "class_0")
ax[2,2].hist(class_1, label = "class_1")
ax[2,2].legend()

plt.show()

# Fisher
fish = Fisher()
fig, ax = plt.subplots(1,3)
fish.fit(np.array(train_PCA), np.array(train_Y))
res = fish.predict(np.array(test_PCA), False)
class_0 = res.ravel()[np.where(test_Y == 0)[0]]
class_1 = res.ravel()[np.where(test_Y == 1)[0]]
ax[0].hist(class_0, label = "class_0")
ax[0].hist(class_1, label = "class_1")
ax[0].set_title("PCA")
ax[0].legend()
print("Fisher\n")
pred_train = fish.predict(np.array(train_PCA))
pred_test = fish.predict(np.array(test_PCA))
print(accuracy_score(train_Y.ravel(), pred_train.ravel()), recall_score(train_Y.ravel(), pred_train.ravel()), recall_score(train_Y.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y.ravel(), pred_test.ravel()), recall_score(test_Y.ravel(), pred_test.ravel()), recall_score(test_Y.ravel(), pred_test.ravel(), pos_label=0),"\n")


fish.fit(np.array(train_normalized_base), np.array(train_Y_normalized))
res = fish.predict(np.array(test_normalized_base), False)
class_0 = res.ravel()[np.where(test_Y_normalized == 0)[0]]
class_1 = res.ravel()[np.where(test_Y_normalized == 1)[0]]
ax[1].hist(class_0, label = "class_0")
ax[1].hist(class_1, label = "class_1")
ax[1].set_title("Base")
ax[1].legend()
pred_train = fish.predict(np.array(train_normalized_base))
pred_test = fish.predict(np.array(test_normalized_base))
print(accuracy_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel(), pos_label=0),"\n")


fish.fit(np.array(train_normalized_no_corr), np.array(train_Y_normalized))
res = fish.predict(np.array(test_normalized_no_corr), False)
class_0 = res.ravel()[np.where(test_Y_normalized == 0)[0]]
class_1 = res.ravel()[np.where(test_Y_normalized == 1)[0]]
ax[2].hist(class_0, label = "class_0")
ax[2].hist(class_1, label = "class_1")
ax[2].set_title("No_corr")
ax[2].legend()
pred_train = fish.predict(np.array(train_normalized_no_corr))
pred_test = fish.predict(np.array(test_normalized_no_corr))
print(accuracy_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel()), recall_score(train_Y_normalized.ravel(), pred_train.ravel(), pos_label=0),"\n")
print(accuracy_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel()), recall_score(test_Y_normalized.ravel(), pred_test.ravel(), pos_label=0),"\n")
plt.show()