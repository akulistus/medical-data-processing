import funcs as f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from normalizer import Normalizer
from models import LogitRegression, Fisher
from sklearn.decomposition._pca import PCA
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

print(test[test_Y == 1]*(-1))
print(test)

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
train_Y = train_Y[train_normalized_base.index]

test_normalized_base = test_normalized[(test_normalized < upper_bound).all(axis=1)]
test_normalized_base = test_normalized[(test_normalized > lower_bound).all(axis=1)]
test_Y = test_Y[test_normalized_base.index]

#Remove columns with corr > 0.9
train_normalized_no_corr, test_normalized_no_corr = f.delete_corr(train_normalized_base, test_normalized_base, 0.9)

#PCA
PCA_obj = PCA(n_components=3)
PCA_obj.fit(train)
train_PCA = pd.DataFrame(PCA_obj.transform(train))
test_PCA = pd.DataFrame(PCA_obj.transform(test))

#Plot 3D scatter for three devidable feachers
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train_PCA[0], train_PCA[1], train_PCA[2])
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train_normalized_no_corr["param_10"],
                train_normalized_no_corr["param_14"],
                train_normalized_no_corr["param_15"])
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(train_normalized_base["param_10"],
                train_normalized_base["param_14"],
                train_normalized_base["param_15"])
plt.show()

#Histograms
f.plot_hist(train_normalized_no_corr)

#add column of ones
train_normalized_no_corr_ones, test_normalized_no_corr_ones = f.add_ones(train_normalized_no_corr, test_normalized_no_corr)

# Log_reg
LogReg = LogitRegression(learning_rate=0.01, iterations=10000)
LogReg.fit(np.array(train_normalized_no_corr_ones), np.array(train_Y), np.array(test_normalized_no_corr_ones), np.array(test_Y))
plt.plot(LogReg.acc_loss, label = "Train")
plt.plot(LogReg.acc_val_loss, label = "Test")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.show()

# Fisher
fish = Fisher()
fish.fit(train_normalized_no_corr, train_Y)
proj1, proj2 = fish.predict(train_normalized_no_corr, train_Y)
plt.hist(proj1)
plt.hist(proj2)
plt.show()