import funcs as f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from normalizer import Normalizer
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

# fig, ax = plt.subplots(4,4)
# param = "param_15"
# ax[0,0].scatter(train[f"param_1"], train[param])
# ax[0,1].scatter(train[f"param_2"], train[param])
# ax[0,2].scatter(train[f"param_3"], train[param])
# ax[0,3].scatter(train[f"param_4"], train[param])

# ax[1,0].scatter(train[f"param_5"], train[param])
# ax[1,1].scatter(train[f"param_6"], train[param])
# ax[1,2].scatter(train[f"param_7"], train[param])
# ax[1,3].scatter(train[f"param_8"], train[param])

# ax[2,0].scatter(train[f"param_9"], train[param])
# ax[2,1].scatter(train[f"param_10"], train[param])
# ax[2,2].scatter(train[f"param_11"], train[param])
# ax[2,3].scatter(train[f"param_12"], train[param])

# ax[3,0].scatter(train[f"param_13"], train[param])
# ax[3,1].scatter(train[f"param_14"], train[param])
# ax[3,2].scatter(train[f"param_15"], train[param])
# ax[3,3].scatter(train[f"param_0"], train[param])
# plt.show()

#Normalize data
norm = Normalizer()
train_normalized = norm.normalize(train)
test_normalized = (test-norm.mean)/norm.std

#Remove outliers from test using same quantils?
Q1 = train_normalized.quantile(0.25, axis=0)
Q3 = train_normalized.quantile(0.75, axis=0)
IQR = Q3.subtract(Q1)

upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR

train_normalized_base = train_normalized[(train_normalized < upper_bound).all(axis=1)]
train_normalized_base = train_normalized[(train_normalized > lower_bound).all(axis=1)]

test_normalized_base = test_normalized[(test_normalized < upper_bound).all(axis=1)]
test_normalized_base = test_normalized[(test_normalized > lower_bound).all(axis=1)]

#Remove columns with corr > 0.9
train_normalized_no_corr, test_normalized_no_corr = f.delete_corr(train_normalized_base, test_normalized_base, 0.9)

#PCA
PCA_obj = PCA(n_components=3)
PCA_obj.fit(train)
train_PCA = pd.DataFrame(PCA_obj.transform(train))
test_PCA = pd.DataFrame(PCA_obj.transform(test))

#Plot 3D scatter for three devidable feachers
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(train_PCA[0], train_PCA[1], train_PCA[2])
# ax.scatter(train_normalized_no_corr["param_5"], 
#                 train_normalized_no_corr["param_11"], #3
#                 train_normalized_no_corr["param_15"]) #5
# ax.scatter(train_normalized_base["param_5"],
#                 train_normalized_base["param_11"],
#                 train_normalized_base["param_15"])
# plt.show()