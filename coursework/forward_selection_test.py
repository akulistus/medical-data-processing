import pandas as pd
import numpy as np
import funcs as f
from normalizer import Normalizer
from sklearn.model_selection import train_test_split
from forward_selection import ForwardSelection
from log_reg import LogitRegression
from fisher import Fisher

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

features = train_normalized_base.columns.values.tolist()

# LogReg = LogitRegression(learning_rate=0.01, iterations=10000)
# FS = ForwardSelection(LogReg, 2)
# FS.fit(train_normalized_base, train_Y_normalized, test_normalized_base, test_Y_normalized)
# print(FS.selected_features)

fisher = Fisher()
FS = ForwardSelection(fisher, 3)
FS.fit(train_normalized_base, train_Y_normalized, test_normalized_base, test_Y_normalized)
print(FS.selected_features)