import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from normalizer import Normalizer
from sklearn.model_selection import train_test_split

#Create header row
header = []
for i in range(0, 16):
    header.append(f"param_{i}")
header.append("class")

#Read data
data = pd.read_csv('./coursework/var_14.txt', sep=" ", names=header)
train, test = train_test_split(data, train_size=0.8, shuffle=True, stratify=data["class"])

#Normalize data
norm = Normalizer()
train_normalized = norm.normalize(train)
test_normalized = (test-norm.mean)/norm.std

#Remove outliers from test using same quantils?
Q1 = train_normalized.quantile(0.25, axis=0)
Q3 = train_normalized.quantile(0.75, axis=0)
IQR = Q3.subtract(Q1)
print(Q3)

upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR

train_normalized = train_normalized[(train_normalized < upper_bound).all(axis=1)]
train_normalized = train_normalized[(train_normalized > lower_bound).all(axis=1)]

