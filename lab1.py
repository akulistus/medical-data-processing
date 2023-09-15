import numpy as np
import matplotlib.pyplot as plt
mean1 = 5
std1 = 2
N = 10
x1 = np.random.normal(mean1, std1, [1, N])

mean2 = 2
std2 = 1
x2 = np.random.normal(mean2, std2, [1, N])
print(x1)
print(x2)
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
x1_std = np.mean(x1)
x2_std = np.mean(x2)
c = np.corrcoef(x1, x2)
print(x1_mean)
print(x2_mean)
print(x1_std)
print(x2_std)
print(c)