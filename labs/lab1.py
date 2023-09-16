import numpy as np

mean = 5
std = 2
N = 10
x1 = np.random.normal(mean, std, [1, N])

mean = 2
std = 1
x2 = np.random.normal(mean, std, [1, N])

print(x1)
print(x2)
x1_mean = np.mean(x1)
x2_mean = np.mean(x2)
x1_std = np.std(x1)
x2_std = np.std(x2)
c = np.corrcoef(x1, x2)
print(f"{x1_mean=}")
print(f"{x2_mean=}")
print(f"{x1_std=}")
print(f"{x2_std=}")
print(c)
