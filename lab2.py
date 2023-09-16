import numpy as np
import matplotlib.pyplot as plt

mean = 5
std = 2
N = 10
data_x = np.random.normal(mean, std, [1, N])

mean = 2
std = 1
N = 10
data_y = np.random.normal(mean, std, [1, N])

plt.scatter(data_x, data_y, marker='*')
plt.title('Скатерограмма')
plt.xlabel('Значение х')
plt.ylabel('Значение y')
plt.show()

hist_data = np.vstack((data_x, data_y))
plt.hist(hist_data.T)
plt.xlabel('Значение х')
plt.ylabel('Количество попавших значений')
plt.title('Гистограмма')
plt.show()

plt.boxplot([data_x.ravel(), data_y.ravel()], notch=True)
plt.xlabel('Блок')
plt.ylabel('Значения величины')
plt.title('Коробочковая диаграмма')
plt.show()
