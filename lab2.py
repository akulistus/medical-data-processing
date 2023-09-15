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
fig, axs = plt.subplots(2,2)
axs[0,0].plot(data_x, data_y)
axs[0,0].set_xlabel('Значение х')
axs[0,0].set_ylabel('Значение y')
axs[0,0].set_title('y = x ** 2')
axs[1, 0].scatter(data_x, data_y, marker='*')
axs[1, 0].set_xlabel('Значение х')
axs[1, 0].set_ylabel('Значение y')
axs[1, 0].set_title('Скатерограмма')
hist_data = np.vstack((data_x, data_y))
axs[0, 1].hist(hist_data.T)
axs[0, 1].set_xlabel('Значение х')
axs[0, 1].set_ylabel('Количество попавших значений')
axs[0, 1].set_title('Гистограмма')
axs[1, 1].boxplot((data_x.ravel(), data_y.ravel()), notch=True)
axs[1, 1].set_xlabel('Блоки')
axs[1, 1].set_ylabel('Значения величины')
axs[1, 1].set_title('Коробочковая диаграмма')
plt.show()