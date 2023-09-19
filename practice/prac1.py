import numpy as np
import matplotlib.pyplot as plt
from random import randrange


def create_linear_array(b_0:int = 5, b_1:int = 3, N:int = 500) -> tuple:
    x_r = np.random.randint(1,10, size=N)
    x = np.cumsum(x_r)
    err = np.random.randn(N) * 1000
    res = b_0 + b_1 * x + err

    return(x, res)

def create_squared_array(a:int = -5, b:int = 3, c:int = 5, N:int = 500) -> tuple:
    x_r = np.random.randint(1,10, size=N)
    x = np.cumsum(x_r)
    print(x)
    err = np.random.randn(N) * 1000000
    res = a*(x**2) +b*x + c + err
    
    return (x,res)


def lin_reg(x,y):
    b1 = (np.mean(x*y) - np.mean(x)*np.mean(y))/(np.mean(x**2)-np.mean(x)**2)
    b0 = np.mean(y) - b1*np.mean(x)
    lenreg_y = b0 + b1*x
    
    return lenreg_y


x, y= create_linear_array()
lenreg_y = lin_reg(x,y)
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.show()

x, y= create_squared_array()
lenreg_y = lin_reg(x,y)
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.show()