import numpy as np

def create_linear_array(b_0:int = 5, b_1:int = 3, N:int = 500) -> tuple:
    x_r = np.random.randint(1,10, size=N)
    x = np.cumsum(x_r)
    err = np.random.randn(N) * 1000
    res = b_0 + b_1 * x + err

    return(x, res)

def create_squared_array(a:int = -5, b:int = 3, c:int = 5, N:int = 500) -> tuple:
    x_r = np.random.randint(1,10, size=N)
    x = np.cumsum(x_r)
    err = np.random.randn(N) * 1000000
    res = a*(x**2) +b*x + c + err
    
    return (x,res)

def lin_reg(x,y):
    b1 = (np.mean(x*y) - np.mean(x)*np.mean(y))/(np.mean(x**2)-np.mean(x)**2)
    b0 = np.mean(y) - b1*np.mean(x)
    lenreg_y = b0 + b1*x
    
    return lenreg_y

def remains(y_norm: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_norm - y_pred

def SSE(y_norm: np.ndarray, y_pred: np.ndarray) -> float:
    tmp = (y_norm - y_pred)**2
    return sum(tmp)

def SST(y_norm: np.ndarray, y_pred: np.ndarray) -> float:
    y_mean = np.mean(y_norm)
    tmp = (y_pred - y_mean)**2
    return sum(tmp)

def R_squared(y_norm: np.ndarray, y_pred: np.ndarray) -> float:
    _SSE = SSE(y_norm, y_pred)
    _SST = SST(y_norm, y_pred)
    return 1 -(_SSE/_SST)

def RSS(e:np.ndarray) -> float:
    tmp = e**2
    return sum(tmp)

def SpirmenCoeff(x: np.ndarray, e: np.ndarray):
    re = sorted(e)
    rx = sorted(x)
    n = len(x)
    d = []

    for i in range(n):
        ind_rx = rx.index(x[i])
        ind_re = re.index(e[i])
        d.append(ind_rx - ind_re)
        rx[ind_rx] = None
        re[ind_re] = None
        
    d = np.array(d)
    d = d ** 2
    r = 1-(6*sum(d)/(n*((n**2)-1)))
    t = r*np.sqrt(n-2)/np.sqrt(1-r**2)
    return np.abs(t)

def GoldfeldQuandtTest(e: np.ndarray, k: int) -> float:
    k_1 = e[:k]
    k_2 = e[len(e)-k:]
    return RSS(k_2)/RSS(k_1)