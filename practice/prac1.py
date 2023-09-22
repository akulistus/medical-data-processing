import matplotlib.pyplot as plt
import src.funcs as f

x, y= f.create_linear_array()
lenreg_y = f.lin_reg(x,y)
e = f.remains(y, lenreg_y)
RSS = f.RSS(e)
gold = f.GoldfeldQuandtTest(e, 125)
spir = f.SpirmenCoeff(x, e)
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.title(f"{gold=},{RSS=}, {spir=}")
plt.show()

x, y= f.create_squared_array()
lenreg_y = f.lin_reg(x,y)
e = f.remains(y, lenreg_y)
RSS = f.RSS(e)
gold = f.GoldfeldQuandtTest(e, 125)
spir = f.SpirmenCoeff(x, e)
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.title(f"{gold=},{RSS=}, {spir=}")
plt.show()
