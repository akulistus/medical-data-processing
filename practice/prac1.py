import matplotlib.pyplot as plt
import src.funcs as f

x, y= f.create_linear_array()
lenreg_y = f.lin_reg(x,y)
e = f.remains(y, lenreg_y)
sse = round(f.SSE(y, lenreg_y),2)
R_sq = round(f.R_squared(y, lenreg_y),2)
# RSS = f.RSS(e)
gold = f.GoldfeldQuandtTest(e, 125)
spir = f.SpirmenCoeff(x, e)
print("Тест ранговой корреляции Спирмена: {}\nТест Голдфелда Квандта {}".format(spir, gold))
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.title(f"{sse=},{R_sq=}")
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.scatter(x,y, c='#00ced1')
ax1.plot(x,lenreg_y,  c='#ff1493')
ax2.scatter(x,e, c='#00ced1')
ax3.hist(e)
plt.show()


x, y= f.create_squared_array()
lenreg_y = f.lin_reg(x,y)
e = f.remains(y, lenreg_y)
sse = round(f.SSE(y, lenreg_y),2)
R_sq = round(f.R_squared(y, lenreg_y),2)
# RSS = f.RSS(e)
gold = f.GoldfeldQuandtTest(e, 125)
spir = f.SpirmenCoeff(x, e)
print("Тест ранговой корреляции Спирмена: {}\nТест Голдфелда Квандта {}".format(spir, gold))
plt.scatter(x,y, c='#00ced1')
plt.plot(x,lenreg_y,  c='#ff1493')
plt.title(f"{sse=},{R_sq=}")
plt.show()

fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.scatter(x,y, c='#00ced1')
ax1.plot(x,lenreg_y,  c='#ff1493')
ax2.scatter(x,e, c='#00ced1')
ax3.hist(e)
plt.show()
