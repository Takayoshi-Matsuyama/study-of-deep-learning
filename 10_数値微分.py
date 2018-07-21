import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt2
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_1(x):
    return 0.01*x**2 + 0.1*x


def function_2(x):
    return x[0]**2 + x[1]**2


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)


        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 値を元に戻す

    return grad


# 勾配降下法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


x = np.arange(0.0,  20.0, 0.1)  # 0から20まで、0.1刻みのx配列
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)
plt.plot(x, y)
plt.plot(x, y2)
plt.show()

#print(numerical_diff(function_1, 5))
#print(numerical_diff(function_1, 10))


fig = plt2.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = x2 = np.arange(-3, 3, 0.1)
X1, X2 = np.meshgrid(x1, x2)

Z = np.power(X1, 2) + np.power(X2, 2)

print(Z)

ax.plot_wireframe(X1, X2, Z, rstride=2, cstride=2)

plt2.xlabel("x0")
plt2.ylabel("x1")
plt2.show()

# x0 = 3, x1 = 4 のときのx0に対する偏微分

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))


# x0 = 3, x1 = 4 のときのx1に対する偏微分

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1

print(numerical_diff(function_tmp2, 4.0))


# 勾配の計算

print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# 問：f(x0, x1) = x0^2 + x1^2 の最小値を勾配法で求めよ
def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(f=function_2, init_x=init_x, lr=0.1, step_num=100))

# 学習率が大きすぎる例：lr = 10.0
init_x = np.array([-3.0, 4.0])
print(gradient_descent(f=function_2, init_x=init_x, lr=10.0, step_num=100))

# 学習率が小さすぎる例：lr = 1e-10
init_x = np.array([-3.0, 4.0])
print(gradient_descent(f=function_2, init_x=init_x, lr=1e-10, step_num=100))


'''
arg = np.c_[X1.ravel(), X2.ravel()]

print(x1)
print(x2)
print(X1)
print(X2)
print(arg)
'''

# 【Python】ふたつの配列からすべての組み合わせを評価
# http://kaisk.hatenadiary.com/entry/2014/11/05/041011

"""
Matplotlibのmplot3dで3Dグラフを作成
https://note.nkmk.me/python-matplotlib-mplot3d/

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-15, 15, 0.5)
X, Y = np.meshgrid(x, y)

sigma = 4
Z = np.exp(-(X**2 + Y**2)/(2*sigma**2)) / (2*np.pi*sigma**2)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
plt.savefig("data/dst/matplotlib_mplot3d_surface.png")

ax.clear()
ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
plt.savefig("data/dst/matplotlib_mplot3d_wireframe.png")

ax.clear()
ax.scatter(X, Y, Z, s=1)
plt.savefig("data/dst/matplotlib_mplot3d_scatter.png")
"""