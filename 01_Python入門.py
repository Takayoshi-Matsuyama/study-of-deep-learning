# 算術計算
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)

# データ型
print(type(10))
print(type(2.718))
print(type("hello"))

# 変数
x = 10
print(x)
x = 100
print(x)
y = 3.14
print(x * y)
print(type(x * y))

# リスト
a = [1, 2, 3, 4, 5]
print(a)
print(a[0])
print(a[4])
a[4] = 99
print(a)
print(a[0:2])
print(a[1])
print(a[:3])
print(a[:-1])
print(a[:-2])

# ディクショナリ
me = {'height' : 180}
print(me['height'])
me['weight'] = 70
print(me)

# ブーリアン
hungry = True
sleepy = False
print(type(hungry))
print(not hungry)
print(hungry and sleepy)
print(hungry or sleepy)

# if文
hungry = True
if hungry:
    print("I'm hungry")

# for文
for i in [1,2,3]:
    print(i)

# 関数
def hello():
    print("Hello World!")

def hello(object):
    print("Hello " + object + "!")

hello("cat")

# クラス
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

# NumPy

import numpy as np
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x/2.0) # ブロードキャスト: NumPy配列の各要素とスカラ値との間で計算が行われる。

# NumPyのN次元配列

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

print(A)
print(A * 10) #ブロードキャスト

# NumPy ブロードキャスト

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

# NumPy 要素へのアクセス
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten() # Xを1次元の配列へ変換
print(X)
print(X[np.array([0, 2, 4])]) # インデックスが0, 2, 4番目の要素を取得

print(X > 15)
print(X[X > 15])

# Matplotlib

# 単純なグラフの描画

import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# pyplotの機能

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

# 画像の表示

from matplotlib.image import imread

img = imread('./dataset/lena.png')
plt.imshow(img)
plt.show()
