import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from collections import OrderedDict

import matplotlib.pylab as plt

class MulLayer:
    """乗算レイヤ"""

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y  # xとyをひっくり返す
        dy = dout * self.x
        return dx, dy


class AddLayer:
    """加算レイヤ"""

    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    """Rectified Linear Unit"""
    
    def __init__(self):
        self.mask = None


    def forward(self, x):
        """ReLU 順伝播

        Args:
            x: 入力データのNumpy配列

        Returns: ReLU値のNumpy配列

        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out


    def backward(self, dout):
        """ReLU 逆伝播

        Args:
            dout: ReLU値のNumpy配列

        Returns: ReLU 逆伝播値のNumpy配列

        """
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None


    def forward(self, x):
        """Sigmoid 順伝播

        Args:
            x: 入力データのNumpy配列

        Returns: Sigmoid値のNumpy配列

        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out


    def backward(self, dout):
        """Sigmoid 逆伝播

        Args:
            dout: Sigmoid値のNumpy配列

        Returns: Sigmoid 逆伝播のNumpy配列

        """
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        """Affine変換(行列の積)のレイヤ：初期化

        Args:
            W: 重み (Weight)を表すNumpy2次元配列
            b: バイアス (Bias)を表すNumpy配列
        """
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        """Affine変換 順伝播

        Args:
            x: 入力データのNumpy配列

        Returns: 行列の積の結果のNumpy配列

        """
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out


    def backward(self, dout):
        """Affine変換 逆伝播

        Args:
            dout: 行列の積の結果のNumpy配列

        Returns: 逆伝播のNumpy配列

        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  # 0番目の軸(データを単位とした軸)に対して総和を求める
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        """ SoftmaxWithLoss 順伝播
            Softmaxレイヤは、入力データを受け取り、正規化して出力
            Cross Entropy Errorレイヤは、Softmaxレイヤの出力と教師ラベルを受け取り、損失を出力

        Args:
            x: 入力データのNumpy配列
            t: 教師ラベルのNumpy配列

        Returns: 損失のNumpy配列

        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss


    def backward(self, dout=1):
        """ SoftmaxWithLoss 逆伝播

        Args:
            dout: 損失のNumpy配列

        Returns: 逆伝播のNumpy配列
        　　　　　Softmaxレイヤの出力yと教師ラベルtの差

        """
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class TwoLayerNet:
    """ 2層ニューラルネット """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """クラスインスタンスを初期化する。

        Args:
            input_size: 入力層のニューロンの数
            hidden_size: 隠れ層のニューロンの数
            output_size: 出力層のニューロンの数
            weight_init_std: 初期化時に重みを一律に調整するための係数
        """

        # 重みの初期化(ガウス分布に従う乱数を使い、バイアスは0で初期化する)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        """認識(推論)を行う。

        Args:
            x: 入力データ

        Returns: 認識(推論)の結果

        """
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
        """

        for layer in self.layers.values():
            x = layer.forward(x)

        return x


    def loss(self, x, t):
        """損失関数(交差エントロピー誤差)の値を求める。

        Args:
            x: 入力データ
            t: 教師データ

        Returns: 損失関数(交差エントロピー誤差)の値

        """
        y = self.predict(x)

        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        """認識精度を求める。

        Args:
            x: 入力データ
            t: 教師データ

        Returns: 認識精度

        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def numerical_gradient(self, x, t):
        """重みパラメータの勾配を求める

        Args:
            x: 入力データ
            t: 教師データ

        Returns: 勾配

        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

# -----------------------------------------------------
# リンゴの買い物
print("リンゴの買い物")

apple = 100
apple_num = 2
tax = 1.1

# リンゴの買い物 - layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# リンゴの買い物 - forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# リンゴの買い物 - backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)


# -----------------------------------------------------
# リンゴ2個とみかん3個の買い物
print("リンゴ2個とみかん3個の買い物")
apple = 100
app_num = 2
orange = 150
orange_num = 3
tax = 1.1

# リンゴ2個とみかん3個の買い物 - layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# リンゴ2個とみかん3個の買い物 - forward
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)

# リンゴ2個とみかん3個の買い物 - backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple_num, dapple, dorange, dorange_num, dtax)


# 手書き数字データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1  # 学習率

# 2層ニューラルネットワーク
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10);

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1エポックあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.gradient(x_batch, t_batch)
    #grad = network.numerical_gradient(x_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1エポック毎に認識精度を計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    #print(i)

x = np.arange(0, len(train_loss_list), 1)
plt.plot(x, train_loss_list)
plt.show()

epochs = np.arange(0, len(train_acc_list), 1)
plt.plot(epochs, train_acc_list)
plt.plot(epochs, test_acc_list)
plt.show()
