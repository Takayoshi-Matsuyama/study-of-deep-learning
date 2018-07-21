import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
from collections import OrderedDict

import matplotlib.pylab as plt


class Relu:
    def __init__(self):
        self.mask = None


    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out


    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out


    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None


    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out


    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss


    def backward(self, dout=1):
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


# 手書き数字データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 2層ニューラルネットワーク
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10);

train_loss_list = []

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
    print(i)

x = np.arange(0, len(train_loss_list), 1)
plt.plot(x, train_loss_list)
plt.show()

