
class SGD:
    """Stochastic Gradient Descent (確率的勾配降下法)"""

    def __init__(self, lr=0.01):
        """初期化

        Args:
            lr: Learning Rate (学習係数)
        """
        self.lr = lr


    def update(self, params, grads):
        """パラメータの更新

        Args:
            params: ディクショナリ変数
            grads: ディクショナリ変数

        Returns: なし

        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]

