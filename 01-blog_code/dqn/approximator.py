# /usr/local/bin/python3.7
# -*- coding:utf-8 -*-
# function approximators of reinforcment learning

import numpy as np
import torch
from torch.autograd import Variable
import copy


class Approximator(torch.nn.Module):
    '''base class of different function approximator subclasses
    '''

    def __init__(self, dim_input=1, dim_output=1, dim_hidden=16):
        super(Approximator, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        # function Linear:__init(inputSize, outputSize)
        # hidden layer
        self.linear1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.linear2 = torch.nn.Linear(self.dim_hidden, self.dim_output)

    def predict(self, x):
        # 实现ReLU:->max(0, x)
        # torch.clamp(input,min,max,out=None)-> Tensor
        # 将input中的元素限制在[min,max]范围内并返回一个Tensor
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

    def fit(self, x,
            y,
            criterion=None,
            optimizer=None,
            epochs=1,
            learning_rate=1e-4):
        if criterion is None:
            # MSELoss(reduce=False, size_average=False)
            # 如果 reduce = False，那么 size_average 参数失效，直接返回向量形式的 loss；
            # 如果 reduce = True，那么 loss 返回的是标量
            #   如果 size_average = True，返回 loss.mean();
            #   如果 size_average = False，返回 loss.sum();
            criterion = torch.nn.MSELoss(size_average=False)
        if optimizer is None:
            # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if epochs < 1:
            epochs = 1

        x = self._prepare_data(x)
        y = self._prepare_data(y, False)

        for t in range(epochs):
            y_pred = self.predict(x)
            loss = criterion(y_pred, y)
            # 把梯度置零，也就是把loss关于weight的导数变成0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss

    def _prepare_data(self, x, requires_grad=True):
        '''将numpy格式的数据转化为Torch的Variable
        '''
        if isinstance(x, np.ndarray):
            x = Variable(torch.from_numpy(x), requires_grad=requires_grad)
        if isinstance(x, int):
            x = Variable(torch.Tensor([[x]]), requires_grad=requires_grad)
        # 从from_numpy()转换过来的数据是DoubleTensor形式
        x = x.float()
        if x.data.dim() == 1:
            # 增加一个纬度
            x = x.unsqueeze(0)
        return x

    def __call__(self, x):
        '''根据输入返回输出，类似于 predict 函数
        '''
        x = self._prepare_data(x)
        pred = self.predict(x)
        return pred.data.numpy

    def clone(self):
        '''返回当前模型的深度拷贝对象
        '''
        return copy.deepcopy(self)


def test():
    N, D_in, H, D_out = 64, 100, 50, 1
    # torch.rand(*sizes, out=None) → Tensor
    # 返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。
    # torch.randn(*sizes, out=None) → Tensor,服从标准正态分布
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad=False)

    model = Approximator(D_in, D_out, H)

    model.fit(x, y, epochs=1000)
    print(x[2])
    y_pred = model.predict(x[2])
    print(y[2])
    print(y_pred)
    new_model = model.clone()
    new_pred = new_model.predict(x[2])
    print(new_pred)
    print(model is new_model)


if __name__ == "__main__":
    test()
