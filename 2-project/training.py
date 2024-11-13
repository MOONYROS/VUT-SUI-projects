#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np

import sui_torch as st


class LinearLayer:
    def __init__(self, out_features, in_features):
        self.weights = st.Tensor(np.random.randn(out_features, in_features))
        self.bias = st.Tensor(np.random.randn(out_features)[:, None])

    def forward(self, x):
        return st.add(st.dot_product(self.weights, x), self.bias)

    def parameters(self):
        return [self.weights, self.bias]


class LinearModel:
    def __init__(self, in_features, out_features):
        self.out = LinearLayer(out_features, in_features)

    def forward(self, x):
        x = self.out.forward(x)
        return x

    def parameters(self):
        return self.out.parameters()


class NonLinearModel:
    def __init__(self, in_features, hidden_features, out_features):
        self.hidden = LinearLayer(hidden_features, in_features)
        self.out = LinearLayer(out_features, hidden_features)

    def forward(self, x):
        x = self.hidden.forward(x)
        x = st.relu(x)
        x = self.out.forward(x)
        return x

    def parameters(self):
        return self.out.parameters() + self.hidden.parameters()


def MSELoss(predicted, target):
    diff = st.subtract(predicted, target)
    return st.sui_sum(st.multiply(diff, diff))


class StochasticGradientDescent:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for tensor in self.tensors:
            tensor.value -= self.lr * tensor.grad

    def zero_grad(self):
        for tensor in self.tensors:
            tensor.grad.fill(0)


def train_model(model, xs, ys, opt, nb_epochs):
    no_improvement = 0
    best_loss = float('inf')
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        for x, y in zip(xs, ys):
            opt.zero_grad()
            x = st.Tensor(np.array([[x]]).T)
            y = st.Tensor(np.array([[y]]).T)
            predicted = model.forward(x)
            loss = MSELoss(predicted, y)
            epoch_loss += loss.value
            loss.backward()
            opt.step()

        epoch_loss /= len(xs)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > 5:
                break


def evaluate_model(model, xs, ys):
    total_loss = 0.0
    for x, y in zip(xs, ys):
        x = st.Tensor(np.array([[x]]).T)
        y = st.Tensor(np.array([[y]]).T)
        predicted = model.forward(x)
        loss = MSELoss(predicted, y)
        total_loss += loss.value

    return total_loss/len(xs)


def sample_data(nb_samples):
    xs = np.linspace(-1, 2, nb_samples) + np.random.randn(nb_samples) * 0.1
    function = lambda x: 2*x*x - 1
    ys = function(xs) + np.random.randn(nb_samples) * 0.1

    return xs, ys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--nb-epochs', type=int, default=200)
    parser.add_argument('--do-plots', action='store_true')
    return parser.parse_args()


def plot_model(model, name):
    xs = np.linspace(-1, 2, 100)
    ys = [model.forward(st.Tensor(np.array([[x]]).T)).value.squeeze() for x in xs]

    plt.plot(xs, ys, label=name)


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    train_xs, train_ys = sample_data(40)
    eval_xs, eval_ys = sample_data(10)

    m1 = LinearModel(1, 1)
    opt1 = StochasticGradientDescent(m1.parameters(), lr=1e-4)
    train_model(m1, train_xs, train_ys, opt1, nb_epochs=args.nb_epochs)
    print('Train loss:', evaluate_model(m1, train_xs, train_ys),
          'eval:', evaluate_model(m1, eval_xs, eval_ys))

    m2 = NonLinearModel(1, 5, 1)
    opt2 = StochasticGradientDescent(m2.parameters(), lr=1e-5)
    train_model(m2, train_xs, train_ys, opt2, nb_epochs=args.nb_epochs)
    print('Train loss:', evaluate_model(m2, train_xs, train_ys),
          'eval:', evaluate_model(m2, eval_xs, eval_ys))

    m3 = NonLinearModel(1, 20, 1)
    opt3 = StochasticGradientDescent(m3.parameters(), lr=1e-4)
    train_model(m3, train_xs, train_ys, opt3, nb_epochs=args.nb_epochs)
    print('Train loss:', evaluate_model(m3, train_xs, train_ys),
          'eval:', evaluate_model(m3, eval_xs, eval_ys))

    m4 = NonLinearModel(1, 50, 1)
    opt4 = StochasticGradientDescent(m4.parameters(), lr=1e-4)
    train_model(m4, train_xs, train_ys, opt4, nb_epochs=args.nb_epochs)
    print('Train loss:', evaluate_model(m4, train_xs, train_ys),
          'eval:', evaluate_model(m4, eval_xs, eval_ys))

    if args.do_plots:
        plt.figure()
        plt.scatter(train_xs, train_ys, label='train')
        plt.scatter(eval_xs, eval_ys, label='eval')
        plot_model(m1, 'Linear')
        plot_model(m2, 'NonLinear 5')
        plot_model(m3, 'NonLinear 20')
        plot_model(m4, 'NonLinear 50')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
