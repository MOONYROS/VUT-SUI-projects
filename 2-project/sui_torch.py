import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = str(self.back_op.__class__.__name__)
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        if deltas is not None:
            assert deltas.shape == self.value.shape, f'Expected gradient with shape {
                self.value.shape}, got {deltas.shape}'

            raise NotImplementedError(
                'Backpropagation with deltas not implemented yet')
        else:
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(
                    f'Can only backpropagate a scalar, got shape {self.shape}')

            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.back_op.backward()


class SumBackward:
    def __init__(self, input_tensor):
        self.input = input_tensor

    def backward(self, grad_output):
        self.input.grad += np.full(self.input.shape, grad_output)

        if self.input.back_op:
            self.input.back_op.backward(self.input.grad)


class AddBackward:
    def __init__(self, tensor_a, tensor_b):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

    def backward(self, grad_output):
        self.tensor_a.grad += grad_output
        self.tensor_b.grad += grad_output

        if self.tensor_a.back_op:
            self.tensor_a.back_op.backward(self.tensor_a.grad)
        if self.tensor_b.back_op:
            self.tensor_b.back_op.backward(self.tensor_b.grad)  


class SubtractBackward:
    def __init__(self, tensor_a, tensor_b):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

    def backward(self, grad_output):
        self.tensor_a.grad += grad_output
        self.tensor_b.grad -= grad_output

        if self.tensor_a.back_op:
            self.tensor_a.back_op.backward(self.tensor_a.grad)
        if self.tensor_b.back_op:
            self.tensor_b.back_op.backward(self.tensor_b.grad)  


class MultiplyBackward:
    def __init__(self, tensor_a, tensor_b):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b

    def backward(self, grad_output):
        self.tensor_a.grad += grad_output * self.tensor_b.value
        self.tensor_b.grad += grad_output * self.tensor_a.value

        if self.tensor_a.back_op:
            self.tensor_a.back_op.backward(self.tensor_a.grad)
        if self.tensor_b.back_op:
            self.tensor_b.back_op.backward(self.tensor_b.grad)  


class ReLUBackward:
    def __init__(self, input_tensor):
        self.input = input_tensor

    def backward(self, grad_output):
        relu_grad = grad_output * (self.input.value > 0)
        self.input.backward(relu_grad)

class DotProductBackward:
    def __init__(self, a, b):
        self.tensor_a = a
        self.tensor_b = b

    def backward(self, grad_output):
        self.tensor_a.grad += np.matmul(grad_output, self.tensor_b.value.T)
        self.tensor_b.grad += np.matmul(self.tensor_a.value.T, grad_output)

        if self.tensor_a.back_op:
            self.tensor_a.back_op.backward(self.tensor_a.grad)
        if self.tensor_b.back_op:
            self.tensor_b.back_op.backward(self.tensor_b.grad)  

def sui_sum(tensor: Tensor):
    return Tensor(value=np.array([[np.sum(tensor.value)]]), back_op=SumBackward(tensor))


def add(a: Tensor, b: Tensor):
    if a.shape != b.shape:
        raise ValueError(
            f'Cannot add tensors with shapes {a.shape} and {b.shape}')

    return Tensor(value=(a.value + b.value), back_op=AddBackward(a, b))


def subtract(a: Tensor, b: Tensor):
    if a.shape != b.shape:
        raise ValueError(
            f'Cannot subtract tensors with shapes {a.shape} and {b.shape}')

    return Tensor(value=(a.value - b.value), back_op=SubtractBackward(a, b))


def multiply(a: Tensor, b: Tensor):
    if a.shape != b.shape:
        raise ValueError(
            f'Cannot multiply tensors with shapes {a.shape} and {b.shape}')

    return Tensor(value=(a.value * b.value), back_op=MultiplyBackward(a, b))


def relu(tensor: Tensor):
    return Tensor(value=np.maximum(tensor.value, 0), back_op=ReLUBackward(tensor))


def dot_product(a: Tensor, b: Tensor):
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            # zmenit
            f'Cannot multiply tensors with shapes {a.shape} and {b.shape}')
    return Tensor(value=np.matmul(a.value, b.value), back_op=DotProductBackward(a, b))
