import unittest
from numpy.testing import assert_allclose

import numpy as np
import sui_torch as st


class TestDotProduct(unittest.TestCase):
    def test_vectors_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.0, 3.0]]))
        b = st.Tensor(np.array([[0.5, 0.5, 1.0]]).T)

        c = st.dot_product(a, b)
        expected = np.array([[2.5]])
        assert_allclose(c.value, expected)

    def test_vectors_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.0, 3.0]]))
        b = st.Tensor(np.array([[0.5, 0.5, 1.0]]).T)

        c = st.dot_product(a, b)  # 2.5
        c.backward()

        e_a_grad = np.array([[0.5, 0.5, 1.0]])  # c * b
        assert_allclose(a.grad, e_a_grad)

        e_b_grad = np.array([[1.0, -2.0, 3.0]]).T  # c * a
        assert_allclose(b.grad, e_b_grad)

    def test_matrix_vector_fwd(self):
        a = st.Tensor(np.array([[1.0, -1.0], [0.0, 1.0]]))
        b = st.Tensor(np.array([[2.0, 3.0]]).T)

        c = st.dot_product(a, b)
        expected = np.array([[-1.0, 3.0]]).T
        assert_allclose(c.value, expected)

    def test_matrix_vector_bwd(self):
        a = st.Tensor(np.array([[1.0, -1.0], [0.0, 1.0]]))
        b = st.Tensor(np.array([[2.0, 3.0]]).T)

        c = st.dot_product(a, b)
        c.backward(np.array([[1., 1.]]).T)

        e_b_grad = np.array([[1.0, 0.0]]).T
        assert_allclose(b.grad, e_b_grad)


class TestAggregation(unittest.TestCase):
    def test_sum_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))

        b = st.sui_sum(a)
        expected = np.array([[1.5]])
        assert_allclose(b.value, expected)

    def test_sum_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]).T)

        b = st.sui_sum(a)
        b.backward()

        e_a_grad = np.array([[1.0, 1.0, 1.0]]).T
        assert_allclose(a.grad, e_a_grad)


class TestElementWise(unittest.TestCase):
    def test_add_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]))

        c = st.add(a, b)
        expected = np.array([[2.0, -0.5, 6.0]])
        assert_allclose(c.value, expected)

    def test_add_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]).T)
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]).T)

        c = st.add(a, b)
        deltas = np.array([[1.0, 2.0, 3.0]]).T
        c.backward(deltas)

        e_a_grad = np.array([[1.0, 2.0, 3.0]]).T
        assert_allclose(a.grad, e_a_grad)

        e_b_grad = np.array([[1.0, 2.0, 3.0]]).T
        assert_allclose(b.grad, e_b_grad)

    def test_subtract_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]))

        c = st.subtract(a, b)
        expected = np.array([[0.0, -4.5, 0.0]])
        assert_allclose(c.value, expected)

    def test_subtract_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]).T)
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]).T)

        c = st.subtract(a, b)
        deltas = np.array([[1.0, 2.0, 3.0]]).T
        c.backward(deltas)

        e_a_grad = np.array([[1.0, 2.0, 3.0]]).T
        assert_allclose(a.grad, e_a_grad)

        e_b_grad = np.array([[-1.0, -2.0, -3.0]]).T
        assert_allclose(b.grad, e_b_grad)

    def test_product_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]))

        c = st.multiply(a, b)
        expected = np.array([[1.0, -5.0, 9.0]])
        assert_allclose(c.value, expected)

    def test_product_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]))

        c = st.multiply(a, b)
        deltas = np.array([[1.0, 2.0, 1.0]])
        c.backward(deltas)

        e_a_grad = np.array([[1.0, 4.0, 3.0]])
        e_b_grad = np.array([[1.0, -5.0, 3.0]])

        assert_allclose(a.grad, e_a_grad)
        assert_allclose(b.grad, e_b_grad)


class TestActivation(unittest.TestCase):
    def test_relu_fwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]))

        b = st.relu(a)
        expected = np.array([[1.0, 0.0, 3.0]])
        assert_allclose(b.value, expected)

    def test_relu_bwd(self):
        a = st.Tensor(np.array([[1.0, -2.5, 3.0]]).T)

        b = st.relu(a)
        deltas = np.array([[1.0, 2.0, 4.0]]).T
        b.backward(deltas)

        e_a_grad = np.array([[1.0, 0.0, 4.0]]).T
        assert_allclose(a.grad, e_a_grad)


class TestBackpropagation(unittest.TestCase):
    def test_relu_sum(self):
        a = st.Tensor(np.array([[1.0, -2.0, 3.0]]).T)
        b = st.relu(a)

        c = st.sui_sum(b)
        c.backward()

        e_a_grad = np.array([[1.0, 0.0, 1.0]]).T

        assert_allclose(a.grad, e_a_grad)

    def test_add_sum(self):
        a = st.Tensor(np.array([[1.0, -2.0, 3.0]]).T)
        b = st.Tensor(np.array([[1.0, 2.0, 3.0]]).T)

        c = st.add(a, b)
        d = st.sui_sum(c)
        d.backward()

        e_a_grad = np.array([[1.0, 1.0, 1.0]]).T
        e_b_grad = np.array([[1.0, 1.0, 1.0]]).T

        assert_allclose(a.grad, e_a_grad)
        assert_allclose(b.grad, e_b_grad)
