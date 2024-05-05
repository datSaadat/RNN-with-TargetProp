# -*- coding: utf-8 -*-
"""TPTT_nazmus_v1_noise.ipynb

# Commented out IPython magic to ensure Python compatibility.
"""
Some of the code was taken from: https://github.com/nmanchev/MM-TPTT-RNN/tree/main/task_a

This file implements TPTT using just numpy and manual gradient calculations for activation function
tanh.

"""

import sys
import numpy as np

import pandas as pd
from tempOrder import TempOrderTask
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

np.set_printoptions(precision=10, threshold=sys.maxsize, suppress=True)


class SRNN(object):

    def __init__(
        self,
        X,
        y,
        X_test,
        y_test,
        seq_length,
        n_hid,
        init,
        stochastic,
        hybrid,
        last_layer,
        noise,
        batch_size,
        M,
        rng,
        g_learning_rate,
        f_learning_rate,
        i_learning_rate,
        mem_noise,
    ):
        super(SRNN, self).__init__()

        self.n_inp = X.shape[2]  # [seq size n_inp]
        self.n_out = y.shape[1]  # [size n_out]
        self.M = M

        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test

        self.seq_length = seq_length
        self.n_hid = n_hid
        self.stochastic = stochastic
        self.hybrid = hybrid
        self.noise = noise
        self.last_layer = last_layer
        self.batch_size = batch_size
        self.rng = rng
        self.g_lr = g_learning_rate
        self.f_lr = f_learning_rate
        self.i_lr = i_learning_rate
        self.mem_noise = mem_noise

        # assert seq_length >= 10, "seq_length must be at least 10"

        self.h0 = np.zeros((self.batch_size, self.n_hid), np.float32)

        self.Wxh = self.rand_ortho(
            (self.n_hid, self.n_inp), np.sqrt(6.0 / (self.n_inp + self.n_hid))
        ).T
        self.Whh = self.rand_ortho(
            (self.n_hid, self.n_hid), np.sqrt(6.0 / (self.n_inp + self.n_hid))
        )
        self.Why = self.rand_ortho(
            (self.n_hid, self.n_out), np.sqrt(6.0 / (self.n_hid + self.n_out))
        )

        self.bh = np.zeros(self.n_hid)
        self.by = np.zeros(self.n_out)
        self.Vhh = self.rand_ortho(
            (self.n_hid, self.n_hid), np.sqrt(6.0 / (self.n_hid + self.n_hid))
        )
        self.ch = np.zeros(self.n_hid)
        self.activ = np.tanh
        self.params = OrderedDict()

        self.params["Wxh"] = self.Wxh
        self.params["Whh"] = self.Whh
        self.params["Why"] = self.Why
        self.params["bh"] = self.bh
        self.params["by"] = self.by
        self.params["Vhh"] = self.Vhh
        self.params["ch"] = self.ch

    def sftmx(self, x, axis=1):
        # Subtract the maximum value for numerical stability]
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def rand_ortho(self, shape, irange):
        """
        Generates an orthogonal matrix.

        Parameters
        ----------
        shape  : tuple
            Matrix shape
        irange : float
            Range for the matrix elements

        Returns
        -------
        numpy.ndarray
            An orthogonal matrix of size *shape*
        """
        A = -irange + (2 * irange * np.random.rand(*shape))
        U, _, V = np.linalg.svd(A)
        result = np.dot(U, np.dot(np.eye(U.shape[1], V.shape[0]), V))
        result = result.astype(np.float32)
        return result

    def _sample(self, x):
        return x

    def _f(self, x, hs):
        if self.stochastic:
            hs = self._sample(hs)
        z = self.activ(hs @ self.Whh + x @ self.Wxh + self.bh)
        return z

    def _g(self, x, hs):
        return self.activ(hs @ self.Vhh + x @ self.Wxh + self.ch)

    def _hidden(self, x):
        h = np.empty((self.seq_length, self.batch_size, self.n_hid), np.float32)
        h[0, :, :] = self._f(x[0, :, :], self.h0)

        for t in range(1, self.seq_length):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1].copy())
        return h

    @staticmethod
    def _cross_entropy(y_hat, y):
        return np.mean(np.sum(-y * np.log(y_hat), axis=1))

    @staticmethod
    def _mse(x, y):
        return np.mean((x - y) ** 2)

    def _gaussian(x, noise):
        return np.random.randn(*x.shape) * noise

    # def _mem_gaussian(x, mem_noise):
    #     print(x)
    #     return np.random.randn(np.shape(x)) * mem_noise

    def _get_targets(self, x, hs_tmax, h, cost, ilr, error):
        h_ = np.zeros((self.seq_length, self.batch_size, self.n_hid), np.float32)
        z = np.dot(error, (self.Why).T) / h.shape[1]
        h_[-1, :, :] = hs_tmax - ilr * z
        h_[-1, :, :] = h[-1, :, :] - hs_tmax + h_[-1, :, :]

        for t in range(self.seq_length - 2, -1, -1):
            h_[t] = (
                h[t]
                - self._g(x[t + 1, :, :], h[t + 1])
                + self._g(x[t + 1, :, :], h_[t + 1])
            )

        return h_

    def _calc_g_grads(self, x, h):
        grad_dVhh = np.zeros((self.seq_length, self.n_hid, self.n_hid), np.float32)
        grad_dch = np.zeros((self.seq_length, self.n_hid), np.float32)

        def targets_grads(h, x):
            per = h.shape[1]
            noise_shape = np.shape(h)
            noise = np.random.normal(0, 0.001, noise_shape)
            hp_with_noise_in_h = self._f(x, h + noise)
            h_cap_with_noise = self._g(x, hp_with_noise_in_h)
            h_cap_error = h_cap_with_noise - (h + noise)  # predictions - truth

            def relu_derivative(x):
                # return (x>0)*1-(x<=0)*0.01
                return (x > 0) * 1
                pass

            def tanh_derivative(x):
                return 1 - np.tanh(x) ** 2

            grad_G = (
                np.dot(
                    (2 * h_cap_error * tanh_derivative(h_cap_with_noise)).T,
                    hp_with_noise_in_h,
                ).T
                / per
            )
            grad_g = (
                np.sum(
                    2 * h_cap_error * tanh_derivative(h_cap_with_noise),
                    axis=0,
                    keepdims=True,
                )
                / per
            )
            return grad_G, grad_g

        for t in range(1, len(h)):
            grad_dVhh[t], grad_dch[t] = targets_grads(h[t], x[t, :, :])

        return np.sum(grad_dVhh, 0), np.sum(grad_dch, 0)

    def _calc_f_grads(self, x, h, h_, cost, out, target):
        # h_ -> target

        def forward_grads_final(hp, hp_cap, h):
            # hp_cap -> label
            # hp -> pred output

            pers = h.shape[1]
            hp_error = hp - hp_cap  # predictions - truth
            # self.Why.shape = (100,4)
            # h -> (100, 20), 20 is batch_size
            grad_F = np.dot(h, hp_error) / pers
            grad_f = np.sum(hp_error, axis=0, keepdims=True) / pers
            return grad_F, grad_f

        out_np = out  # .detach().numpy()
        target_np = target  # .detach().numpy()
        h_last_np = h[-1].T

        grad_dwhy, grad_dby = forward_grads_final(out_np, target_np, h_last_np)
        grad_dWhh = np.zeros((self.seq_length, self.n_hid, self.n_hid), np.float32)
        grad_dWxh = np.zeros((self.seq_length, self.n_inp, self.n_hid), np.float32)
        grad_dbh = np.zeros((self.seq_length, self.n_hid), np.float32)

        def forward_grads(hp, hp_cap, h):
            def tanh_derivative(x):
                return 1 - np.tanh(x) ** 2

            pers = h.shape[1]
            hp_error = hp - hp_cap
            grad_F = np.dot((2 * hp_error * tanh_derivative(hp)).T, h).T / pers
            grad_f = (
                np.sum(2 * hp_error * tanh_derivative(hp), axis=0, keepdims=True) / pers
            )

            return grad_F, grad_f

        for t in range(0, len(h)):
            # h -> output
            # h_ -> pred
            grad_dWhh[t], grad_dbh[t] = forward_grads(h[t], h_[t], h[t - 1])
            grad_dbh[t] = grad_dbh[t].flatten()
            grad_dWxh[t], _ = forward_grads(h[t], h_[t], x[t - 1])

        # return dWhh, dWxh, dbh, dwhy, dby
        return (
            np.sum(grad_dWhh, axis=0),
            np.sum(grad_dWxh, axis=0),
            np.sum(grad_dbh, axis=0),
            grad_dwhy,
            grad_dby.flatten(),
        )

    def forward(self, x, y):
        h = self._hidden(x)
        # new work implementation:
        # hs_tmax = self._sample(h[-1]).clone().detach().requires_grad_(True)
        # the above is what the code used for this work, not the OG DTP work we are replicating
        # hs_tmax = h[-1].clone().detach().requires_grad_(True) # in the context of the OG DTP work, hst_max = ht
        hs_tmax = h[-1].copy()  # .clone().detach().requires_grad_(True)
        out = hs_tmax @ self.Why + self.by
        if self.last_layer == "softmax":
            out = self.sftmx(out)
        elif self.last_layer == "linear":
            pass
        else:
            raise Exception("Unsupported classification type.")

        return hs_tmax, h, out

    def _validate(self, x):
        n_val_samples = x.shape[1]
        h0 = np.zeros((n_val_samples, self.n_hid), np.float32)
        h = np.empty((self.seq_length, n_val_samples, self.n_hid), np.float32)
        h[0, :, :] = self._f(x[0, :, :], h0)
        for t in range(1, self.seq_length):
            h[t, :, :] = self._f(x[t, :, :], h[t - 1].copy())

        out = h[-1] @ self.Why + self.by

        if self.last_layer == "softmax":
            out = self.sftmx(out)
        elif self.last_layer != "linear":
            raise Exception("Unsupported classification type.")

        return out

    def run_validation(self, x, y, avg_probs_100=True):

        valid_cost = 0
        valid_err = 0

        if self.stochastic and avg_probs_100:
            out = np.stack([self._validate(x) for i in range(100)]).mean(axis=0)
        else:
            out = self._validate(x)

        if self.last_layer == "softmax":
            valid_cost = self._cross_entropy(out, y)
            y = np.argmax(y, axis=1)
            y_hat = np.argmax(out, axis=1)
            valid_err = np.mean(y_hat != y)
        elif self.last_layer == "linear":
            valid_cost = self._mse(out, y).sum()
            valid_err = np.mean(np.sum((y - out) ** 2, axis=1) > 0.04)
        else:
            raise Exception("Unsupported classification type.")

        if isinstance(valid_err, np.ndarray):
            valid_err = valid_err.item()

        return valid_cost, valid_err

    def _step_g(self, x, y, g_optimizer):
        # g_optimizer.zero_grad()

        h = self._hidden(x)

        # Corrupt targets with noise
        if self.noise != 0:
            h = self._hidden(x)
            h = h + self._gaussian(h)

        def mem_gaussian(x, _mem_noise):
          return np.random.normal(0, _mem_noise*np.max(x), np.shape(x))

        Vhh_grad, ch_grad = self._calc_g_grads(x, h)

        self.Vhh = self.Vhh - Vhh_grad * self.g_lr + mem_gaussian(self.Vhh, self.mem_noise)
        self.ch = self.ch - ch_grad * self.g_lr +  mem_gaussian(self.ch, self.mem_noise)

        # g_optimizer.step()

    def _step_f(self, ilr, x, y, f_optimizer):
        # f_optimizer.zero_grad()

        out = np.zeros((self.batch_size, self.n_out), np.float32)
        for i in range(self.M):
            hs_tmax, h, out_ = self.forward(x, y)
            out = out + out_
        out = out / self.M

        if self.last_layer == "softmax":
            cost = self._cross_entropy(out, y)
        elif self.last_layer == "linear":
            cost = self._mse(out, y).sum()
        else:
            raise Exception("Unsupported classification type.")

        error = out - y
        # with torch.no_grad():
        h_ = self._get_targets(x, hs_tmax, h, cost, ilr, error)

        def mem_gaussian(x, _mem_noise):
          return np.random.normal(0, _mem_noise*np.max(x), np.shape(x))

        dWhh, dWxh, dbh, dwhy, dby = self._calc_f_grads(x, h, h_, cost, out, y)
        self.grad_Whh = dWhh
        self.grad_Wxh = dWxh
        self.grad_bh = dbh
        self.grad_by = dby
        self.grad_Why = dwhy
        self.Whh = self.Whh - self.f_lr * dWhh + mem_gaussian(self.Whh, self.mem_noise)
        self.Wxh = self.Wxh - self.f_lr * dWxh + mem_gaussian(self.Wxh, self.mem_noise)
        self.bh = self.bh - self.f_lr * dbh + mem_gaussian(self.bh, self.mem_noise)
        self.Why = self.Why - self.f_lr * dwhy + mem_gaussian(self.Why, self.mem_noise)
        self.by = self.by - self.f_lr * dby + mem_gaussian(self.by, self.mem_noise)
        # pre_dwhh = self.Whh
        # f_optimizer.step()
        return cost

    def fit(self, ilr, maxiter, g_optimizer, f_optimizer, task, rng, glr, flr, check_interval=1):

        training = True
        epoch = 1
        best = 0

        n_batches = self.X.shape[1] // self.batch_size
        accs = []
        while training & (epoch <= maxiter):

            if epoch == 1:
                # with torch.no_grad():
                _, best = self.run_validation(self.X_test, self.y_test)
                acc = 100 * (1 - best)
                print(
                    "Epoch -- \t Cost -- \t Test Acc: %.2f \t Highest: %.2f"
#                     % (acc, acc)
                )

            cost = 0
            # Inverse mappings
            for i in range(n_batches):
                batch_start_idx = i * self.batch_size
                batch_end_idx = batch_start_idx + self.batch_size
                x = self.X[:, batch_start_idx:batch_end_idx, :]
                y = self.y[batch_start_idx:batch_end_idx, :]
                self._step_g(x, y, g_optimizer)

            # Forward mappings
            for i in range(n_batches):
                batch_start_idx = i * self.batch_size
                batch_end_idx = batch_start_idx + self.batch_size
                x = self.X[:, batch_start_idx:batch_end_idx, :]
                y = self.y[batch_start_idx:batch_end_idx, :]

                cost += self._step_f(ilr, x, y, f_optimizer)
                if np.isnan(cost):
                    print("Cost is NaN. Aborting....")
                    training = False
                    break

            cost = cost / n_batches
            if epoch % check_interval == 0:
                # with torch.no_grad():
                valid_cost, valid_err = self.run_validation(self.X_test, self.y_test)

                print_str = "It: {:10s}\tLoss: %.3f\t".format(str(epoch)) % cost

                whh_grad_np = self.Whh
                vhh_grad_np = self.Vhh
                if np.isnan(whh_grad_np).any():
                    print_str += "ρ|Whh|: -----\t"
                else:
                    print_str += "ρ|Whh|: %.3f\t" % np.max(
                        abs(np.linalg.eigvals(whh_grad_np))
                    )

                if np.isnan(vhh_grad_np).any():
                    print_str += "ρ|Vhh|: -----\t"
                else:
                    print_str += "ρ|Vhh|: %.3f\t" % np.max(
                        abs(np.linalg.eigvals(vhh_grad_np))
                    )

                dWhh = np.linalg.norm(self.grad_Whh)
                dWxh = np.linalg.norm(self.grad_Wxh)
                dWhy = np.linalg.norm(self.grad_Why)

                acc = 100 * (1 - valid_err)

                if acc > best:
                    best = acc

                print_str += "dWhh: %.5f\t dWxh: %.5f\t dWhy: %.5f\t" % (
                    dWhh,
                    dWxh,
                    dWhy,
                )
                print_str += "Acc: %.2f\tVal.loss: %.2f\tHighest: %.2f\t" % (
                    acc,
                    valid_cost,
                    best,
                )
                print_str += "ρ|val_err|: %.3f\t" % (valid_err * 100)
                accs.append(acc)
                print(print_str)

                if valid_err < 0.0001:
                    print("PROBLEM SOLVED.")
                    training = False
            # print(f"Epoch: {epoch}")
            epoch += 1
        # breakpoint()

        import pickle
        from datetime import datetime

        date = datetime.now()
        with open(
            f"accuracy_data_{date}_no_auto_grad_no_torch_{maxiter}_{ilr}_{glr}_{flr}.pkl", "wb"
        ) as f:
            pickle.dump(accs, f)
        return best, cost.item()


def sample_length(min_length, max_length, rng):
    """
    Computes a sequence length based on the minimal and maximal sequence size.

    Parameters
    ----------
    max_length      : maximal sequence length (t_max)
    min_length      : minimal sequence length

    Returns
    -------
    A random number from the max/min interval
    """
    length = min_length

    if max_length > min_length:
        length = min_length + rng.randint(max_length - min_length)

    return length


def run_experiment(
    seed,
    init,
    task_name,
    opt,
    seq,
    hidden,
    stochastic,
    hybrid,
    batch,
    maxiter,
    i_learning_rate,
    f_learning_rate,
    g_learning_rate,
    noise,
    M,
    mem_noise,
    check_interval=10,
):

    # torch.manual_seed(seed)
    model_rng = np.random.RandomState(seed)
    rng = model_rng

    last_layer = "softmax"
    task = "MNIST"
    sample_train = 60000
    sample_test = 10000
    X, y, X_test, y_test = load_MNIST(
        "mnist_8x8",
        one_hot=True,
        norm=False,
        sample_train=sample_train,
        sample_test=sample_test,
    )

    model = SRNN(
        X,
        y,
        X_test,
        y_test,
        seq,
        hidden,
        init,
        stochastic,
        hybrid,
        last_layer,
        noise,
        batch,
        M,
        model_rng,
        g_learning_rate,
        f_learning_rate,
        i_learning_rate,
        mem_noise,
    )

    print("SRNN TPTT Network")
    print("--------------------")
    print("stochastic : %s" % stochastic)
    if stochastic:
        print("MCMC       : %i" % M)
        print("Hybrid     : %s" % hybrid)
    print("task name  : %s" % task_name)
    print("train size : %i" % (X.shape[1]))
    print("test size  : %i" % (X_test.shape[1]))
    print("batch size : %i" % batch)
    print("T          : %i" % seq)
    print("n_hid      : %i" % hidden)
    # print("init       : %s" % init.__name__)
    print("maxiter    : %i" % maxiter)
    print("chk        : %i" % check_interval)
    print("--------------------")
    print("optimiser : %s" % opt)
    print("ilr       : %.5f" % i_learning_rate)
    print("flr       : %.5f" % f_learning_rate)
    print("glr       : %.5f" % g_learning_rate)
    if noise != 0:
        print("noise     : %.5f" % noise)
    else:
        print("noise     : ---")
    print("--------------------")

    g_optimizer = None
    f_optimizer = None
    val_acc, tr_cost = model.fit(
        i_learning_rate, maxiter, g_optimizer, f_optimizer, task, rng, g_learning_rate,
        f_learning_rate, check_interval
    )
    # file_name = "rnn_stptt_" + "t" + str(seq) + "_taskA_i" \
    #             + str(i_learning_rate) + "_f" + str(f_learning_rate) + "_g" \
    #             + str(g_learning_rate) + "_" + init.__name__  + opt.lower()

    # model.plot_classA(file_name + ".png")

    return val_acc, tr_cost


def load_MNIST(data_folder, one_hot=False, norm=True, sample_train=0, sample_test=0):
    """
    Loads, samples (if needed), and one-hot encodes the MNIST data set.

    Parameters
    ----------
    data_folder   : location of the MNIST data
    one_hot       : if True the target labels will be one-hot encoded
    norm          : if True the images will be normalised
    sample_train  : fraction of the train data to use. if set to 0 no sampling
                    will be applied (i.e. 100% of the data is used)
    sample_train  : fraction of the test data to use. if set to 0 no sampling
                    will be applied (i.e. 100% of the data is used)
    Returns
    -------
    X_train - Training images. Dimensions are (784, number of samples, 1)
    y_train - Training labels. Dimensions are (number of samples, 10)
    X_test  - Test images. Dimensions are (784, number of samples, 1)
    y_test  - Test labels. Dimensions are (number of samples, 10)
    """
    """
    X_train = np.genfromtxt(
        "%s/train_X.csv" % data_folder, delimiter=",", dtype=np.float32
    )
    y_train = np.asarray(
        np.fromfile("%s/train_Y.csv" % data_folder, sep="\n"), dtype="int32"
    )

    X_test = np.genfromtxt(
        "%s/test_X.csv" % data_folder, delimiter=",", dtype=np.float32
    )
    y_test = np.asarray(
        np.fromfile("%s/test_Y.csv" % data_folder, sep="\n"), dtype="int32"
    )"""

    """Hung-Ta's Dataset"""

    loaded_data = np.load('/content/drive/My Drive/Codes/MNIST8_group.npy', allow_pickle=True).item()
    X_train = loaded_data['train_X']
    y_train = loaded_data['train_y']
    X_test = loaded_data['test_X']
    y_test = loaded_data['test_y']

    if (sample_train != 0) and (sample_test != 0):

        print("Elements in train : %i" % sample_train)
        print("Elements in test  : %i" % sample_test)

        idx_train = np.random.choice(
            np.arange(len(X_train)), sample_train, replace=False
        )
        idx_test = np.random.choice(np.arange(len(X_test)), sample_test, replace=False)

        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

    if norm:
        print("MNIST NORMALISED!")
        X_train /= 255.0
        X_test /= 255.0

    # Swap axes
    X_train = np.swapaxes(np.expand_dims(X_train, axis=0), 0, 2)
    X_test = np.swapaxes(np.expand_dims(X_test, axis=0), 0, 2)

    # Encode the target labels
    # breakpoint()
    if one_hot:
        onehot_encoder = OneHotEncoder(sparse_output=False, categories="auto")
        y_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = onehot_encoder.fit_transform(y_test.reshape(-1, 1))
    return X_train, y_train, X_test, y_test


def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = 0

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .01

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .0001

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .0002

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .0003

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .0004

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

def main():
    batch = 16
    hidden = 100
    maxiter = 25
    i_learning_rate = 0.0000001
    f_learning_rate = 0.025
    g_learning_rate = 0.00000001
    noise = 0.0
    M = 1
    mem_noise = .0005

    seed = 1234

    # init = nn.init.orthogonal_
    init = None
    sto = False
    hybrid = True  # set to false for no stochasticity

    # Experiment 1 - shallow depth
    seq = 64
    #rng = np.random.RandomState(1234)
    np.random.seed(seed)
    run_experiment(
        seed,
        init,
        "task_A",
        "SGD",
        seq,
        hidden,
        sto,
        hybrid,
        batch,
        maxiter,
        i_learning_rate,
        f_learning_rate,
        g_learning_rate,
        noise,
        M,
        mem_noise,
        check_interval=1,
    )

    # # Experiment 2 - deeper network
    # seq = 30

    # run_experiment(seed, init, "task_A", "Adagrad", seq, hidden, sto, hybrid, batch, maxiter,
    #                i_learning_rate, f_learning_rate, g_learning_rate, noise, M, check_interval=100)


if __name__ == "__main__":
    main()

