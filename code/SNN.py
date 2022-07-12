import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_data):  # input_data = membrane potential- threshold
        ctx.save_for_backward(input_data)
        return input_data.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        lens = 0.5
        gamma = 0.5

        input_data, = ctx.saved_tensors
        grad_input = grad_output.clone()
        scale = 6.0
        height = .15
        temp = utils.gaussian(input_data, mu=0., sigma=lens) * (1. + height)
        - utils.gaussian(input_data, mu=lens, sigma=scale * lens) * height
        - utils.gaussian(input_data, mu=-lens,
                   sigma=scale * lens) * height
        return grad_input * temp.float() * gamma

act_fun_adp = ActFun_adp.apply


# Hyperparameters to optimize:
#  * b_j0: initial threshold
#  * beta: size of threshold adaption
#  * dt: time period (in ms)
#  * R_m: membrane resistance
def mem_update_adp(inputs, mem, spike, tau_adp, tau_m, b, device, b_j0=0.01,
                   dt=1.0, isAdapt=1, R_m=1):
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    ro = torch.exp(-1. * dt / tau_adp).to(device)

    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b

def tensor_with_new_grad(x):
    # return torch.tensor(x.detach().numpy(), requires_grad=True)
    return x.detach()

class SNNLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, is_recurrent, size_in, size_out, device, b_j0, dt=1.0, R_m=1, sparsity=0.0):
        super().__init__()
        self.device = device
        self.dt = dt
        self.b_j0 = b_j0
        self.R_m = R_m
        self.sparsity = sparsity
        self.is_recurrent = is_recurrent

        self.size_in, self.size_out = size_in, size_out
        self.in_connection = nn.Linear(self.size_in, self.size_out)
        self.tau_adp = nn.Parameter(torch.Tensor(self.size_out))
        self.tau_m = nn.Parameter(torch.Tensor(self.size_out))
        nn.init.xavier_uniform_(self.in_connection.weight)

        nn.init.constant_(self.in_connection.bias, 0)
        nn.init.normal_(self.tau_adp, 700, 25)
        if self.is_recurrent:
            nn.init.normal_(self.tau_m, 20, 5)
        else:
            nn.init.normal_(self.tau_m, 20, 1)
        self.b = 0

        if self.sparsity != 0.0:
            self.in_weight_mask = nn.Parameter(
                (torch.rand(self.in_connection.weight.shape) > sparsity).float())

        if self.is_recurrent:
            self.recurrent_connection = nn.Linear(self.size_out, self.size_out)
            nn.init.xavier_uniform_(self.recurrent_connection.weight)
            nn.init.constant_(self.recurrent_connection.bias, 0)
            if self.sparsity != 0.0:
                self.rec_weight_mask = nn.Parameter(
                    (torch.rand(self.recurrent_connection.weight.shape) > sparsity).float())

    def reset_weight_sparsity(self):
        if self.sparsity != 0.0:
            with torch.no_grad():
                self.in_connection.weight *= self.in_weight_mask
                if self.is_recurrent:
                    self.recurrent_connection.weight *= self.rec_weight_mask

    def get_base_parameters(self):
        base_parameters = [self.in_connection.weight, self.in_connection.bias]
        if self.is_recurrent:
            base_parameters += [self.recurrent_connection.weight,
                                self.recurrent_connection.bias]
        return base_parameters

    def init_model_state(self, batch_size):
        self.b = self.b_j0
        self.mem = torch.zeros(
            batch_size, self.size_out).to(self.device)
        self.spike = torch.zeros(
            batch_size, self.size_out).to(self.device)

    def detach_state(self):
        self.mem= tensor_with_new_grad(self.mem)
        self.b = tensor_with_new_grad(self.b)
        self.spike = tensor_with_new_grad(self.spike)

    def forward(self, x):
        input = self.in_connection(x.float())
        if self.is_recurrent:
            input += self.recurrent_connection(self.spike)
        self.mem, self.spike, _, self.b = mem_update_adp(input, self.mem,
            self.spike, self.tau_adp, self.tau_m, self.b, self.device,
            dt=self.dt, b_j0=self.b_j0, R_m=self.R_m)

        return self.spike


class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, device, b_j0, dt=1.0, R_m=1, sparsity=0.0):
        super().__init__()
        self.device = device
        self.sparsity = sparsity
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        hidden_dims = [int(x / math.sqrt(1 - sparsity)) for x in hidden_dims]
        layer_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(1, len(layer_dims)):
            layers.append(SNNLayer(True, layer_dims[i - 1], layer_dims[i],
                device, b_j0, dt=dt, R_m=R_m, sparsity=sparsity))

        # Append non-recurrect output layer
        layers.append(SNNLayer(False, layer_dims[-1], output_dim,
                device, b_j0, dt=dt, R_m=R_m, sparsity=sparsity))

        self.layers = nn.Sequential(*layers)

        if sparsity != 0.0:
            self.reset_weight_sparsity()


    def init_model_state(self, batch_size):
        for layer in self.layers:
            layer.init_model_state(batch_size)

        self.output_sumspike = torch.zeros(
            batch_size, self.output_dim).to(self.device)

    def detach_state(self):
        for layer in self.layers:
            layer.detach_state()
        self.output_sumspike = self.output_sumspike.detach()

    def reset_weight_sparsity(self):
        for layer in self.layers:
            layer.reset_weight_sparsity()

    def get_base_parameters(self):
        base_parameters = []
        for layer in self.layers:
            base_parameters += layer.get_base_parameters()
        return base_parameters

    def get_neuron_count(self):
        return sum(self.hidden_dims) + self.output_dim


    def forward(self, input_data):
        batch_size, sequence_size, _ = input_data.shape
        self.init_model_state(batch_size)

        for i in range(sequence_size):
            self.output_sumspike += self.layers(input_data[:, i, :])
        return self.output_sumspike


    def forward_TBPTT(self, input_data, backprop_step, kernel_size=(1, 4),
                     stride=(1, 1), encoding="rate", time_per_window=0,
                     neurons_per_pixel=0, train=False, labels=None,
                     criterion=None, optimizer=None):
        batch_size, sequence_size, _ = input_data.shape

        self.init_model_state(batch_size)
        for i in range(sequence_size):
            self.output_sumspike += self.layers(input_data[:, i, :])
            if train and i > 0 and i % backprop_step == 0:
                loss = criterion(self.output_sumspike, labels)
                # Getting gradients w.r.t. parameters
                loss.backward()
                # Updating parameters
                optimizer.step()
                self.reset_weight_sparsity()

                self.detach_state()
                optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        return self.output_sumspike, loss
