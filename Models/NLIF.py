import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class NLIF(nn.Module):
    # W, U, I_o, O
    free_parameters = ['w', 'W_in', 'I_o', 'O']  # 0,2,3,5,8
    # parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8], 'tau_g': [5., 6.]}
    parameter_init_intervals = {'w': [0., 1.], 'W_in': [0., 1.], 'I_o': [0.2, 0.6], 'O': [0.5, 2.]}

    def __init__(self, N=30, w_mean=0.4, w_var=0.9):
        super(NLIF, self).__init__()
        # self.device = device

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros((self.N,))
        self.s_fast = torch.zeros_like(self.v)  # syn. conductance

        rand_ws_syn = (w_mean - w_var) + 2 * w_var * torch.randn((self.N, self.N))
        rand_ws_fast = (w_mean - w_var) + (2) * w_var * torch.randn((self.N, self.N))
        rand_ws_in = (w_mean - w_var) + (2) * w_var * torch.randn((2, self.N))
        I_o = torch.ones((N,))
        # nt = T(neuron_types).float()
        # self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)

        self.W_syn = nn.Parameter(FT(rand_ws_syn.clamp(-1., 1.)), requires_grad=True)
        self.W_fast = nn.Parameter(FT(rand_ws_fast.clamp(-1., 1.)), requires_grad=True)
        self.W_in = nn.Parameter(FT(rand_ws_in.clamp(-1., 1.)), requires_grad=True)  # "U" - input weights
        self.I_o = nn.Parameter(FT(I_o), requires_grad=True)  # tonic input current
        self.O = nn.Parameter(torch.ones((2, N)), requires_grad=True)  # linear readout

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)  # only used for W_syn
        # self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v_reset = 0.
        self.tau_m = 10.
        self.tau_s = 10.
        # self.tau_s_fast = 1.

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.W_syn.register_hook(lambda grad: static_clamp_for_matrix(grad, -1., 1., self.W_syn))
        self.W_fast.register_hook(lambda grad: static_clamp_for_matrix(grad, -1., 1., self.W_fast))
        self.W_in.register_hook(lambda grad: static_clamp_for_matrix(grad, -1., 1., self.W_in))
        self.O.register_hook(lambda grad: static_clamp_for_matrix(grad, -1., 1., self.O))
        self.I_o.register_hook(lambda grad: static_clamp_for(grad, -1., 1., self.I_o))

    def get_parameters(self):
        params_list = {}

        params_list['W_syn'] = self.W_syn.data
        params_list['W_syn_fast'] = self.W_syn_fast.data
        params_list['W_in'] = self.W_in.data
        params_list['I_o'] = self.O.data
        params_list['O'] = self.O.data

        params_list['tau_m'] = self.tau_m
        params_list['tau_s'] = self.tau_s

        return params_list

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros_like(self.v)  # syn. conductance
        self.s_fast = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()
        self.s_fast = self.s_fast.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, x_in):
        # (WIP) Correct NIF formulation
        I_syn = (self.s).matmul(self.W_syn * self.self_recurrence_mask)  # should be pos w recurrence too
        I_fast_syn = self.s_fast.matmul(self.W_fast)
        I_in = x_in.matmul(self.W_in)

        I_tot = I_syn + I_fast_syn + I_in + self.I_o
        dv = (I_tot - self.v) / self.tau_m
        v_next = torch.add(self.v, dv)

        gating = v_next.clamp(-1., 1.)
        ds = (gating * dv.clamp(-1., 1.) - self.s) / self.tau_s  # TODO: ensure integrals = 1
        self.s = self.s + ds
        ds_fast = (gating * dv.clamp(-1., 1.) - self.s_fast)
        self.s_fast = self.s_fast + ds_fast

        spiked_pos = (v_next >= 1.).float()
        spiked_neg = (v_next <= -1.).float()
        not_spiked = torch.ones_like(gating) - (spiked_pos - 1.) / -1. + (spiked_neg - 1.) / -1.  # OR || +

        # self.s_fast = spiked_pos - spiked_neg + not_spiked * (-self.s_fast)  # / tau_s_fast = 1.
        self.v = not_spiked * v_next  # + spiked * 0.

        readout = self.O.matmul(self.s)
        # return (spiked_pos + spiked_neg), readout
        return spiked_pos, readout, self.v

