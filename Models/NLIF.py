import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class NLIF(nn.Module):
    free_parameters = ['w', 'W_in', 'O', 'I_o']  # 0,2,3,5,8
    # parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8], 'tau_g': [5., 6.]}
    parameter_init_intervals = {'w': [0., 1.], 'W_in': [0., 1.], 'O': [0.5, 2.], 'I_o': [0.2, 0.6]}

    def __init__(self, N=30, w_mean=0.4, w_var=0.25):
        super(NLIF, self).__init__()
        # self.device = device

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros((self.N,))
        self.s_fast = torch.zeros_like(self.v)  # syn. conductance

        rand_ws_syn = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        rand_ws_fast = (w_mean/2. - w_var/2.) + (2/2.) * w_var * torch.rand((self.N, self.N))
        rand_ws_in = (w_mean/5. - w_var/5.) + (2/5.) * w_var * torch.rand((2, self.N))
        # nt = T(neuron_types).float()
        # self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)

        self.W_syn = nn.Parameter(FT(rand_ws_syn), requires_grad=True)
        self.W_fast = nn.Parameter(FT(rand_ws_fast), requires_grad=True)
        self.W_in = nn.Parameter(FT(rand_ws_in), requires_grad=True)

        self.O = nn.Parameter(torch.ones((2, N)), requires_grad=True)  # linear readout

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        # self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v_reset = 0.
        self.tau_m = 10.
        self.tau_s = 10.
        # self.tau_s_fast = 1.
        self.I_tonic = 0.

        # self.register_backward_clamp_hooks()

    # def register_backward_clamp_hooks(self):
    #     self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
    #     self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
    #     self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_g))
    #
    #     self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_list = []
        params_list.append(self.W_syn.data)
        params_list.append(self.W_syn_fast.data)
        params_list.append(self.W_in.data)

        params_list.append(self.tau_m)
        params_list.append(self.tau_s)

        return params_list

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
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
        I_syn = (self.s).matmul(self.W_syn * self.self_recurrence_mask)
        I_fast_syn = self.s_fast.matmul(self.W_fast * self.self_recurrence_mask)
        I_in = x_in.matmul(self.W_in)

        I_tot = I_syn + I_fast_syn + I_in + self.I_tonic
        dv = (I_tot - self.v) / self.tau_m
        v_next = torch.add(self.v, dv)

        gating = v_next
        ds = (gating.clamp(-1., 1.) * dv - self.s) / self.tau_s  # ensure integral = 1
        self.s = self.s + ds
        ds_fast = (gating.clamp(-1., 1.) * dv - self.s_fast)
        self.s_fast = self.s_fast + ds_fast

        spiked_pos = (v_next >= 1.).float()
        spiked_neg = (v_next <= -1.).float()
        not_spiked = (spiked_pos - 1.) / -1. + (spiked_neg - 1.) / -1.  # OR || +

        # self.s_fast = spiked_pos - spiked_neg + not_spiked * (-self.s_fast)  # / tau_s_fast = 1.
        self.v = not_spiked * v_next  # + spiked * 0.

        readout = self.O.matmul(self.s)
        return (spiked_pos + spiked_neg), readout

