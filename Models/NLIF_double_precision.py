import torch
import torch.nn as nn
from torch import DoubleTensor as DT

from Models.TORCH_CUSTOM import static_clamp_for_matrix, static_clamp_for


class NLIF_double_precision(nn.Module):
    # W, U, I_o, O
    free_parameters = ['w', 'W_in', 'I_o', 'O']  # 0,2,3,5,8
    # parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8], 'tau_g': [5., 6.]}
    parameter_init_intervals = {'w': [0., 1.], 'W_in': [0., 1.], 'I_o': [0.2, 0.6], 'O': [0.5, 2.]}

    def __init__(self, N=30, w_mean=0., w_var=0.4):
        super(NLIF_double_precision, self).__init__()
        # self.device = device

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N

        self.v = torch.zeros((self.N,), dtype=torch.double)
        self.s = torch.zeros((self.N,), dtype=torch.double)
        self.s_fast = torch.zeros_like(self.v, dtype=torch.double)

        self.self_recurrence_mask = torch.ones((self.N, self.N), dtype=torch.double) - torch.eye(self.N, self.N, dtype=torch.double)
        self.self_recurrence_mask = self.self_recurrence_mask.double()

        rand_ws_syn = (w_mean - w_var) + 2 * w_var * torch.randn((self.N, self.N))
        # rand_ws_syn = rand_ws_syn * self.self_recurrence_mask
        rand_ws_syn = self.self_recurrence_mask * rand_ws_syn
        rand_ws_fast = (w_mean - w_var) + (2) * w_var * torch.randn((self.N, self.N))
        rand_ws_fast = self.self_recurrence_mask * rand_ws_fast
        # rand_ws_fast = torch.zeros((self.N, self.N))
        rand_ws_in = (w_mean - w_var) + (2) * w_var * torch.randn((self.N, 2))
        I_o = 0.1 * torch.randn((N,)).clip(-1., 1.).double()

        self.w_lim = 2.
        self.W_syn = nn.Parameter(DT(rand_ws_syn.clamp(-self.w_lim, self.w_lim).double()), requires_grad=True)
        self.W_fast = nn.Parameter(DT(rand_ws_fast.clamp(-self.w_lim, self.w_lim).double()), requires_grad=True)
        self.W_in = nn.Parameter(DT(rand_ws_in.clamp(-self.w_lim, self.w_lim).double()), requires_grad=True)  # "U" - input weights
        self.O = nn.Parameter(DT(torch.randn((2, self.N)).clamp(-self.w_lim, self.w_lim).double()), requires_grad=True)  # linear readout
        self.I_o = nn.Parameter(DT(I_o), requires_grad=True)  # tonic input current
        # self.I_o = DT(torch.zeros((self.N,)))  # tonic input current

        self.v_reset = torch.DoubleTensor([0.])[0]
        self.tau_m = torch.DoubleTensor([10.])[0]
        self.tau_s = torch.DoubleTensor([10.])[0]
        self.tau_s_fast = torch.DoubleTensor([1.])[0]
        # self.Delta = 0.1

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.W_syn.register_hook(lambda grad: static_clamp_for_matrix(grad, -self.w_lim, self.w_lim, self.W_syn))
        self.W_fast.register_hook(lambda grad: static_clamp_for_matrix(grad, -self.w_lim, self.w_lim, self.W_fast))
        self.W_in.register_hook(lambda grad: static_clamp_for_matrix(grad, -self.w_lim, self.w_lim, self.W_in))
        self.O.register_hook(lambda grad: static_clamp_for_matrix(grad, -self.w_lim, self.w_lim, self.O))

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

        self.v = torch.zeros((self.N,), dtype=torch.double)
        self.s = torch.zeros_like(self.v, dtype=torch.double)
        self.s_fast = torch.zeros_like(self.v, dtype=torch.double)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()
        self.s_fast = self.s_fast.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, x_in):
        # try:
        I_in = self.W_in.matmul(x_in)
        I_fast_syn = (self.self_recurrence_mask * self.W_fast).matmul(self.s_fast)
        I_syn = (self.self_recurrence_mask * self.W_syn).matmul((self.s))
        # except RuntimeError as re:
        #     print('re')

        I_tot = I_syn + I_fast_syn + I_in + self.I_o
        # I_tot = I_syn + I_fast_syn + I_in
        dv = ((I_tot) / self.tau_m)
        v_next = torch.add(self.v, dv)

        gating = v_next.clamp(0., 1.)
        ds = (gating * dv.clamp(-1., 1.) - self.s) / self.tau_s
        self.s = self.s + ds
        ds_fast = (gating * dv.clamp(-1., 1.) - self.s_fast) / self.tau_s_fast
        # ds_fast = (gating * dv.clamp(-1., 1.))
        self.s_fast = self.s_fast + ds_fast

        spiked_pos = (v_next >= 1.)
        spiked_neg = (v_next <= -1.)
        spiked = (spiked_pos + spiked_neg).double()
        not_spiked = torch.ones_like(spiked, dtype=torch.double)-spiked

        # self.s_fast = spiked_pos - spiked_neg + not_spiked * (-self.s_fast)  # / tau_s_fast = 1.
        self.v = not_spiked * v_next  # + spiked * 0.

        readout = self.O.matmul(self.s)
        # return (spiked_pos + spiked_neg), readout
        return spiked, readout, self.v, self.s, self.s_fast
        # return spiked_pos, readout, I_in, I_syn

