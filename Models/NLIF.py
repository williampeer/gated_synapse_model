import torch
import torch.nn as nn
from torch import FloatTensor as FT

from Models.TORCH_CUSTOM import static_clamp_for_matrix, static_clamp_for


class NLIF(nn.Module):
    # W, U, I_o, O
    free_parameters = ['W_in', 'I_o', 'O']  # 0,2,3,5,8
    # parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8], 'tau_g': [5., 6.]}
    parameter_init_intervals = {'W_in': [0., 1.], 'I_o': [0.2, 0.6], 'O': [0.5, 2.]}

    def __init__(self, parameters, N=30, w_mean=0.15, w_var=0.15):
        super(NLIF, self).__init__()
        # self.device = device

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N

        self.v = torch.zeros((self.N,))
        self.s = torch.zeros((self.N,))
        self.s_fast = torch.zeros((self.N,))

        self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)

        rand_ws_syn = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        # rand_ws_syn = rand_ws_syn * self.self_recurrence_mask
        rand_ws_syn = self.self_recurrence_mask * rand_ws_syn
        rand_ws_fast = (w_mean - w_var) + (2) * w_var * torch.rand((self.N, self.N))
        rand_ws_fast = self.self_recurrence_mask * rand_ws_fast
        # rand_ws_in = (w_mean - w_var) + (2) * w_var * torch.rand((self.N, 2))
        # rand_ws_O = (w_mean - w_var) + (2) * torch.rand((2, self.N))
        rand_ws_in = torch.randn((N, 2))
        rand_ws_O = torch.randn((2, N))

        I_o = 0.05 * torch.rand((N,)).clip(-1., 1.)

        if parameters:
            for key in parameters.keys():
                if key == 'W_in':
                    rand_ws_in = FT(parameters[key])
                elif key == 'W_fast':
                    rand_ws_fast = FT(parameters[key])
                elif key == 'W_syn':
                    rand_ws_syn = FT(parameters[key])
                elif key == 'O':
                    rand_ws_O = FT(parameters[key])
                elif key == 'I_o':
                    I_o = FT(parameters[key])

        self.w_lim = 2.
        self.W_syn = nn.Parameter(FT(rand_ws_syn.clamp(-self.w_lim, self.w_lim)), requires_grad=True)
        self.W_fast = nn.Parameter(FT(rand_ws_fast.clamp(-self.w_lim, self.w_lim)), requires_grad=True)
        self.W_in = nn.Parameter(FT(rand_ws_in.clamp(-self.w_lim, self.w_lim)), requires_grad=True)  # "U" - input weights
        self.O = nn.Parameter(FT(rand_ws_O.clamp(-self.w_lim, self.w_lim)), requires_grad=True)  # linear readout
        self.I_o = nn.Parameter(FT(I_o), requires_grad=True)  # tonic input current
        # self.I_o = FT(torch.zeros((self.N,)))  # tonic input current

        self.v_reset = 0.
        self.tau_m = 10.
        self.tau_s = 10.
        self.tau_s_fast = 1.
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
        params_list['W_fast'] = self.W_fast.data
        params_list['W_in'] = self.W_in.data
        params_list['I_o'] = self.I_o.data
        params_list['O'] = self.O.data

        params_list['tau_m'] = self.tau_m
        params_list['tau_s'] = self.tau_s

        return params_list

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        # self.v = torch.zeros((self.N,))
        # self.s = torch.zeros_like(self.v)
        # self.s_fast = torch.zeros_like(self.v)

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.s = self.s.clone().detach()
        self.s_fast = self.s_fast.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, x_in):
        I_in = self.W_in.matmul(x_in)
        I_fast_syn = (self.self_recurrence_mask * self.W_fast).matmul(self.s_fast)
        I_syn = (self.self_recurrence_mask * self.W_syn).matmul(self.s)

        I_tot = I_syn + I_fast_syn + I_in + self.I_o
        dv = ((I_tot) / self.tau_m)
        v_next = torch.add(self.v, dv)

        # TODO: Implement different gating models
        gating = v_next.clamp(0., 1.)
        ds = (gating * dv.clamp(-1., 1.) - self.s) / self.tau_s
        self.s = self.s + ds
        ds_fast = (gating * dv.clamp(-1., 1.) - self.s_fast) / self.tau_s_fast
        self.s_fast = self.s_fast + ds_fast

        spiked_pos = (v_next >= 1.)
        spiked_neg = (v_next <= -1.)
        spiked = (spiked_pos + spiked_neg).float()
        not_spiked = torch.ones_like(spiked)-spiked

        # self.s_fast = spiked_pos - spiked_neg + not_spiked * (-self.s_fast)  # / tau_s_fast = 1.
        self.v = not_spiked * v_next  # + spiked * 0.

        readout = self.O.matmul(self.s)
        # return (spiked_pos + spiked_neg), readout
        return spiked, readout, self.v, self.s, self.s_fast
        # return spiked_pos, readout, I_in, I_syn

