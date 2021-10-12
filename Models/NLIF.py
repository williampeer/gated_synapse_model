import torch
import torch.nn as nn
from torch import FloatTensor as FT
from torch import tensor as T

from Models.TORCH_CUSTOM import static_clamp_for, static_clamp_for_matrix


class NLIF(nn.Module):
    free_parameters = ['w', 'W_in', 'O', 'I_o']  # 0,2,3,5,8
    # parameter_init_intervals = {'E_L': [-64., -55.], 'tau_m': [3.5, 4.0], 'G': [0.7, 0.8], 'tau_g': [5., 6.]}
    parameter_init_intervals = {'w': [0., 1.], 'W_in': [0., 1.], 'O': [0.5, 2.], 'I_o': [0.2, 0.6]}

    def __init__(self, parameters, N=12, w_mean=0.4, w_var=0.25, neuron_types=T([1, -1])):
        super(NLIF, self).__init__()
        # self.device = device
        rand_ws = None

        if parameters:
            for key in parameters.keys():
                if key == 'tau_m':
                    tau_m = FT(torch.ones((N,)) * parameters[key])
                elif key == 'E_L':
                    E_L = FT(torch.ones((N,)) * parameters[key])
                elif key == 'tau_g':
                    tau_g = FT(torch.ones((N,)) * parameters[key])
                elif key == 'preset_weights':
                    rand_ws = torch.abs(parameters['preset_weights'])
                    assert rand_ws.shape[0] == N and rand_ws.shape[1] == N, "shape of weights matrix should be NxN"

        __constants__ = ['N', 'norm_R_const', 'self_recurrence_mask']
        self.N = N
        # self.norm_R_const = (delta_theta_s - E_L) * 1.1

        self.v = torch.zeros((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance

        if rand_ws is None:
            rand_ws = (w_mean - w_var) + 2 * w_var * torch.rand((self.N, self.N))
        nt = T(neuron_types).float()
        self.neuron_types = torch.transpose((nt * torch.ones((self.N, self.N))), 0, 1)
        self.w = nn.Parameter(FT(rand_ws), requires_grad=True)  # initialise with positive weights only
        # self.self_recurrence_mask = torch.ones((self.N, self.N)) - torch.eye(self.N, self.N)
        self.self_recurrence_mask = torch.ones((self.N, self.N))

        self.v_reset = FT(0.)
        # self.tau_m = FT(tau_m).clamp(1.5, 8.)
        # self.tau_g = FT(tau_g).clamp(1., 12,)
        self.tau_m = FT(10.)
        self.tau_g = FT(10.)
        # self.tau_fast = FT(1.)

        self.register_backward_clamp_hooks()

    def register_backward_clamp_hooks(self):
        self.E_L.register_hook(lambda grad: static_clamp_for(grad, -80., -35., self.E_L))
        self.tau_m.register_hook(lambda grad: static_clamp_for(grad, 1.5, 8., self.tau_m))
        self.tau_g.register_hook(lambda grad: static_clamp_for(grad, 1., 12., self.tau_g))

        self.w.register_hook(lambda grad: static_clamp_for_matrix(grad, 0., 1., self.w))

    def get_parameters(self):
        params_list = []
        # parameter_names = ['w', 'E_L', 'tau_m', 'tau_s', 'G', 'f_v', 'delta_theta_s', 'b_s', 'delta_V']
        params_list.append(self.w.data)
        # params_list.append(self.E_L.data)
        params_list.append(self.tau_m.data)
        params_list.append(self.tau_g.data)

        return params_list

    def reset(self):
        for p in self.parameters():
            p.grad = None
        self.reset_hidden_state()

        self.v = self.E_L.clone().detach() * torch.ones((self.N,))
        self.g = torch.zeros_like(self.v)  # syn. conductance

    def reset_hidden_state(self):
        self.v = self.v.clone().detach()
        self.g = self.g.clone().detach()
        self.theta_s = self.theta_s.clone().detach()

    def name(self):
        return self.__class__.__name__

    def forward(self, I_ext):
        # TODO: Correct NIF formulation
        W_syn = self.w * self.neuron_types
        I_syn = (self.g).matmul(self.self_recurrence_mask * W_syn)
        I_fast_syn = self.g_fast.matmul(self.W_fast)

        I_tot = I_syn + I_fast_syn + I_ext.matmul(self.W_in) + self.I_tonic
        dv = ((self.v_reset - self.v) + I_tot) / self.tau_m
        v_next = torch.add(self.v, dv)

        # TODO: Add
        gating = (v_next / self.v_thresh_pos).clamp(0., 1.)
        dv_max = (self.v_thresh_pos - self.v_reset)
        ds = (-self.s + gating * (dv / dv_max).clamp(0., 1.)) / self.tau_s
        self.s = self.s + ds

        # non-differentiable, hard threshold for nonlinear reset dynamics
        spiked = (v_next >= self.theta_s).float()
        not_spiked = (spiked - 1.) / -1.

        self.g = spiked + not_spiked * (-self.g/self.tau_g)
        self.g_fast = spiked + not_spiked * (-self.g_fast)

        self.theta_s = torch.add((1-self.b_s) * self.theta_s, spiked * self.delta_theta_s)
        v_reset = self.E_L + self.f_v * (self.v - self.E_L) - self.delta_V
        self.v = torch.add(spiked * v_reset, not_spiked * v_next)

        readout = self.O * self.s
        return readout

