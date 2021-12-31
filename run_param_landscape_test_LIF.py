import sys

import torch

import IO
import util
from Models.LIF import LIF
from Test.parameter_landscape_test import plot_param_landscape

A_coeffs = [torch.randn((4,))]
phase_shifts = [torch.rand((4,))]
t = 240
num_steps = 100

prev_timestamp = '12-31_10-51-36-589'
fname = 'LIF_exp_auto_encode_random_seed_23'
load_data = torch.load(IO.PATH + LIF.__name__ + '/' + prev_timestamp + '/' + fname + IO.fname_ext)
snn_target = load_data['model']

A_in = torch.tensor([-1., 0.5])
A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])
period_ms = 40
tau_filter = 50.

# if exp_type is ExpType.AutoEncoding:
period_ms = torch.tensor([period_ms, period_ms / 2, period_ms / 3, period_ms / 4])
phase_shifts_1 = torch.tensor([0., 0.1, 0.2, 0.3])
phase_shifts_2 = phase_shifts_1 + 3.141592 / 4
# inputs, target_outputs = util.auto_encoder_task_input_output(t=t, period_ms=period_ms, tau_filter=tau_filter,
#                                                              Delta=Delta, A_in=A_in, phase_shifts=phase_shifts)
# inputs_1 = util.generate_sum_of_sinusoids_vector(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)), phase_shifts=phase_shifts_1)
# inputs_2 = util.generate_sum_of_sinusoids_vector(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)), phase_shifts=phase_shifts_2)
inputs_1 = util.generate_sum_of_sinusoids(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)),
                                          phase_shifts=torch.rand((4,)))
inputs_2 = util.generate_sum_of_sinusoids(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)),
                                          phase_shifts=torch.rand((4,)))
inputs = torch.vstack([inputs_1, inputs_2]).T
target_outputs = util.auto_encode_input(inputs, tau_filter=tau_filter)
# elif exp_type is ExpType.GeneralPredictiveEncoding:
# inputs, target_outputs = util.general_predictive_encoding_task_input_output(t=t, period_ms=period_ms,
#                                                                             tau_filter=tau_filter,
#                                                                             Delta=Delta, A_in=A_in, A_mat=A_mat)
# tar_spikes, tar_readouts, tar_vs, tar_ss, tar_s_fasts = util.feed_inputs_sequentially_return_tuple(snn_target, current_inputs.clone().detach())

current_inputs = torch.tensor(inputs.clone().detach(), requires_grad=True)
target_signal = target_outputs

# other_parameters = experiments.draw_from_uniform(microGIF.parameter_init_intervals, N=snn_target.N)
other_parameters = snn_target.get_parameters()
print('snn_target.N: {}'.format(snn_target.N))
# other_parameters['N'] = snn_target.N
# other_parameters = snn_target.parameters()
# free_parameters = ['w', 'E_L', 'tau_m', 'G', 'f_v', 'f_I', 'delta_theta_s', 'b_s', 'a_v', 'b_v', 'theta_inf', 'delta_V', 'tau_s']
plot_param_landscape(LIF, [-70., -40.], [1.5, 10.], 'E_L', 'tau_m', other_parameters, target_signal, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=int(snn_target.N), fname_addition='white_noise')
plot_param_landscape(LIF, [-70., -40.], [1., 12.], 'E_L', 'tau_s', other_parameters, target_signal, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise')
plot_param_landscape(LIF, [1.5, 10.], [1., 12.], 'tau_m', 'tau_s', other_parameters, target_signal, num_steps=num_steps, inputs=current_inputs.clone().detach(), N=snn_target.N, fname_addition='white_noise')

sys.exit(0)
