import os
import sys

import numpy as np
import torch

import util
from Models.LIF import LIF
from Models.NLIF import NLIF


def calc_rmse(readout, target):
    return torch.sqrt(torch.mean(torch.pow(target-readout, 2))).detach().numpy()


def get_rmse(model, random_seed, exp_type, t=120):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    period_ms = 40
    tau_filter = 50.

    A_in = torch.tensor([-1., 0.5])
    A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])

    if exp_type == 'AutoEncoding':
        cur_period_ms = torch.tensor([period_ms, period_ms / 2, period_ms / 3, period_ms / 4])
        phase_shifts_1 = torch.tensor([0., 0.1, 0.2, 0.3])
        phase_shifts_2 = phase_shifts_1 + 3.141592 / 4
        # inputs, target_outputs = util.auto_encoder_task_input_output(t=t, period_ms=period_ms, tau_filter=tau_filter,
        #                                                              Delta=Delta, A_in=A_in, phase_shifts=phase_shifts)
        # inputs_1 = util.generate_sum_of_sinusoids_vector(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)), phase_shifts=phase_shifts_1)
        # inputs_2 = util.generate_sum_of_sinusoids_vector(t=t, period_ms=period_ms, A_coeff=torch.randn((4,)), phase_shifts=phase_shifts_2)
        inputs_1 = util.generate_sum_of_sinusoids(t=t, period_ms=cur_period_ms, A_coeff=torch.randn((4,)),
                                                  phase_shifts=torch.rand((4,)))
        inputs_2 = util.generate_sum_of_sinusoids(t=t, period_ms=cur_period_ms, A_coeff=torch.randn((4,)),
                                                  phase_shifts=torch.rand((4,)))
        inputs = torch.vstack([inputs_1, inputs_2]).T
        target_outputs = util.auto_encode_input(inputs, tau_filter=tau_filter)
    elif exp_type == 'GeneralPredictiveEncoding':
        inputs, target_outputs = util.general_predictive_encoding_task_input_output(t=t, period_ms=period_ms,
                                                                                    tau_filter=tau_filter,
                                                                                    Delta=1., A_in=A_in, A_mat=A_mat)
    else:
        raise NotImplementedError()

    _ = torch.tensor(inputs.clone().detach(), requires_grad=True)
    current_inputs = torch.tensor(inputs.clone().detach(), requires_grad=True)
    spikes, readouts, v, s, s_fast = util.feed_inputs_sequentially_return_tuple(model, current_inputs)
    print('sum model outputs: {}'.format(readouts.sum()))

    return calc_rmse(readouts, target=target_outputs.clone().detach())


experiments_path = '/home/william/repos/archives_snn_inference/sut_archive_gating/saved/'
# exp_types = ['AutoEncoding', 'GeneralPredictiveEncoding']

model_class_lookup = {'LIF': LIF, 'NLIF': NLIF}

# for exp_type in exp_types:
# model_type_dirs = os.listdir(experiments_path)
model_type_dirs = ['LIF', 'NLIF']

res = { 'AutoEncoding': { 'LIF': [], 'NLIF': [] }, 'GeneralPredictiveEncoding': { 'LIF': [], 'NLIF': [] } }
for model_type_str in model_type_dirs:
    if not model_type_str.__contains__("plot_data"):
        model_class = model_class_lookup[model_type_str]
        # model_class = microGIF
        if os.path.exists(experiments_path + model_type_str):
            exp_uids = os.listdir(experiments_path + '/' + model_type_str)
            for euid in exp_uids:
                files = os.listdir(experiments_path + '/' + model_type_str + '/' + euid)
                file_name = list(filter(lambda x: x.__contains__('LIF_exp'), files))[0]  # 'LIF_exp_auto_encode_random_seed_23'
                random_seed = file_name.split('random_seed_')[1]
                exp_type = file_name.split('_random_seed')[0].split('exp_')[1]

                load_data = torch.load(experiments_path + '/' + model_type_str + '/' + euid + '/' + file_name)
                snn = load_data['model']

                cur_rmse = get_rmse(snn, random_seed, exp_type)
                res[exp_type][model_type_str].append(cur_rmse)
        else:
            print('path does not exist: {}'.format(experiments_path + '/' + model_type_str))

# sys.exit()
print('res:', res)
for exp_type, exp_key in enumerate(res):
    for model_type, m_name in enumerate(res[exp_key]):
        print('exp_key: {}, m_name: {}'.format(exp_key, m_name))
        print('mean_rmse: ', np.mean(res[exp_key][m_name]))
