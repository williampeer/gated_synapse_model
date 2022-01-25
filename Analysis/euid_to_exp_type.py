import os

fig_path = '/home/william/repos/archives_snn_inference/archive_full_gating_0401/archive/figures/'
exp_types = ['AutoEncoding', 'GeneralPredictiveEncoding']
model_types = ['LIF', 'NLIF']

euid_to_exp_type = {}
euid_to_model_type = {}
for exp_type in exp_types:
    for model_type in model_types:
        exp_uids = os.listdir(fig_path + '/' + exp_type + '/' + model_type)
        for euid in exp_uids:
            euid_to_exp_type[euid] = exp_type
            euid_to_model_type[euid] = model_type
