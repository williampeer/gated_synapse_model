import os
import sys

import numpy as np
import torch

import IO
import plot


def proper_label(p_name):
    return p_name\
        .replace('W_syn', 'W_{syn}')\
        .replace('W_fast', 'W_{fast}')\
        .replace('W_in', 'W_{in}')\
        .replace('tau_theta', '\\tau_{\\theta}')\
        .replace('J_theta', 'J_{\\theta}')\
        .replace('tau_m', '\\tau_m')\
        .replace('tau_s', '\\tau_s')\
        .replace('Delta_u', '\\Delta_u')\
        .replace('theta_inf', '\\theta_{inf}')\
        .replace('delta_theta_s', '\\delta_{\\theta_s}')


GT_model_by_class = { 'LIF': '12-31_10-51-36-589',
                      'NLIF': '12-31_10-52-10-075' }

# archive_path = '/home/william/repos/snn_inference/saved/plot_data/'
# archive_path = '/media/william/p6/archive_14122021/archive/saved/plot_data/'
# archive_path = '/home/william/repos/archives_snn_inference/archive_1612/archive/saved/plot_data/'
archive_path = '/home/william/repos/archives_snn_inference/archive_gating_p_scapes/archive/saved/plot_data/'
tar_path = IO.PATH
# model_type_dirs = ['LIF', 'GLIF', 'microGIF']
model_type_dirs = ['LIF', 'NLIF']
# model_type_dirs = ['microGIF']
for mt_str in model_type_dirs:
    mt_dir = 'test_{}'.format(mt_str)
    specific_plot_files = os.listdir(archive_path + mt_dir)
    for sp_file in specific_plot_files:
        load_data = torch.load(archive_path + mt_dir + '/' + sp_file)
        save_data = load_data['plot_data']
        saved_fname = save_data['fname']
        # model_N = int(saved_fname.split('_N_')[1].split('_')[0])
        model_N = 4
        # data = {'p1s': p1s, 'p2s': p2s, 'summary_statistic': summary_statistic,
        #             'p1_name': p1_name, 'p2_name': p2_name, 'statistic_name': statistic_name,
        #             'exp_type': exp_type, 'uuid': uuid, 'fname': fname}

        N_dim = int(np.sqrt(len(save_data['p1s'])))  # assuming equal length of p1s and p2s
        heat_mat = np.zeros((N_dim, N_dim))
        summary_norm_const = np.sign(save_data['summary_statistic'][0]) * np.max(np.abs(save_data['summary_statistic']))  # loss, rate
        statistic_name = save_data['statistic_name']  # loss, rate

        for i in range(len(save_data['p1s'])):
            # x_ind = int(save_data['p1s'][i] / p1_last)
            # y_ind = int(save_data['p2s'][i] / p2_last)
            x_ind = i % N_dim
            y_ind = int(i/N_dim)
            heat_mat[x_ind, y_ind] = save_data['summary_statistic'][i] / summary_norm_const

        # prev_timestamp = '11-16_11-21-13-903'
        fname = '{}_exp_auto_encode_random_seed_23'.format(mt_str)
        GT_lookup_str = mt_str
        if mt_str == 'microGIF' and model_N == 4:
            GT_lookup_str = 'mesoGIF'
        GT_euid = GT_model_by_class[GT_lookup_str]
        load_data_target = torch.load(tar_path + mt_str + '/' + GT_euid + '/' + fname + IO.fname_ext)
        snn_target = load_data_target['model']
        target_params = snn_target.get_parameters()
        tar_p1 = target_params[save_data['p1_name']].numpy()
        tar_p2 = target_params[save_data['p2_name']].numpy()

        # WIP: Fix all this.
        # p1_last = save_data['p1s'][-1]
        # p2_last = save_data['p2s'][-1]
        interval_1 = save_data['p1s'][-1] - save_data['p1s'][0]  # interval
        # e.g. -40 + 70 = 30
        #   pt: -50. => tp1i := (-50. + 70) / 30 = 2/3. OK.
        #   pt: 2. => tp1 := (2-1.5) / (3.5-1.5) = 1/4. OK.
        interval_2 = save_data['p2s'][-1] - save_data['p2s'][0]  # interval
        # t_p1_index = int(N_dim * ((np.mean(tar_p1) - save_data['p1s'][0]) / interval_1))
        # t_p2_index = int(N_dim * ((np.mean(tar_p2) - save_data['p2s'][0]) / interval_2))
        # target_coords = [t_p1_index, t_p2_index]
        xticks = []
        yticks = []
        for i_tick in range(N_dim):
            xticks.append(save_data['p1s'][i_tick*N_dim])
            yticks.append(save_data['p2s'][i_tick])
        # ---------------- target data feature request from Arno ------------------

        axes = ['${}$'.format(proper_label(save_data['p1_name'])), '${}$'.format(proper_label(save_data['p2_name']))]
        exp_type = 'test'; uuid = 'export_p_landscape_2d'
        # model_N =
        plot.plot_p_landscape_heatmap(heat_mat, axes, exp_type, uuid, fname=mt_str+'/test_export_2d_heatmap_N_{}_{}_{}_{}.eps'.format(model_N, statistic_name, save_data['p1_name'], save_data['p2_name']),
                                      target_coords=False, xticks=xticks, yticks=yticks, cbar_label=statistic_name)

sys.exit()
