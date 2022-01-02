import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import IO

plt.rcParams.update({'font.size': 12})


def plot_spike_train(spike_train, title, uuid, exp_type='default', fname='spiketrain_test'):

    data = {'spike_history': spike_train, 'title': title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_spiketrain')

    plt.figure()
    # assuming neuron indices to be columns and reshaping to rows for plotting
    time_indices = torch.reshape(torch.arange(spike_train.shape[0]), (spike_train.shape[0], 1))
    # ensure binary values:
    spike_train = torch.round(spike_train)
    neuron_spike_times = spike_train * time_indices.float()

    for neuron_i in range(spike_train.shape[1]):
        if neuron_spike_times[:, neuron_i].nonzero().sum() > 0:
            plt.plot(torch.reshape(neuron_spike_times[:, neuron_i].nonzero(), (1, -1)).numpy(),
                     neuron_i+1, '.k', markersize=4.0)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.yticks(range(neuron_i+2))
    plt.title(title)

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    # IO.makedir_if_not_exists('/figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_neuron(membrane_potentials_through_time, uuid, exp_type='default', title='Neuron activity',
                ylabel='Membrane potential', fname='plot_neuron_test', use_legend=False, legend=False):
    data = {'membrane_potentials_through_time': membrane_potentials_through_time, 'title': title, 'uuid': uuid,
            'exp_type': exp_type, 'ylabel': ylabel, 'fname': fname}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')

    # plt.figure()
    plt.plot(np.arange(membrane_potentials_through_time.shape[0]), membrane_potentials_through_time)

    # plt.title(title)
    if use_legend:
        legend = []
        for i in range(len(membrane_potentials_through_time)):
            legend.append('N.{}'.format(i+1))
        plt.legend(legend, loc='upper left', ncol=4)
    if legend:
        plt.legend(legend)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    # plt.show()

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    plt.close()


def plot_spike_train_projection(spikes, uuid='test', exp_type='default', title=False, fname=False, legend=None, export=False):
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)

    time_indices = torch.reshape(torch.arange(spikes.shape[0]), (spikes.shape[0], 1)).float()
    # ensure binary values:
    spike_history = torch.round(spikes)
    model_spike_times = spike_history * time_indices

    # ensure binary values:
    for neuron_i in range(spike_history.shape[1]):
        if model_spike_times[:, neuron_i].nonzero().sum() > 0:
            spike_times_reshaped = torch.reshape(model_spike_times[:, neuron_i].nonzero(), (1, -1))
            ax.scatter3D(spike_times_reshaped.numpy(),
                         (torch.ones_like(spike_times_reshaped) * neuron_i + 1.1).numpy(),
                         zs=0, label='Model')

    plt.xlabel('Time ($ms$)')
    plt.ylabel('Neuron')
    ax.set_zlabel('Parameters $P \in \Re^\mathbf{D}$')
    ax.set_zticks(range(-1, 2))
    if neuron_i > 20:
        ax.set_yticks(range(int((neuron_i + 1) / 10), neuron_i + 1, int((neuron_i + 1) / 10)))
    else:
        ax.set_yticks(range(1, neuron_i + 2))
    ax.set_xticks(range(0, 5000, 1000))
    ax.set_ylim(0, neuron_i + 2)

    if not fname:
        fname = 'spike_train_projection' + IO.dt_descriptor()

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    fig.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_heatmap(heat_mat, axes, exp_type, uuid, fname):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'heat_mat': heat_mat, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_heatmap')

    for row_i in range(heat_mat.shape[0]):
        for col_i in range(heat_mat.shape[1]):
            if np.isnan(heat_mat[row_i][col_i]):
                heat_mat[row_i][col_i] = 0.

    a = plt.imshow(heat_mat, cmap="PuOr", vmin=-1, vmax=1)
    cbar = plt.colorbar(a)
    # cbar.set_label("correlation coeff.")
    plt.xticks(np.arange(0, len(heat_mat), 5))
    plt.yticks(np.arange(0, len(heat_mat)))
    plt.ylabel(axes[0])
    plt.xlabel(axes[1])
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


def plot_p_landscape_heatmap(heat_mat, axes, exp_type, uuid, fname, target_coords=False, xticks=False, yticks=False, v_min=0, v_max=1, cbar_label='loss'):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'heat_mat': heat_mat, 'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_heatmap')

    for row_i in range(heat_mat.shape[0]):
        for col_i in range(heat_mat.shape[1]):
            if np.isnan(heat_mat[row_i][col_i]):
                heat_mat[row_i][col_i] = 0.

    fig = plt.figure()
    im = plt.imshow(heat_mat, cmap="PuOr", vmin=v_min, vmax=v_max)
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)
    ticks_fmt = lambda x: float('{:.2f}'.format(x))
    if xticks:
        N_dim = len(xticks)
        tar_xticks = [xticks[0], xticks[int(N_dim/2)], xticks[-1]]
        tar_xticks = list(map(ticks_fmt, tar_xticks))
        plt.xticks([0, int(N_dim/2), N_dim-1], tar_xticks)
    else:
        plt.xticks(np.arange(0, len(heat_mat), 5))
    if yticks:
        N_dim = len(yticks)
        tar_yticks = [yticks[0], yticks[int(N_dim / 2)], yticks[-1]]
        tar_yticks = list(map(ticks_fmt, tar_yticks))
        plt.yticks([0, int(N_dim / 2), N_dim - 1], [yticks[0], yticks[int(N_dim / 2)], yticks[-1]])
    else:
        plt.yticks(np.arange(0, len(heat_mat)))
    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    if target_coords:
        plt.scatter(target_coords[0], target_coords[1], color='magenta', marker='x', s=30.0)
    # plt.show()

    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    plt.close()
    return fig



def plot_loss(loss, uuid, exp_type='default', custom_title=False, fname=False):
    if not fname:
        fname = 'loss_'+IO.dt_descriptor()
    else:
        fname = fname
    data = {'loss': loss, 'exp_type': exp_type, 'custom_title': custom_title, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_loss')

    plt.plot(loss)
    # plt.legend(['Training loss', 'Test loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # ax.yaxis.set_major_formatter(FormatStrFormatter('% 1.2f'))
    # plt.xticks(range(len(loss_arr+1)))
    if custom_title:
        plt.title(custom_title)
    else:
        plt.title('Loss')

    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)
    # plt.show()
    plt.close()


def plot_weights_dots(U, O, uuid, exp_type):
    plt.scatter(U[:,0], U[:,1])
    plt.scatter(O[:,0], O[:,1])
    plt.xlabel('$i,j$')
    plt.ylabel('$i,j$')
    plt.title('U, O')
    plt.legend(['U', 'O'])
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    plt.savefig(full_path+'_weights_O_U.png')
    plt.close()


# ------------------------------
def decompose_param_pair_trajectory_plot(param_2D, current_targets, name, path):
    if os.path.exists(path + '.png'):
        return

    params_by_exp = np.array(param_2D).T
    num_of_parameters = params_by_exp.shape[0]

    plt.rcParams.update({'font.size': 8})
    plt.locator_params(axis='x', nbins=2)

    fig = plt.figure()
    big_ax = fig.add_subplot(111)
    big_ax.set_axis_on()
    big_ax.grid(False)
    name = name.replace('tau', '\\tau').replace('spike_threshold', '\\theta')
    big_ax.set_title('Parameter trajectory for ${} \\times {}$'.format(name, name))
    big_ax.set_xlabel('${}$'.format(name), labelpad=20)
    big_ax.set_ylabel('${}$'.format(name), labelpad=34)
    big_ax.set_xticks([])
    big_ax.set_yticks([])
    print('num_of_parameters: {}'.format(num_of_parameters))
    axs = fig.subplots(nrows=num_of_parameters - 1, ncols=num_of_parameters - 1, sharex=True, sharey=True)
    dot_msize = 5.0
    if num_of_parameters == 2:
        # if current_targets is not False:
            # x_min = float('{}'.format(np.min(np.concatenate([params_by_exp[0], [current_targets[0]]]))))
            # x_max = float('\n{}'.format(np.max(np.concatenate([params_by_exp[0], [current_targets[0]]]))))
            # plt.xticks([x_min, x_max])

        p_len = len(params_by_exp[0])
        colors = cm.viridis(np.linspace(0, 1, p_len))
        for p_i in range(p_len):
            plt.scatter(params_by_exp[0][p_i], params_by_exp[1][p_i], color=colors[p_i], marker='o', s=dot_msize)

        if current_targets is not False:
            plt.scatter(current_targets[0], current_targets[1], color='black', marker='x',
                           s=2. * dot_msize)  # test 2*dot_msize
    else:
        [axi.set_axis_off() for axi in axs.ravel()]

        for i in range(num_of_parameters):
            for j in range(i+1, num_of_parameters):
                cur_ax = axs[j - 1, i]
                cur_ax.set_axis_on()
                cur_ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # if current_targets is not False:
                #     x_min = float('{}'.format(np.min(np.concatenate([params_by_exp[i], [current_targets[i]]]))))
                #     x_max = float('\n{}'.format(np.max(np.concatenate([params_by_exp[i], [current_targets[i]]]))))
                #     cur_ax.set_xticks([x_min, x_max])

                # try:
                p_len = len(params_by_exp[i])
                colors = cm.viridis(np.linspace(0, 1, p_len))
                for p_i in range(p_len):
                    cur_ax.scatter(params_by_exp[i][p_i], params_by_exp[j][p_i], color=colors[p_i], marker='o', s=dot_msize)

                if current_targets is not False:
                    cur_ax.scatter(current_targets[i], current_targets[j], color='black', marker='x', s=2.*dot_msize)  # test 2*dot_msize

    if not path:
        path = './figures/{}/{}/param_subplot_inferred_params_{}'.format('default', 'test_uuid', IO.dt_descriptor())
    # plt.show()
    fig.savefig(path + '.png')
    plt.close()


def plot_parameter_inference_trajectories_2d(param_means, target_params, param_names, exp_type, uuid, fname, custom_title):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    if not fname:
        fname = 'new_inferred_params_{}'.format(IO.dt_descriptor())
    path = full_path + fname

    if not os.path.exists(path):
        data = {'param_means': param_means, 'target_params': target_params, 'exp_type': exp_type, 'uuid': uuid, 'custom_title': custom_title, 'fname': fname}
        IO.save_plot_data(data=data, uuid=uuid, plot_fn='plot_parameter_inference_trajectories_2d')

        for p_i, p_k in enumerate(param_means):  # assuming a dict., for all parameter combinations
            current_targets = False
            if target_params is not False:
                if p_k in target_params:
                    current_targets = target_params[p_k]

            cur_p = np.array(param_means[p_k])
            name = '{}'.format(p_k)

            # silently fail for 3D params (weights)
            if len(cur_p.shape) == 2:
                param_path = path+'_param_{}'.format(p_k)
                if not os.path.exists(param_path) and not os.path.exists(param_path + '.png'):
                    # decompose_param_pair_trajectory_plot(cur_p[:,:,:4], current_targets[:,:,:4], name=name, path=param_path)
                    max_index = min(5, len(cur_p))
                    if current_targets is not False:
                        current_targets = current_targets[:max_index]
                    decompose_param_pair_trajectory_plot(cur_p[:, :max_index], current_targets, name=name, path=param_path)


def plot_parameter_landscape(p1s, p2s, p1_name, p2_name, summary_statistic, statistic_name, exp_type, uuid, fname):
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists('./figures/' + exp_type + '/')
    IO.makedir_if_not_exists(full_path)

    data = {'p1s': p1s, 'p2s': p2s, 'summary_statistic': summary_statistic,
            'p1_name': p1_name, 'p2_name': p2_name, 'statistic_name': statistic_name,
            'exp_type': exp_type, 'uuid': uuid, 'fname': fname}
    IO.save_plot_data(data=data, uuid=uuid, plot_fn=fname)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(p1s, p2s, summary_statistic, cmap=plt.cm.jet, linewidth=0.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('${}$'.format(p1_name))
    ax.set_ylabel('${}$'.format(p2_name))
    ax.set_zlabel('${}$'.format(statistic_name))
    # ax.view_init(30, 45)
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()
