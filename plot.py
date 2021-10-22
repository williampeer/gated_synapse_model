import matplotlib.pyplot as plt
import torch
import numpy as np
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
                ylabel='Membrane potential', fname='plot_neuron_test', use_legend=False):
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
    plt.xticks(np.arange(0, len(heat_mat)))
    plt.yticks(np.arange(0, len(heat_mat)))
    plt.ylabel(axes[0])
    plt.xlabel(axes[1])
    # plt.show()
    plt.savefig(fname=full_path + fname)
    plt.close()


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
