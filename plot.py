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
                ylabel='Membrane potential', fname='plot_neuron_test'):
    data = {'membrane_potentials_through_time': membrane_potentials_through_time, 'title': title, 'uuid': uuid,
            'exp_type': exp_type, 'ylabel': ylabel, 'fname': fname}
    IO.save_plot_data(data=data, uuid='test_uuid', plot_fn='plot_neuron')
    legend = []
    for i in range(len(membrane_potentials_through_time)):
        legend.append('N.{}'.format(i+1))
    plt.figure()
    plt.plot(np.arange(membrane_potentials_through_time.shape[0]), membrane_potentials_through_time)
    plt.legend(legend, loc='upper left', ncol=4)
    # plt.title(title)
    plt.xlabel('Time (ms)')
    plt.ylabel(ylabel)
    # plt.show()
    full_path = './figures/' + exp_type + '/' + uuid + '/'
    IO.makedir_if_not_exists(full_path)
    plt.savefig(fname=full_path + fname)


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