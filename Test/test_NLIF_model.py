import sys

import numpy as np
import torch

import matplotlib.pyplot as plt

import util
from Models.NLIF import NLIF
from metrics import original_loss
import plot
from util import auto_encoder_task_input_output, feed_inputs_sequentially_return_tuple

torch.autograd.set_detect_anomaly(True)

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    params = {}
    snn = NLIF(N=30)
    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    # Delta = 0.1/snn.N
    Delta = 0.1

    t = 2400
    inputs, target_outputs = auto_encoder_task_input_output(t=t, period_ms=300, tau_syn=200., Delta = Delta)
    print('#inputs sum: {}'.format(inputs.sum()))
    print('#targets sum: {}'.format(target_outputs.sum()))
    # spikes, readouts = feed_inputs_sequentially_return_tuple(snn, inputs)
    # print('sum model outputs: {}'.format(readouts.sum()))

    spikes_zero_input, readouts_zero_input, v_zero_in, s_zero_in = feed_inputs_sequentially_return_tuple(snn, torch.zeros((t,2)))
    print('sum model outputs no input: {}'.format(readouts_zero_input.sum()))
    plot.plot_neuron(readouts_zero_input.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid='test', fname='test_plot_readouts_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot.plot_neuron(v_zero_in.detach().numpy(), ylabel='v', title='Test plot vs', uuid='test', fname='test_plot_v_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot.plot_spike_train(spikes_zero_input, title='Test spikes', uuid='test', fname='test_plot_spikes_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))

    # loss = original_loss(readouts, desired_output=target_outputs)
    # print('loss: {}'.format(loss))

    plot.plot_neuron(inputs.detach().numpy(), ylabel='input current', title='Test plot inputs', uuid='test', fname='test_plot_inputs_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot.plot_neuron(target_outputs.detach().numpy(), ylabel='target output', title='Test plot target outputs', uuid='test', fname='test_plot_itargets_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    # plot.plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid='test', fname='test_plot_readouts_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    # plot.plot_spike_train(spikes, title='Test spikes', uuid='test', fname='test_plot_spikes_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))

    optim_params = list(snn.parameters())
    optimiser = torch.optim.SGD(optim_params, lr=0.1)

    losses = []
    for i in range(1):
        optimiser.zero_grad()

        current_inputs = torch.tensor(inputs.clone().detach(), requires_grad=True)
        spikes, readouts, v, s = feed_inputs_sequentially_return_tuple(snn, current_inputs)
        print('sum model outputs: {}'.format(readouts.sum()))
        loss = original_loss(readouts, desired_output=target_outputs.clone().detach())
        print('loss: {}'.format(loss))
        try:
            loss.backward(retain_graph=True)
        except RuntimeError as re:
            print(re)

        # for p_i, param in enumerate(list(snn.parameters())):
        #     print('grad for param #{}: {}'.format(p_i, param.grad))

        optimiser.step()

        plot.plot_spike_train(spikes, title='Test spikes', uuid='test',
                              fname='test_plot_spikes_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid='test',
                         fname='test_plot_readouts_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_neuron(v.detach().numpy(), ylabel='membrane potential', title='Test plot $v$', uuid='test',
                         fname='test_plot_v_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_neuron(s.detach().numpy(), ylabel='syn currents', title='Test plot $s$', uuid='test',
                         fname='test_plot_s_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
        plt.close()

        losses.append(loss.clone().detach().data)
        util.release_computational_graph(model=snn, inputs=current_inputs)

    plot.plot_loss(losses, uuid='test', exp_type='default', fname='plot_loss_test')

    plot.plot_heatmap(snn.W_syn.data, ['W_syn_col', 'W_row'], uuid='test', exp_type='default', fname='test_heatmap_W')
    plot.plot_heatmap(snn.W_fast.data, ['W_fast_col', 'W_fast_row'], uuid='test', exp_type='default', fname='test_heatmap_W_fast')
    plot.plot_heatmap(snn.W_in.data, ['W_in_col', 'W_in_row'], uuid='test', exp_type='default', fname='test_heatmap_W_in')
    plot.plot_heatmap(snn.O.data, ['O_col', 'O_row'], uuid='test', exp_type='default', fname='test_heatmap_O')

sys.exit(0)
