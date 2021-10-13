import sys

import numpy as np
import torch

from Models.NLIF import NLIF
from metrics import original_loss
from plot import plot_spike_train_projection, plot_spike_train, plot_neuron
from util import auto_encoder_task_input_output, feed_inputs_sequentially_return_tuple

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    params = {}
    snn = NLIF(N=30)

    inputs, target_outputs = auto_encoder_task_input_output(t=1200)
    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(inputs.sum()))
    spikes, readouts = feed_inputs_sequentially_return_tuple(snn, inputs)
    print('sum model outputs: {}'.format(readouts.sum()))

    loss = original_loss(readouts, desired_output=target_outputs)

    print('loss: {}'.format(loss))
    plot_neuron(inputs.detach().numpy(), ylabel='input current', title='Test plot inputs', uuid='test', fname='test_plot_inputs_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot_neuron(target_outputs.detach().numpy(), ylabel='target output', title='Test plot target outputs', uuid='test', fname='test_plot_targets_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid='test', fname='test_plot_readouts_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
    # plot_spike_train(spikes, title='Test spikes', uuid='test', fname='test_plot_spikes_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))

    # optim_params = list(snn.parameters())
    # optimiser = torch.optim.SGD(optim_params, lr=0.015)
    # optimiser.zero_grad()
    #
    # for i in range(3):
    #     current_inputs = sine_modulated_white_noise(t=5000, N=snn.N)
    #     current_inputs.retain_grad()
    #
    #     spike_probs, spikes = model_util.feed_inputs_sequentially_return_tuple(snn, current_inputs)
    #
    #     m = torch.distributions.bernoulli.Bernoulli(spike_probs)
    #     loss = -m.log_prob(sample_targets).sum()
    #
    #     loss.backward(retain_graph=True)
    #
    #     for p_i, param in enumerate(list(snn.parameters())):
    #         print('grad for param #{}: {}'.format(p_i, param.grad))
    #
    # optimiser.step()
    # hard_thresh_spikes_sum = torch.round(spikes).sum()
    # print('spikes sum: {}'.format(hard_thresh_spikes_sum))
    # soft_thresh_spikes_sum = (spikes > 0.333).sum()
    # zero_thresh_spikes_sum = (spikes > 0).sum()
    # print('thresholded spikes sum: {}'.format(torch.round(spikes).sum()))
    # print('=========avg. hard rate: {}'.format(1000*hard_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    # print('=========avg. soft rate: {}'.format(1000*soft_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))
    # print('=========avg. zero thresh rate: {}'.format(1000*zero_thresh_spikes_sum / (spikes.shape[1] * spikes.shape[0])))

sys.exit(0)
