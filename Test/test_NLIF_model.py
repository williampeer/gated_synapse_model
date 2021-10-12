import sys

import numpy as np
import torch

from Models.NLIF import NLIF

for random_seed in range(3, 4):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    snn = NLIF()

    sample_inputs = sine_modulated_white_noise(t=5000, N=snn.N)
    print('- SNN test for class {} -'.format(snn.__class__.__name__))
    print('#inputs: {}'.format(sample_inputs.sum()))
    _, sample_targets = model_util.feed_inputs_sequentially_return_tuple(snn, sample_inputs)
    sample_targets = sample_targets.clone().detach()

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
    plot_spike_train_projection(spikes, fname='test_projection_{}_ext_input'.format(snn.__class__.__name__) + '_' + str(random_seed))

sys.exit(0)
