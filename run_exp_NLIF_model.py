import enum
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import IO
import plot
import util
from Models.NLIF import NLIF
from metrics import original_loss

torch.autograd.set_detect_anomaly(True)


class ExpType(enum.Enum):
    AutoEncoding = 1
    GeneralPredictiveEncoding = 2


def main(argv):
    print('Argument List:', str(argv))

    learn_rate = 0.005
    # exp_type = ExpType.AutoEncoding
    exp_type = ExpType.GeneralPredictiveEncoding
    num_seeds = 1
    N = 30
    lambda_regularize = 0.1 / N
    t = 2400
    # Delta = 0.1/snn.N

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_exp_NLIF_model.py -lr <learning-rate>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-nsds", "--num-seeds"):
            num_seeds = int(args[i])
        elif opt in ("-lregul", "--lambda-regularize"):
            lambda_regularize = float(args[i])
        elif opt in ("-N", "--network-size"):
            N = int(args[i])
        elif opt in ("-t", "--time-per-iteration"):
            t = float(args[i])
        elif opt in ("-et", "--exp-type"):
            exp_type = ExpType[args[i]]

    for random_seed in range(3, 3+num_seeds):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        snn = NLIF(N=N)
        print('- SNN test for class {} -'.format(snn.__class__.__name__))

        uuid = snn.__class__.__name__ + '/' + IO.dt_descriptor()

        if exp_type is ExpType.AutoEncoding:
            Delta = 1.
            inputs, target_outputs = util.auto_encoder_task_input_output(t=t, period_ms=600, tau_syn=200., Delta = Delta)
        elif exp_type is ExpType.GeneralPredictiveEncoding:
            A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])
            inputs, target_outputs = util.general_predictive_encoding_task_input_output(t=t, period_ms=600, tau_syn=200., A_mat=A_mat)
        else:
            raise NotImplementedError("ExpType not in predefined enum.")

        print('#inputs sum: {}'.format(inputs.sum()))
        print('#targets sum: {}'.format(target_outputs.sum()))
        # spikes, readouts = feed_inputs_sequentially_return_tuple(snn, inputs)
        # print('sum model outputs: {}'.format(readouts.sum()))

        spikes_zero_input, readouts_zero_input, v_zero_in, s_zero_in = util.feed_inputs_sequentially_return_tuple(snn, torch.zeros((t,2)))
        print('sum model outputs no input: {}'.format(readouts_zero_input.sum()))
        plot.plot_neuron(readouts_zero_input.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid=uuid, exp_type=exp_type.name, fname='test_plot_readouts_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_neuron(v_zero_in.detach().numpy(), ylabel='v', title='Test plot vs', uuid=uuid, exp_type=exp_type.name, fname='test_plot_v_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_spike_train(spikes_zero_input, title='Test spikes', uuid=uuid, exp_type=exp_type.name, fname='test_plot_spikes_no_input_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))

        # loss = original_loss(readouts, desired_output=target_outputs)
        # print('loss: {}'.format(loss))

        plot.plot_neuron(inputs.detach().numpy(), ylabel='input current', title='Test plot inputs', uuid=uuid, exp_type=exp_type.name, fname='test_plot_inputs_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
        plot.plot_neuron(target_outputs.detach().numpy(), ylabel='target output', title='Test plot target outputs', uuid=uuid, exp_type=exp_type.name, fname='test_plot_itargets_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
        # plot.plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid=uuid, fname='test_plot_readouts_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))
        # plot.plot_spike_train(spikes, title='Test spikes', uuid=uuid, fname='test_plot_spikes_{}'.format(snn.__class__.__name__) + '_' + str(random_seed))

        optim_params = list(snn.parameters())
        optimiser = torch.optim.SGD(optim_params, lr=learn_rate)

        losses = []
        for i in range(50):
            print('training iter: {}..'.format(i))
            optimiser.zero_grad()

            current_inputs = torch.tensor(inputs.clone().detach(), requires_grad=True)
            spikes, readouts, v, s = util.feed_inputs_sequentially_return_tuple(snn, current_inputs)
            print('sum model outputs: {}'.format(readouts.sum()))
            loss = original_loss(readouts, desired_output=target_outputs.clone().detach(), lambda_regularize=lambda_regularize)
            print('loss: {}'.format(loss))
            try:
                loss.backward(retain_graph=True)
            except RuntimeError as re:
                print(re)

            # for p_i, param in enumerate(list(snn.parameters())):
            #     print('grad for param #{}: {}'.format(p_i, param.grad))

            optimiser.step()

            plot.plot_spike_train(spikes, title='Test spikes', uuid=uuid, exp_type=exp_type.name,
                                  fname='test_plot_spikes_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
            plot.plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid=uuid, exp_type=exp_type.name,
                             fname='test_plot_readouts_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
            plot.plot_neuron(v.detach().numpy(), ylabel='membrane potential', title='Test plot $v$', uuid=uuid, exp_type=exp_type.name,
                             fname='test_plot_v_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
            plot.plot_neuron(s.detach().numpy(), ylabel='syn currents', title='Test plot $s$', uuid=uuid, exp_type=exp_type.name,
                             fname='test_plot_s_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
            plt.close()

            losses.append(loss.clone().detach().data)
            util.release_computational_graph(model=snn, inputs=current_inputs)

        cur_fname = '{}_exp_{}_random_seed_{}'.format(snn.__class__.__name__, 'auto_encode', random_seed)
        IO.save(snn, loss=losses, uuid=uuid, fname=cur_fname)

        plot.plot_loss(losses, uuid=uuid, exp_type=exp_type.name, fname='plot_loss_test')

        plot.plot_heatmap(snn.W_syn.data, ['W_syn_col', 'W_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_W')
        plot.plot_heatmap(snn.W_fast.data, ['W_fast_col', 'W_fast_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_W_fast')
        plot.plot_heatmap(snn.W_in.data, ['W_in_col', 'W_in_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_W_in')
        plot.plot_heatmap(snn.O.data, ['O_col', 'O_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_O')


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.exit(0)
