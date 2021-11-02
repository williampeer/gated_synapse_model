import enum
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import IO
import plot
import util
from Models import NLIF_transposed
from metrics import original_loss

torch.autograd.set_detect_anomaly(True)


class ExpType(enum.Enum):
    AutoEncoding = 1
    GeneralPredictiveEncoding = 2


def main(argv):
    print('Argument List:', str(argv))

    learn_rate = 0.02
    # exp_type = ExpType.AutoEncoding
    exp_type = ExpType.GeneralPredictiveEncoding
    num_seeds = 1
    N = 30
    train_iters = 50
    plot_modulo = 10
    # lambda_regularize = 0.1 / N
    # lambda_regularize = 0.01 / N
    lambda_regularize = 0.01
    Delta = 1.
    # Delta = 0.1 / N
    period_ms = 40
    t = 1200
    tau_filter = 50.

    # Delta = 0.1/snn.N

    opts = [opt for opt in argv if opt.startswith("-")]
    args = [arg for arg in argv if not arg.startswith("-")]
    for i, opt in enumerate(opts):
        if opt == '-h':
            print('run_exp_NLIF_model.py -lr <learning-rate>')
            sys.exit()
        elif opt in ("-lr", "--learning-rate"):
            learn_rate = float(args[i])
        elif opt in ("-ti", "--training-iterations"):
            train_iters = int(args[i])
        elif opt in ("-nsds", "--num-seeds"):
            num_seeds = int(args[i])
        elif opt in ("-lregul", "--lambda-regularize"):
            lambda_regularize = float(args[i])
        elif opt in ("-N", "--network-size"):
            N = int(args[i])
        elif opt in ("-t", "--time-per-iteration"):
            t = float(args[i])
        elif opt in ("-D", "--Delta"):
            Delta = float(args[i])
        elif opt in ("-et", "--exp-type"):
            exp_type = ExpType[args[i]]

    for random_seed in range(3, 3+num_seeds):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        # snn = Models.NLIF.NLIF(N=N)
        snn = NLIF_transposed.NLIF(N=N)
        # snn = snn.double()
        print('- SNN test for class {} -'.format(snn.__class__.__name__))

        uuid = snn.__class__.__name__ + '/' + IO.dt_descriptor()

        A_in = torch.tensor([-1., 0.5])
        if exp_type is ExpType.AutoEncoding:
            period_ms = torch.tensor([period_ms, period_ms/2])
            inputs, target_outputs = util.auto_encoder_task_input_output(t=t, period_ms=period_ms, tau_filter=tau_filter,
                                                                         Delta=Delta, A_in=A_in)
        elif exp_type is ExpType.GeneralPredictiveEncoding:
            A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])
            inputs, target_outputs = util.general_predictive_encoding_task_input_output(t=t, period_ms=period_ms, tau_filter=tau_filter,
                                                                                        Delta=Delta, A_in=A_in, A_mat=A_mat)
        else:
            raise NotImplementedError("ExpType not in predefined enum.")

        print('#inputs sum: {}'.format(inputs.sum()))
        print('#targets sum: {}'.format(target_outputs.sum()))
        # spikes, readouts = feed_inputs_sequentially_return_tuple(snn, inputs)
        # print('sum model outputs: {}'.format(readouts.sum()))

        spikes_zero_input, readouts_zero_input, v_zero_in, s_zero_in, s_fast_zero_in = util.feed_inputs_sequentially_return_tuple(snn, torch.zeros((t,2), dtype=torch.double))
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
        # optimiser = torch.optim.SGD(optim_params, lr=learn_rate)
        optimiser = torch.optim.Adam(optim_params, lr=learn_rate)

        losses = []
        for i in range(train_iters):
            print('training iter: {}..'.format(i))
            optimiser.zero_grad()

            current_inputs = torch.tensor(inputs.clone().detach(), requires_grad=True)
            spikes, readouts, v, s, s_fast = util.feed_inputs_sequentially_return_tuple(snn, current_inputs)
            print('sum model outputs: {}'.format(readouts.sum()))
            loss = original_loss(readouts, desired_output=target_outputs.clone().detach(), lambda_regularize=lambda_regularize)
            print('loss: {}'.format(loss))
            try:
                loss.backward(retain_graph=True)
            except RuntimeError as re:
                print(re)

            # for p_i, param in enumerate(list(snn.parameters())):
            #     print('grad for param #{}: {}'.format(p_i, param.grad))
            print('W_fast.grad: {}'.format(snn.W_fast.grad))

            optimiser.step()

            if i % plot_modulo == 0 or i == train_iters-1:
                plot.plot_spike_train(spikes, title='Test spikes', uuid=uuid, exp_type=exp_type.name,
                                      fname='test_plot_spikes_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
                plot.plot_neuron(readouts.detach().numpy(), ylabel='readouts', title='Test plot readouts', uuid=uuid, exp_type=exp_type.name,
                                 fname='test_plot_readouts_train_iter_{}_{}'.format(i, snn.__class__.__name__) + '_' + str(random_seed))
                plt.close()

            losses.append(loss.clone().detach().data)
            util.release_computational_graph(model=snn, inputs=current_inputs)

        plot.plot_neuron(v.detach().numpy(), ylabel='membrane potential', title='Test plot $v$', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_v_{}_seed_{}'.format(snn.__class__.__name__, random_seed), legend=['v'])
        plot.plot_neuron(s.detach().numpy(), ylabel='syn currents', title='Test plot $s$', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_s_{}_seed_{}'.format(snn.__class__.__name__, random_seed), legend=['s'])
        plot.plot_neuron(torch.hstack([readouts, target_outputs]).data, ylabel='outputs', title='Outputs', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_outputs_{}_seed_{}'.format(snn.__class__.__name__, random_seed), legend=['readouts', '', 'targets', ''])
        plot.plot_neuron(torch.vstack([snn.W_in.matmul(current_inputs.T)[6], (snn.W_fast.matmul(s_fast.T))[6]]).T.data, ylabel='input components', title='Test input components', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_input_components_{}_seed_{}'.format(snn.__class__.__name__, random_seed), legend=['$Ui$', '$W_f s_f$'])
        tot_input_current = snn.W_in.matmul(current_inputs.T)[6] + (snn.W_fast.matmul(s_fast.T))[6]
        plot.plot_neuron(tot_input_current.data, ylabel='total input current', title='Test plot $I_{tot}$', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_I_tot_{}_seed_{}'.format(snn.__class__.__name__, random_seed), legend=['$I_{tot}$'])
        plot.plot_neuron(torch.vstack([v[:,6], snn.W_in.matmul(current_inputs.T)[6]]).T.detach().numpy(), ylabel='membrane potential components', title='Test plot $v$', uuid=uuid,
                         exp_type=exp_type.name, fname='test_plot_mem_voltage_single_neuron_{}_seed_'.format(snn.__class__.__name__) + '_' + str(random_seed), legend=['$v$', '$dv_{in}$'])
        plot.plot_weights_dots(snn.W_in.data, snn.O.T.data, uuid, exp_type.name)

        cur_fname = '{}_exp_{}_random_seed_{}'.format(snn.__class__.__name__, 'auto_encode', random_seed)
        IO.save(snn, loss=losses, uuid=uuid, fname=cur_fname)

        plot.plot_loss(losses, uuid=uuid, exp_type=exp_type.name, custom_title='Loss, lr={}, optim: {}'.format(learn_rate, optimiser.__class__.__name__), fname='plot_loss_test')

        def sort_matrix_wrt_weighted_centers(mat):
            center_tuples = []
            for row_i in range(mat.shape[0]):
                weighted_center = int((mat[row_i,:] * torch.arange(mat.shape[1])).sum()/mat.shape[1])
                center_tuples.append((weighted_center, row_i))
            center_tuples.sort()
            new_mat = torch.zeros_like(mat)
            for row_i in range(mat.shape[0]):
                new_mat[row_i] = mat[center_tuples[row_i][1]]
            return new_mat

        plot.plot_heatmap(snn.W_syn.data, ['W_syn_col', 'W_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_W')
        plot.plot_heatmap(snn.W_fast.data, ['W_fast_col', 'W_fast_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_W_fast')
        plot.plot_heatmap((- snn.W_in.matmul(snn.O)).data, ['-UO column', '-UO row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_minUO')
        plot.plot_heatmap((- snn.W_in.matmul(snn.O)).T.data, ['-UO.T column', '-UO.T row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_minUO_T')

        sorted_W_fast = sort_matrix_wrt_weighted_centers(snn.W_fast)
        sorted_minUO = sort_matrix_wrt_weighted_centers(-snn.W_in.matmul(snn.O))

        plot.plot_heatmap(sorted_W_fast.data, ['sorted_W_fast_col', 'sorted_W_fast_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_z_sorted_W_fast')
        plot.plot_heatmap(sorted_minUO.data, ['sorted_minUO_col', 'sorted_minUO_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_z_sorted_minUO')

        plot.plot_heatmap(snn.W_in.data, ['W_in_col', 'W_in_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_2_W_in')
        plot.plot_heatmap(snn.O.T.data, ['O_col', 'O_row'], uuid=uuid, exp_type=exp_type.name, fname='test_heatmap_2_O_T')

        return snn


if __name__ == "__main__":
    snn = main(sys.argv[1:])
    # sys.exit(0)
