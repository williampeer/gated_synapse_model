import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    membrane_potentials, model_spiketrain = model(inputs[0])
    for x_in in inputs[1:]:
        v, spikes = model(x_in)
        membrane_potentials = torch.vstack([membrane_potentials, v])
        model_spiketrain = torch.vstack([model_spiketrain, spikes])

    return membrane_potentials, model_spiketrain


# low-pass filter
def auto_encoder_task_input_output(t=4800, tau_syn=10., Delta = 0.1):
    period = 1200
    t = torch.reshape(torch.arange(t), (-1, 1))
    input = Delta * torch.sin(period * t)
    out_dot = torch.tensor([])
    out_dot = torch.vstack([out_dot, input[0]/tau_syn])
    for t_i in range(t.shape[0]-1):
        out_dot = torch.vstack([out_dot, (out_dot[-1]+input[t_i])/tau_syn])
    return (input, out_dot)


def general_predictive_coding_task_input_output():
    pass
