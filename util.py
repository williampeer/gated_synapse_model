import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    var_a, model_spiketrain = model(inputs[0])
    for x_in in inputs[1:]:
        readout, spikes = model(x_in)
        var_a = torch.vstack([var_a, readout])
        model_spiketrain = torch.vstack([model_spiketrain, spikes])

    return var_a, model_spiketrain


# low-pass filter
def auto_encoder_task_input_output(t=4800, tau_syn=10., Delta = 0.1):
    period = 1200
    t = torch.reshape(torch.arange(t), (-1, 1))
    input = Delta * torch.sin(period * t)
    out_dot = torch.torch.tensor([input[0]/tau_syn])
    for t_i in range(t.shape[0]-1):
        out_dot = torch.vstack([out_dot, out_dot[-1]+(input[t_i]-out_dot[-1])/tau_syn])
    return (torch.ones((2,)) * input, torch.ones((2,)) * out_dot)


def general_predictive_coding_task_input_output():
    pass
