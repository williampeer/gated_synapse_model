import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    var_a, model_spiketrain, vs, ss = model(inputs[0])
    for x_in in inputs[1:]:
        readout, spikes, v, s = model(x_in)
        var_a = torch.vstack([var_a, readout])
        model_spiketrain = torch.vstack([model_spiketrain, spikes])
        vs = torch.vstack([vs, v])
        ss = torch.vstack([ss, s])

    return var_a, model_spiketrain, vs, ss


# low-pass filter
def auto_encoder_task_input_output(t=4800, period_ms=1200, tau_syn=200., Delta = 1.):
    period_rads = (3.141592 / period_ms)
    input = Delta * torch.sin(period_rads * torch.reshape(torch.arange(0, t), (t, 1)))
    out_dot = torch.torch.tensor([input[0]/tau_syn])
    for t_i in range(t-1):
        out_dot = torch.vstack([out_dot, out_dot[-1]+(input[t_i]-out_dot[-1])/tau_syn])
    return (torch.ones((2,)) * input, torch.ones((2,)) * out_dot)


def general_predictive_coding_task_input_output():
    pass


def release_computational_graph(model, inputs=None):
    if model is not None:
        model.reset()
    if inputs is not None and hasattr(inputs, 'grad'):
        inputs.grad = None
