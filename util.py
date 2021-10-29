import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    spikes, readouts, vs, ss = model(inputs[0])
    for x_in in inputs[1:]:
        spiked, readout, v, s = model(x_in)
        spikes = torch.vstack([spikes, spiked])
        readouts = torch.vstack([readouts, readout])
        vs = torch.vstack([vs, v])
        ss = torch.vstack([ss, s])

    return spikes, readouts, vs, ss


# low-pass filter
def auto_encoder_task_input_output(t=2400, period_ms=50, tau_filter=100., Delta = 1., A_mat = torch.tensor([1., 0.5])):
    period_rads = (3.141592 / period_ms)
    input = Delta * torch.sin(period_rads * torch.reshape(torch.arange(0, t), (t, 1)))
    out_dot = torch.torch.tensor([input[0]/tau_filter])
    for t_i in range(t-1):
        out_dot = torch.vstack([out_dot, out_dot[-1]+(input[t_i]-out_dot[-1])/tau_filter])
    return (A_mat * (input), A_mat * (out_dot))


# Linear dynamic relationships between desired I-O signals.
def general_predictive_encoding_task_input_output(t=2400, period_ms=50, tau_filter=100., Delta = 1.,
                                                  A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])):
    period_rads = (3.141592 / period_ms)
    assert A_mat is not None and len(A_mat.shape) == 2, "A_mat must be defined and not none."
    input = Delta * torch.sin(torch.ones((2, 1)) * period_rads * torch.arange(0, t))
    out_dot = input[:,0]/tau_filter
    out_dot = torch.vstack([out_dot, out_dot])
    for t_i in range(t-1):
        # A_mat.matmul(
        dv_out = (A_mat.matmul(out_dot[-1,:]) - out_dot[-1,:] + input[:,t_i]) / tau_filter
        out_next = out_dot[-1,:] + dv_out
        out_dot = torch.vstack([out_dot, out_next])
    return (input.T, out_dot[1:,:])


def release_computational_graph(model, inputs=None):
    if model is not None:
        model.reset()
    if inputs is not None and hasattr(inputs, 'grad'):
        inputs.grad = None
