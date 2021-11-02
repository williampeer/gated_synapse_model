import torch


def feed_inputs_sequentially_return_tuple(model, inputs):
    print('Feeding {} inputs sequentially through SNN in time'.format(inputs.size(0)))
    spikes, readouts, vs, ss, s_fasts = model(inputs[0])
    for x_in in inputs[1:]:
        spiked, readout, v, s, s_fast = model(x_in)
        spikes = torch.vstack([spikes, spiked])
        readouts = torch.vstack([readouts, readout])
        vs = torch.vstack([vs, v])
        ss = torch.vstack([ss, s])
        s_fasts = torch.vstack([s_fasts, s_fast])

    return spikes, readouts, vs, ss, s_fasts


# low-pass filter
def auto_encoder_task_input_output(t=2400, period_ms=50, tau_filter=100., Delta = 1., A_in = 0.):
    period_rads = (3.141592 / period_ms)
    input = Delta * A_in * torch.sin(period_rads * torch.reshape(torch.arange(0, t).double(), (t, 1)))
    out_dot = input[0,:]/tau_filter
    out_dot = torch.vstack([out_dot, out_dot])
    for t_i in range(t-1):
        dv_out = (-out_dot[-1, :] + input[t_i, :]) / tau_filter
        out_next = out_dot[-1, :] + dv_out
        out_dot = torch.vstack([out_dot, out_next])
    return (input, out_dot[1:,:])


# Linear dynamic relationships between desired I-O signals.
def general_predictive_encoding_task_input_output(t=2400, period_ms=50, tau_filter=10., Delta = 1.,
                                                  A_in=torch.tensor([1.0, 0.5]),
                                                  A_mat = torch.tensor([[-0.7, 0.36], [-2.3, -0.1]])):
    period_rads = (3.141592 / period_ms)
    assert A_mat is not None and len(A_mat.shape) == 2, "A_mat must be defined and not none."
    input = Delta * A_in * torch.sin(period_rads * torch.reshape(torch.arange(0, t).double(), (t, 1)))
    outputs = (input[0,:])/tau_filter
    outputs = torch.vstack([outputs, outputs])
    for t_i in range(t-1):
        dv_out = (A_mat.matmul(outputs[-1,:]) - outputs[-1,:] + input[t_i, :]) / tau_filter
        out_next = outputs[-1,:] + dv_out
        outputs = torch.vstack([outputs, out_next])
    return (input, outputs[1:,:])


def release_computational_graph(model, inputs=None):
    if model is not None:
        model.reset()
    if inputs is not None and hasattr(inputs, 'grad'):
        inputs.grad = None
