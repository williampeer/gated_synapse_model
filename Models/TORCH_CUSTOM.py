from Log import Logger

LOG = Logger('gradient_clamping_errors')


def static_clamp_for(new_grad, l, m, p):
    for i in range(p.shape[0]):
        p_out_of_bounds = p[i] < l or m < p[i]
        if p_out_of_bounds:
            LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p[i], l, m, new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m, p[i] - l)  # the gradient is subtracted. test with Adam too.

    return new_grad.data


def static_clamp_for_scalar(new_grad, l, m, p):
    p_out_of_bounds = p < l or m < p
    if p_out_of_bounds:
        LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p, l, m, new_grad))
    new_grad.data.clamp_(p - m, p - l)  # the gradient is subtracted. test with Adam too.

    return new_grad.data


def static_clamp_for_matrix(new_grad, l, m, p):
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p_out_of_bounds = p[i][j] < l or m < p[i][j]
            if p_out_of_bounds:
                LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p[i][j], l, m, new_grad[i][j]))
            new_grad[i].data.clamp_(p[i][j] - m, p[i][j] - l)  # the gradient is subtracted. test with Adam too.

    return new_grad.data


def static_clamp_for_vector_bounds(new_grad, l, m, p):
    for i in range(p.shape[0]):
        p_out_of_bounds = p[i] < l[i] or m[i] < p[i]
        if p_out_of_bounds:
            LOG.log('PARAMETER OUT OF BOUNDS. \ni: {}, p: {},\nl: {}, m: {},\ngrad: {}'.format(i, p[i], l[i], m[i], new_grad[i]))
        new_grad[i].data.clamp_(p[i] - m[i], p[i] - l[i])  # the gradient is subtracted. test with Adam too.

    return new_grad.data
