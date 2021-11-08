import torch


def euclid_norm(mat):
    return torch.sqrt((mat * mat).sum() + 1e-12)


def original_loss(output, desired_output, lambda_regularize=0.1):
    overall_activity = output
    assert output.shape[0] > output.shape[1]
    rate_scale = output.shape[0] / 1000.
    # loss = (euclid_norm(output-desired_output) + lambda_regularize*rate_scale*euclid_norm(overall_activity)) / 2.
    loss = (euclid_norm(output-desired_output) + lambda_regularize*euclid_norm(overall_activity)) / 2.
    return loss


def euclid_dist(vec1, vec2):
    pow_res = torch.pow(torch.sub(vec2, vec1), 2)
    return torch.sqrt(pow_res.sum() + 1e-12)


def firing_rate_distance(model_spikes, target_spikes):
    mean_model_rate = model_spikes.sum(dim=0) * 1000. / model_spikes.shape[0]  # Hz
    mean_targets_rate = target_spikes.sum(dim=0) * 1000. / target_spikes.shape[0]  # Hz
    return euclid_dist(mean_targets_rate, mean_model_rate)


# an approximation using torch.where
def torch_van_rossum_convolution(spikes, tau):
    decay_kernel = torch.exp(-torch.tensor(1.) / tau)
    convolved_spiketrain = spikes.clone()
    one_row_of_zeros = torch.zeros((1, spikes.shape[1]))
    for i in range(int(3*tau)):
        tmp_shifted_conv = torch.cat([one_row_of_zeros, convolved_spiketrain[:-1]])
        convolved_spiketrain = torch.where(spikes < 0.5, tmp_shifted_conv.clone() * decay_kernel, spikes.clone())
    return convolved_spiketrain


def van_rossum_dist(spikes, target_spikes, tau):
    c1 = torch_van_rossum_convolution(spikes=spikes, tau=tau)
    c2 = torch_van_rossum_convolution(spikes=target_spikes, tau=tau)
    return euclid_dist(c1, c2)
