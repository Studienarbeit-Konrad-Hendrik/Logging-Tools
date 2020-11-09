import torch
import torch.nn.functional as func
import matplotlib.pyplot as plt

#
# takes in two (b, dp) tensors
# b = batch size,
# dp = datapoints
def local_avg_distance(pred_data, target_data, filter_size=100):
    delta = torch.abs(pred_data - target_data)

    delta_expanded = delta.unsqueeze(1)

    weight = torch.ones((1, 1, filter_size)) / filter_size

    result = func.conv1d(delta_expanded, weight)

    return torch.mean(result.squeeze(1), dim=0)


def heatmap_avg_distance(pred_data, target_data, filter_size=100):
    data = local_avg_distance(pred_data, target_data, filter_size=filter_size)
    expanded = data.unsqueeze(0)

    x_vals = torch.arange(data.shape[0])

    fig, (ax,ax2) = plt.subplots(nrows=2, sharex=True)

    fig.set_figwidth(20)
    fig.suptitle("Average delta over "+str(filter_size)+" points")

    ax.imshow(expanded, cmap='magma', aspect='auto')
    ax.get_yaxis().set_visible(False)

    ax2.plot(x_vals, data)
    ax2.set_xlabel('timesteps')

    ax2.set_ylabel('avg. delta ('+str(filter_size)+' steps)')

    return fig


def heatmap_avg_distance_sigle_signal(pred_data, target_data, filter_size=100):
    data = local_avg_distance(pred_data, target_data, filter_size=filter_size)
    expanded = data.unsqueeze(0)

    x_vals = torch.arange(data.shape[0])
    x_vals_sig = torch.arange(target_data.shape[1]) * (data.shape[0] / target_data.shape[1])

    fig, (ax,ax2,ax3) = plt.subplots(nrows=3, sharex=True)

    fig.set_figwidth(20)
    fig.set_figheight(8)
    fig.suptitle("Average delta over "+str(filter_size)+" points")

    ax.imshow(expanded, cmap='magma', aspect='auto')
    ax.get_yaxis().set_visible(False)

    ax2.plot(x_vals, data)
    ax2.set_ylabel('avg. delta ('+str(filter_size)+' steps)')

    ax3.plot(x_vals_sig, target_data[0])
    ax3.set_xlabel('timesteps')
    ax3.set_ylabel("Amplitude")

    return fig
