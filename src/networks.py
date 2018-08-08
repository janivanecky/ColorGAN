import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F


def get_random_latent_vector(batch_size, latent_size):
    result = torch.randn(batch_size, latent_size)
    return result


def std_layer(x):
    N = x.shape[0]                                         # N = number of samples
    mean = x.mean(dim=0)                                   # Compute mean of samples
    variance = ((x - mean) ** 2).mean(dim=0)               # Compute variance of samples
    std = (variance + 1e-8).sqrt() * np.sqrt(N / (N - 1))  # Compute stddev, With Bessel's correction
    avg = std.mean().view(1, 1).repeat(x.size()[0], 1)     # Average over individual color positions
    x = torch.cat((x, avg), 1)                             # Concatenate averaged std with input data
    return x


def average_generators(g1, g2, g_out, beta):
    params1 = dict(g1.named_parameters())
    params2 = dict(g2.named_parameters())

    params_out = dict(g_out.named_parameters())
    for weight_name in params1.keys():
        params_out[weight_name].data.copy_(beta * params1[weight_name].data + (1 - beta) * params2[weight_name].data)

    g_out.load_state_dict(params_out)


class Generator(torch.nn.Module):
    def __init__(self, latent_size, layer_sizes, output_color_count, activation_fn=F.relu):
        super(Generator, self).__init__()
        self.activation_fn = activation_fn
        self.output_color_count = output_color_count
        self.input_size = latent_size

        input_sizes = [latent_size] + layer_sizes
        output_sizes = layer_sizes + [output_color_count * 3]
        self.layers = []
        for input_size, output_size in zip(input_sizes, output_sizes):
            self.layers.append(torch.nn.Linear(input_size, output_size))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # Output layer uses tanh as activation,
            # the rest of layers can use different activations.
            if i == len(self.layers) - 1:
                x = F.tanh(layer(x))
            else:
                x = self.activation_fn(layer(x))
        x = x.view(-1, 3, self.output_color_count) #  (N, C * W) -> (N, C, W)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self, layer_sizes, input_color_count, activation_fn=F.relu):
        super(Discriminator, self).__init__()
        self.activation_fn = activation_fn
        self.input_color_count = input_color_count
        
        # ``+ 1`` is to account for std feature from std_layer
        input_sizes = [input_color_count * 3] + [layer_size + 1 for layer_size in layer_sizes]
        output_sizes = layer_sizes + [1]
        self.layers = []
        for input_size, output_size in zip(input_sizes, output_sizes):
            self.layers.append(torch.nn.Linear(input_size, output_size))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        x = x.view(-1, self.input_color_count * 3)  # Flatten (N, C, W) -> (N, C * W)
        for i, layer in enumerate(self.layers):
            # Don't apply std_layer to input (image)
            if i > 0:
                x = std_layer(x)
            x = layer(x)

            # Apply activation function to all layers except the output layer.
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
        return x