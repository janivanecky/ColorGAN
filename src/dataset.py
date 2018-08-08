import json

import numpy as np
import torch
import torch.utils.data

def hex_to_float(hex_string):
    hex_r = hex_string[:2]
    hex_g = hex_string[2:4]
    hex_b = hex_string[4:]
    r = float(int(hex_r, 16)) / 255.0
    g = float(int(hex_g, 16)) / 255.0
    b = float(int(hex_b, 16)) / 255.0
    return r, g, b

def float_to_hex(r, g, b):
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    hex_string = "%0.2X%0.2X%0.2X" % (r, g, b)
    return hex_string

class Dataset(torch.utils.data.Dataset):
    def __init__(self, json_path, shuffle_colors=False):
        self.shuffle_colors = shuffle_colors
        self.samples = self.load_samples(json_path)

    @staticmethod
    def load_samples(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        samples = data['data']
        samples = [[hex_to_float(color) for color in palette] for palette in samples]
        samples = np.array(samples, dtype=np.float32)
        samples = samples * 2.0 - 1.0
        samples = np.transpose(samples, (0, 2, 1)) # (N, W, C) -> (N, C, W)
        return samples

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        palette = self.samples[index]
        if self.shuffle_colors:
            # np.random.permutation acts on the first dimension only, so we need to tranpose before and after.
            palette = np.transpose(palette, (1, 0)) # (N, C) -> (C, N)
            palette = np.random.permutation(palette)
            palette = np.transpose(palette, (1, 0)) # (C, N) -> (N, C)
        return palette