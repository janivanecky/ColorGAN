import argparse
import torch
import time
import numpy as np
import cv2

import src.dataset as dataset
import src.networks as networks
import src.viz as viz

def generate(model_path, color_count, show_colors):
    torch.manual_seed(time.time())
    device = torch.device("cpu")

    # Set up generator
    generator = torch.load(model_path)
    latent_size = generator.input_size

    # Generate random colors
    z = networks.get_random_latent_vector(color_count, latent_size).to(device)
    generated_colors = generator.forward(z).cpu().data.numpy() * 0.5 + 0.5
    
    # Print header.
    color_count_in_palette = generated_colors.shape[2]
    print(' '.join(['COLOR' + str(i) for i in range(1, 1 + color_count_in_palette)]))
    print('=' * (7 * color_count_in_palette - 1))

    # Print colors in hex format
    for colors in generated_colors:
        colors = np.transpose(colors, (1, 0))
        colors = [dataset.float_to_hex(*color) for color in colors]
        print(' '.join(colors))

    # Show image with colors, if requested
    if show_colors:
        imgs = viz.colors_to_img(generated_colors)
        cv2.imshow("Generated colors", imgs)
        cv2.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to generator model file.")
    parser.add_argument("color_count", help="Number of colors to generate.", nargs="?", default=16, type=int)
    parser.add_argument("--show_colors", help="If specified, colors will be shown as image.", action="store_true")
    args = parser.parse_args()

    generate(args.model, args.color_count, args.show_colors)