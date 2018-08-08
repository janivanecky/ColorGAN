import argparse
import cv2
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

import torch
import torch.utils.data
import torch.nn.functional as F

import src.dataset as dataset
import src.loss as loss
import src.metrics as metrics
import src.networks as networks
import src.utils as utils
import src.viz as viz

# Constants
COLOR_COUNT = 5
VIZ_SAMPLE_COUNT = 32
GP_LAMBDA = 10
DEFAULT_DATASET_PATH = "color_dataset.json"
LOSS_TYPES = ['GAN', 'RGAN']
LOSS_MAPPING_GEN = {
    'GAN': loss.loss_gen,
    'RGAN': loss.loss_gen_rgan,
}
LOSS_MAPPING_DIS = {
    'GAN': loss.loss_dis,
    'RGAN': loss.loss_dis_rgan,
}

# Default values for parameters
LATENT_SIZE = 32
LAYER_SIZES = [32, 64]
LOSS = 'GAN'
SHUFFLE_COLORS = False
BATCH_SIZE = 32
EPOCH_COUNT = 10000
VALIDATION_PERIOD = 100
LEARNING_RATE = 1e-4
GEN_AVG_BETA = 0.9999


def train(dataset_path,
          experiment_dir,
          batch_size=BATCH_SIZE,
          latent_size=LATENT_SIZE,
          layer_sizes=LAYER_SIZES,
          shuffle_colors=SHUFFLE_COLORS,
          loss_type=LOSS,
          epoch_count=EPOCH_COUNT,
          learning_rate=LEARNING_RATE,
          validation_period=VALIDATION_PERIOD,
          show_progress=True,
          random_seed=42):
    device = torch.device("cpu")
    torch.manual_seed(random_seed)
    if show_progress:
        plt.ion()

    # Open handle to logging file.
    log_file = open(utils.get_log_save_path(experiment_dir), 'w')
    os.mkdir(utils.get_color_save_dir_path(experiment_dir))

    # Set up dataset.
    color_dataset = dataset.Dataset(dataset_path, shuffle_colors)
    data_loader = torch.utils.data.DataLoader(color_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Set up networks.
    generator = networks.Generator(latent_size, layer_sizes, COLOR_COUNT).to(device)
    generator_avg = networks.Generator(latent_size, layer_sizes, COLOR_COUNT).to(device)
    discriminator = networks.Discriminator(layer_sizes[::-1], COLOR_COUNT).to(device)

    # Set up optimizers.
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Set up loss functions.
    assert loss_type in LOSS_TYPES, "Invalid loss type specified {}, must be in {}.".format(loss_type, LOSS_TYPES)
    loss_generator = LOSS_MAPPING_GEN[loss_type]
    loss_discriminator = LOSS_MAPPING_DIS[loss_type]

    # Set up data used for testing.
    test_set = color_dataset.samples
    test_noise = networks.get_random_latent_vector(len(test_set), latent_size).to(device)

    wds, best_wd, best_epoch = [], 1.0, 0
    for epoch in range(0, epoch_count + 1):
        # Train for one epoch.
        for batch in data_loader:
            # Generate fake colors and feed them to discriminator.
            batch_size = batch.size()[0]
            z = networks.get_random_latent_vector(batch_size, latent_size).to(device)
            fake_colors = generator.forward(z)
            discriminator_out_fake = discriminator.forward(fake_colors)

            # Feed real samples to discriminator.
            real_colors = batch.to(device)
            discriminator_out_real = discriminator.forward(real_colors)

            # Update generator.
            optimizer_g.zero_grad()
            loss_g = loss_generator(discriminator_out_fake, discriminator_out_real)
            loss_g.backward(retain_graph=True)
            optimizer_g.step()

            # Update discriminator.
            optimizer_d.zero_grad()
            loss_d = loss_discriminator(discriminator_out_fake, discriminator_out_real)
            loss_d.backward()
            optimizer_d.step()

            # Average generators.
            networks.average_generators(generator_avg, generator, generator_avg, GEN_AVG_BETA)
        
        if epoch % validation_period == 0:
            # Compute wasserstein distance for averaged generator.
            test_colors_generated_avg = generator_avg.forward(test_noise).cpu().data.numpy().reshape((-1, 3 * COLOR_COUNT))
            test_colors_real = test_set.reshape((-1, 3 * COLOR_COUNT))
            wd = metrics.sliced_wasserstein_distance(test_colors_generated_avg, test_colors_real, 1024, COLOR_COUNT)
            wds.append(wd)

            # Print current epoch's status and store into log.
            status = "EPOCH: {}; WD: {}".format(epoch, wd)
            if show_progress:
                print(status)
            log_file.write(status + '\n')
            log_file.flush()

            # Plot wasserstein distance progress.
            wd_plot_x = np.arange(0, epoch + 1, validation_period)
            wd_plot_y = wds
            plt.plot(wd_plot_x, wd_plot_y)
            plt.xlabel("Epochs")
            plt.ylabel("Sliced Wasserstein Distance")
            plt.savefig(utils.get_plot_save_path(experiment_dir))
            if show_progress:
                plt.show()
                plt.pause(0.001)
            else:
                plt.clf()

            # Create comparison image comparing colors generated by current
            # generator and averaged generator to colors from the training set.
            noise_viz = test_noise[:VIZ_SAMPLE_COUNT]
            generated_colors = generator.forward(noise_viz).cpu().data.numpy() * 0.5 + 0.5
            generated_colors = viz.colors_to_img(generated_colors)

            generated_colors_avg = generator_avg.forward(noise_viz).cpu().data.numpy() * 0.5 + 0.5
            generated_colors_avg = viz.colors_to_img(generated_colors_avg)

            real_colors = test_set[:VIZ_SAMPLE_COUNT] * 0.5 + 0.5
            real_colors = viz.colors_to_img(real_colors)

            comparison_img = viz.create_comparison_img(generated_colors, generated_colors_avg, real_colors)
            cv2.imwrite(utils.get_color_save_path(experiment_dir, epoch), comparison_img)
            if show_progress:
                cv2.imshow('Generated colors comparison', comparison_img)

            # If the averaged generator achieved lowest wasserstein distance, save its snapshot.
            if wd < best_wd:
                if epoch > 0:
                    os.remove(utils.get_model_save_path(experiment_dir, best_epoch))
                torch.save(generator_avg, utils.get_model_save_path(experiment_dir, epoch))
                best_wd, best_epoch = wd, epoch
        
        # Necessary call to make windows reponsive.
        if show_progress:
            cv2.waitKey(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs='?', help="Path to JSON file containing dataset.")
    parser.add_argument("--config", help="Path to config file.")
    parser.add_argument("--results_dir", help="Directory where to store experiment.")
    parser.add_argument("--validation_period", help="Validate every N epochs.", type=int, default=VALIDATION_PERIOD)
    parser.add_argument("--silent", help="If specified, no output will be shown during the training.", action="store_true")
    args = parser.parse_args()

    # Get path to dataset. If no dataset path was provided through CLI args,
    # download the dataset and use that one.
    dataset_path = args.dataset
    if dataset_path is None:
        if not os.path.exists(DEFAULT_DATASET_PATH):
            print("No dataset was provided through arguments, downloading...")
            from download import download
            download(DEFAULT_DATASET_PATH)
            print("Download succesful, stored the dataset at {}.".format(DEFAULT_DATASET_PATH))
        dataset_path = DEFAULT_DATASET_PATH

    # If config file was specified, load it, if not, use empty dictionary
    # as config, which will result in default values being used.
    config = {}
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.load(f)

    # Load training parameters from the config.
    # If some parameter is not specified in the config, use default value.
    batch_size = config.get("batch_size", BATCH_SIZE)
    latent_size = config.get("latent_size", LATENT_SIZE)
    shuffle_colors = config.get("shuffle_colors", SHUFFLE_COLORS)
    loss_type = config.get("loss_type", LOSS)
    epoch_count = config.get("epoch_count", EPOCH_COUNT)
    learning_rate = config.get("learning_rate", LEARNING_RATE)
    layer_sizes = config.get("layer_sizes", LAYER_SIZES)

    # Set up experiment directory.
    experiment_dir = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if args.results_dir:
        utils.mkdir_p(args.results_dir)
        experiment_dir = os.path.join(args.results_dir, experiment_dir)
    os.mkdir(experiment_dir)
    utils.save_config(experiment_dir, batch_size, latent_size, layer_sizes, shuffle_colors,
                      loss_type, epoch_count, learning_rate)

    train(dataset_path, experiment_dir, batch_size, latent_size, layer_sizes, shuffle_colors,
          loss_type, epoch_count, learning_rate, args.validation_period, not args.silent)
