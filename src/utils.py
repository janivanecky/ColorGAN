import os
import yaml

def get_model_save_path(experiment_dir, epoch):
    return "{}/model_{}.pt".format(experiment_dir, epoch)

def get_color_save_path(experiment_dir, epoch):
    return "{}/colors/{}.png".format(experiment_dir, epoch)

def get_color_save_dir_path(experiment_dir):
    return "{}/colors".format(experiment_dir)

def get_config_save_path(experiment_dir):
    return "{}/config.yaml".format(experiment_dir)

def get_plot_save_path(experiment_dir):
    return "{}/wasserstein_distance.png".format(experiment_dir)

def get_log_save_path(experiment_dir):
    return "{}/log.txt".format(experiment_dir)

def mkdir_p(path):
    try:
        os.mkdir(path)
    except:
        pass

def save_config(experiment_dir, batch_size, latent_size, layer_sizes, shuffle_colors, loss_type, epoch_count, learning_rate):
    path = get_config_save_path(experiment_dir)
    with open(path, 'w') as f:
        config = {
            "batch_size": batch_size,
            "latent_size": latent_size,
            "layer_sizes": layer_sizes,
            "shuffle_colors": shuffle_colors,
            "loss_type": loss_type,
            "epoch_count": epoch_count,
            "learning_rate": learning_rate,
        }
        yaml.dump(config, f, default_flow_style=False)