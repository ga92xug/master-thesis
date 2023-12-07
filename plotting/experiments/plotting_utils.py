import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import wandb
from hydra import compose, initialize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def smooth_data(data, w: int = 3, keep_size: bool = False):
    smoothed = np.convolve(data, np.ones(w), 'valid') / w
    if keep_size:
        # Pad the smoothed array to match the original size
        pad_width = (w - 1) // 2
        smoothed = np.pad(smoothed, (pad_width, pad_width), mode='edge')
    return smoothed

def plot_init(individual_location:str, override: bool = False):
    """
    Initializes all plot functions. Gets the config and sets the font size.
    """
    overrides = []
    if override:
        overrides = [
            f"experiments={individual_location}", 
            # tell hydra to look for the config in the experiments folder
            "hydra.searchpath=[file://plotting/conf/experiments]"
        ]

    with initialize(config_path="../conf", version_base="1.2"):
        cfg = compose(config_name="plotting_config", overrides=overrides)

    # Set the font size for labels, tick labels, and titles
    plt.rcParams.update({'font.size': cfg.fontsize})

    return cfg, cfg.save_folder + individual_location


def get_fig_size(fig_size : Tuple, textwidth_in: float = 5.78853, reduction: float = 1.0):
    """
    This function calculates the figure size in inches based on the textwidth of the latex document.
    """
    
    fig_width = textwidth_in * reduction
    
    # Calculate the ratio of the figure width to the figure height
    ratio = fig_size[0] / fig_size[1]

    # Calculate the figure height in inches
    fig_height = fig_width / ratio

    fig_size = (fig_width, fig_height)
    return fig_size

def label2color(name: str):
    name2color_dict = {
        "Eq-NASNet": mcolors.CSS4_COLORS["red"],
        "ViT": mcolors.CSS4_COLORS["blue"],
        "ViT pre-trained": mcolors.CSS4_COLORS["navy"],
        "EfficientNet": mcolors.CSS4_COLORS["limegreen"],
        "EfficientNet pre-trained": mcolors.CSS4_COLORS["green"],
        "Eq-WRN-16-4": mcolors.CSS4_COLORS["peru"],
        "WRN-16-4": mcolors.CSS4_COLORS["saddlebrown"],
        "NAS-on": mcolors.CSS4_COLORS["purple"],
    }

    if name in name2color_dict:
        # direct match
        return name2color_dict[name]
    else:
        # try to find a match
        for key in name2color_dict:
            if key in name:
                return name2color_dict[key]
            
    # no match found
    # use hash to get a color
    names = list(mcolors.CSS4_COLORS)
    
    return mcolors.CSS4_COLORS[names[hash(name) % len(mcolors.CSS4_COLORS)]] # [hash(name) % len(mcolors.CSS4_COLORS)]


def save_plot(
        figure: plt.Figure, 
        name: str, 
        folder_name: str,
        file_format: str = "pdf",
    ):
    """
    Safe a matplotlib figure to a file.
    """
    name = name.replace(' ', '_').lower()

    if not os.path.exists(f'{folder_name}'):
        os.makedirs(f'{folder_name}')

    figure.savefig(f'{folder_name}/{name}.{file_format}', dpi=300, bbox_inches = "tight")


