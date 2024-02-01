import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import figure
from pathlib import Path, PurePath
from typing import Dict, Union, List

import numpy as np
from PIL import Image

def plot_dup_images(
    orig: str,
    image_list: List,
    scores: bool = False,
    outfile: str = None,
) -> None:
    """
    Plotting function for plot_duplicates() defined below.

    Args:
        image_dir: image directory where all files in duplicate_map are present.
        orig: filename for which duplicates are to be plotted.
        image_list: List of duplicate filenames, could also be with scores (filename, score).
        scores: Whether only filenames are present in the image_list or scores as well.
        outfile:  Name of the file to save the plot.
    """
    n_ims = len(image_list)
    ncols = 4  # fixed for a consistent layout
    nrows = int(np.ceil(n_ims / ncols)) + 1
    fig = figure.Figure(figsize=(10, 14))

    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    ax = plt.subplot(
        gs[0, 1:3]
    )  # Always plot the original image in the middle of top row
    ax.imshow(Image.open(orig))
    ax.set_title('Duplicated Found: %d, %s' % (len(image_list), orig.split("/")[-1]))
    ax.axis('off')

    for i in range(0, n_ims):
        row_num = (i // ncols) + 1
        col_num = i % ncols

        ax = plt.subplot(gs[row_num, col_num])
        if scores:
            ax.imshow(Image.open(image_list[i][0]))
            # val = _formatter(image_list[i][1])
            # title = ' '.join([image_list[i][0], f'({val})'])
            # title = paths_dict[image_list[i][0]]
        else:
            ax.imshow(Image.open(image_list[i]))
            # title = image_list[i]
            # title = paths_dict[image_list[i]]

        ax.set_title(image_list[i].split("/")[-1], fontsize=6)
        ax.axis('off')
    gs.tight_layout(fig)
    
    return plt.gcf()

    if outfile:
        plt.savefig(outfile, dpi=300)

    plt.show()
    plt.close()
    return 