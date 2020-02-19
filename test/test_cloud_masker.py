import logging
import subprocess
import time
import unittest

import gilly_utilities as gu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..assimila import CloudMasker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class TestCloudMaskerUnit(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        cls.cloud_masker = CloudMasker(mask)

    def test_something(self):
        self.assertEqual(True, False)


class TestCloudMaskerStress(object):
    pass


def compute_mandelbrot(n_max, some_threshold, nx, ny):
    # A grid of c-values
    x = np.linspace(-2, 1, nx)
    y = np.linspace(-1.5, 1.5, ny)

    c = x[:, None] + 1j * y[None, :]

    # Mandelbrot iteration

    z = c
    for j in range(n_max):
        z = z ** 2 + c

    mandelbrot_set = (abs(z) < some_threshold)

    return mandelbrot_set


def run_command(cmd):
    """
    Runs a command on the command line. Similar to 
    treadmill.easy.run_command,
    but this method has not restrictions on what can be execute. Use 
    with caution!

    :param cmd: command to run as an ordered list
    :return: list of output lines
    """
    logger.debug("Running [{cmd}]".format(cmd=cmd))
    return [line.decode(encoding="utf-8") for line in
            subprocess.check_output(cmd, shell=True).splitlines()]


def stress_test(num_sizes=None, num_clouds=None, num_iters=None,
                save_data=None, load_data=False, save_plot=None):
    if load_data is False:
        # Lets stress test this sucker
        np.random.seed(0)
        num_sizes = 5 if num_sizes is None else num_sizes
        num_clouds = 5 if num_clouds is None else num_clouds
        num_iters = 200 if num_iters is None else num_iters
        total = num_sizes * num_clouds * num_iters

        box_size = np.geomspace(0.5 * 10 ** 4, 1, num=num_sizes, dtype=int)
        box_entropy = np.geomspace(10 ** -5, 100, num=num_clouds,
                                   dtype=np.float64)
        results = np.zeros((num_sizes, num_clouds, num_iters),
                           dtype=np.float64)
        with gu.progress("Box Size: %s, Box Entropy: %s,"
                         "Iteration: %s. Time: %s",
                         total) as progress:
            for i, size in enumerate(box_size):
                for j, mask_num in enumerate(box_entropy):
                    for k in range(num_iters):
                        # Work out the number of possible values you want to
                        # randomise in your grid. The idea here is that the
                        # mask number will dictate the percentage of the grid
                        # that is a cloud i.e. 1 to 100%.
                        entropy = int(
                            max(1, size ** 2 - (size ** 2 * (mask_num / 100))))

                        # Create the grid with a "percentage" being cloud"
                        mask = np.random.randint(entropy, size=(
                            int(size), int(size))).astype(np.uint8)

                        # Initialise the cloud masker class
                        cloud_masker = CloudMasker(mask)

                        # Pass the mask by 1 value.
                        t1 = time.time()
                        mask_padded = cloud_masker.pad_mask(masked_value=0,
                                                            padding=1,
                                                            merge_masks=False)
                        t2 = time.time()
                        results[i, j, k] = t2 - t1

                        # Update the progress bar.
                        progress.update(size, mask_num, k, t2 - t1)

        # Save for saves keepings
        np.savez_compressed(save_data,
                            box_size=box_size,
                            box_entropy=box_entropy,
                            box_results=results)

    # Load the data
    else:
        data = np.load(load_data)
        box_size = data['box_size']
        box_entropy = data['box_entropy']
        results = data['box_results']

    # Calculate the mean of all the iterations for each box size and entropy.
    results = np.nanmedian(results, axis=2)

    # Work out the min and max values for plotting
    vmin = max(10 ** -5, np.nanmin(results))
    vmax = gu.truncate(np.nanmax(results), floor=False)

    print(results)
    print(f"Results size: {results.shape}")
    print(f"Min: {vmin}. Max: {vmax}")

    # Plot the data as a color mesh
    plt.pcolormesh(box_size**2, box_entropy, np.array(results).T,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap=plt.get_cmap('tab20b'))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Size of the grid box (dimensionless)")
    plt.ylabel("Cloud percentage (%)")
    plt.ylim(10**-5, 100)

    # Fix the aspect ratio
    ax = plt.gca()
    f = plt.gcf()
    gu.fixed_aspect_ratio(ax=ax, ratio=1, adjustable=None, xscale='log',
                          yscale='log')

    cbar = plt.colorbar()
    cbar.set_label('Time taken (s)')

    # Set size of the figure
    f.set_size_inches(5, 4)

    plt.tight_layout()

    if save_plot is None:
        plt.show()
    else:
        plt.savefig(save_plot, bbox_inches='tight', pad_inches=0.1, dpi=300)


def mandlebrot_test(fapth):
    mask = compute_mandelbrot(100, 50., 601, 401).astype(np.uint8)

    # Initialise the CloudMasker program
    cloud_masker = CloudMasker(mask)

    # Make Mandelbrot
    with gu.progress("", 150) as progress:
        for pad in range(150):
            mask_padded = cloud_masker.pad_mask(masked_value=1, padding=pad,
                                                merge_masks=False)
            cloud_masker._mask = mask_padded
            cloud_masker.output_mask("{fpath}mandlebrot_{pad}.tiff".format(
                fpath=fapth, pad=str(pad).rjust(3, "0")))
            progress.update()

    # Make video of the mandlebrot
    ffmpeg_cmd = 'ffmpeg -r 25 -start_number 000 -i mandlebrot_%03d.tiff '\
                 '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -preset slow ' \
                 '-profile:v high -level:v 4.0 -pix_fmt yuv420p -crf 22 ' \
                 '-codec:a aac -b:v 50000k -minrate 50000k -maxrate 50000k ' \
                 '-an mandlebrot_timelapse.mov'
    logger.info(f"Creating mandlebrot video using FFMPEG cmd: {ffmpeg_cmd}")
    run_command(ffmpeg_cmd)


def standard_test(fpath):
    """Creates the simple grid and outputs it."""

    # Lets do a simples
    mask = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0]], dtype=np.uint8)

    # Initialise the CloudMasker program
    cloud_masker = CloudMasker(mask)

    # Save the original cloud mask
    cloud_masker.output_mask(fpath + "_original.tiff")

    # Pad the cloud by 1 pixel.
    mask_padded = cloud_masker.pad_mask(masked_value=1, padding=1,
                                        merge_masks=False)

    # Save the whole mask
    cloud_masker._mask = mask_padded
    cloud_masker.output_mask(fpath + "_padded.tiff")


if __name__ == '__main__':
    # Perform the unit tests of the cloud masker program.
    # unittest.main()

    # Perform the standard test on the cloud masker program.
    fpath_plot = "Interview\\images\\standard_test"
    # standard_test(fpath_plot)

    # Perform the mandlebrot test on the cloud masker program.
    fpath_plot = "Interview\\images\\mandlebrot\\"
    # mandlebrot_test(fpath_plot)

    # Perform the stress test on the cloud masker program.
    fpath_data = "Interview\\data\\stress_test_backup_005.npz"
    fpath_plot = "Interview\\images\\stress_test_005.png"
    stress_test(num_sizes=50, num_clouds=50, num_iters=20,
                save_data=fpath_data,
                save_plot=fpath_plot)
