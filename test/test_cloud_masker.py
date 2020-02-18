import logging
import unittest
import gilly_utilities as gu
import numpy as np
import time

import matplotlib.pyplot as plt
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


def stress_test(num_sizes=None, num_clouds=None, num_iters=None,
                save_data=None, load_data=False, save_plot=None):
    if load_data is False:
        # Lets stress test this sucker
        np.random.seed(0)
        num_sizes = 5 if num_sizes is None else num_sizes
        num_clouds = 5 if num_clouds is None else num_clouds
        num_iters = 200 if num_iters is None else num_iters
        total = num_sizes * num_clouds * num_iters

        box_size = np.geomspace(1, 10 ** 2, num=num_sizes, dtype=int)
        box_entropy = np.geomspace(1, 100, num=num_clouds, dtype=int)
        results = np.zeros((num_sizes, num_clouds, num_iters),
                           dtype=np.float64)
        with gu.progress("Entropy: %s", total) as progress:
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
                        mask_padded = cloud_masker.pad_mask(masked_value=1,
                                                            padding=1,
                                                            merge_masks=False)
                        results[i, j, k] = time.time() - t1

                        # Update the progress bar.
                        progress.update(entropy)

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
    results = np.nanmean(results, axis=2)

    # Work out the min and max values for plotting
    vmin = max(10**-6, np.nanmin(results))
    vmax = np.nanmax(results)

    print(f"Min: {vmin}. Max: {vmax}")

    # Plot the data as a color mesh
    plt.pcolormesh(box_size, box_entropy, results, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Size of the grid box (dimensionless)")
    plt.ylabel("Cloud percentage (%)")

    plt.colorbar()
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


def standard_test(fpath):
    """Creates the simple grid and outputs it."""

    # Lets do a simples
    mask = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)

    # Initialise the CloudMasker program
    cloud_masker = CloudMasker(mask)

    # Save the original cloud mask
    cloud_masker.output_mask(fpath + "_original.tiff")

    # Pad the cloud by 1 pixel.
    mask_padded = cloud_masker.pad_mask(masked_value=1, padding=1,
                                        merge_masks=False)

    # Save the whole mask
    cloud_masker.output_mask(fpath + "_padded.tiff", extra_bands=[mask_padded])


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
    fpath_data = "Interview\\data\\stress_test_backup.npz"
    fpath_plot = "Interview\\images\\stress_test.png"
    stress_test(num_sizes=500, num_clouds=500, num_iters=200,
                load_data=fpath_data,
                save_plot=fpath_plot)