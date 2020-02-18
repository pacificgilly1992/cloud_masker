import errno
import logging
import os

import gdal
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class CloudMasker(object):
    gdal_driver = gdal.GetDriverByName("GTiff")
    compress_sup = ['LZW', 'DEFLATE', 'ZSTD']

    def __init__(self, mask: np.ndarray) -> None:
        """
        :param mask : array_like
            Any 2-dimensional array like object with integer or floats.
        """
        # Ensure mask is a 2D array.
        self._mask = np.atleast_2d(mask)

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @staticmethod
    def ensure_dir(fpath: str, isdir: bool = True) -> str:
        """
        Ensure that a named directory exists; if it does not, attempt to
        create it.

        :param fpath: str
            The file path of the directory you want to check exists. You
            can also specify both directory and filename.
        :param isdir: bool, optional
            Specify whether fpath you gave was a directory path or file
            path
        :return: str
            The same fpath supplied to this method. Useful for nesting
            these simple calls on a single line.
        """
        fpath = os.path.abspath(fpath)
        logger.debug("Creating directory: {dir}".format(
            dir=os.path.dirname(fpath)))
        try:
            os.makedirs(fpath) if isdir else os.makedirs(os.path.dirname(fpath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return fpath

    @staticmethod
    def _bool2int(mask: np.ndarray, invert: bool = False) -> tuple:
        """
        Get the index position for each True element in a 2D array.
         
        :param mask: np.ndarray
            The 2D dimensional array with type bool. you want to get the 
            True indices for.
        :param invert: bool, optional
            Specify whether you want to get the indices of the False 
            value instead.
        
        :rtype: 2x np.ndarray
        :return: The x and y positional indices for all the True values,
            returned as a tuple:(x, y) with x and y having the same len.
        """
        # Invert the boolean array if invert is True.
        if invert:
            mask = ~mask

        # Create an index array in both dimensions (i.e. [1,2,3,4,5 ... n]
        # and [[1],[2],[3],[4],[5], ... [n]])
        mask_y = np.arange(mask.size).reshape(mask.shape) % mask.shape[1]
        mask_x = np.arange(mask.size).reshape(mask.shape[::-1]).T \
                 % mask.shape[0]

        # Filter the x/y integer arrays by the bool mask to get out the x
        # and y indices for each True position.
        return mask_x[mask], mask_y[mask]

    def pad_mask(self, padding: int = 1, masked_value: int = 1,
                 pad_mask_value: int = None, merge_masks: bool = False,
                 overwrite: bool = True, dtype: type = np.uint8) -> np.ndarray:
        """
        Pad any integer mask by an integer amount

        :param padding: int or array_like, optional
            Specify the amount of padding (# of elements) you want
            around each identified mask. Default is 1 element padding.
        :param masked_value: int or float, optional
            Specify the value used for the mask. Default is 1.
        :param pad_mask_value: int or float, optional
            Specify the value of the padded mask around the original
            mask. Default is 1 greater than masked_value.
        :param merge_masks: bool, optional
            Specify whether to merge original and padded mask to
            new_mask_value. Default is False.
        :param overwrite: bool, optional
            Specify whether you want to overwrite any other 
        :param dtype: type, optional
            The numpy data type you want to the returned mask to have.
            See https://numpy.org/doc/1.18/user/basics.types.html

        :rtype np.ndarray
            The padded mask as a numpy array.
        """

        # Create a duplicate mask to imprint the pad onto.
        mask_pad = np.zeros_like(self._mask).astype(dtype)

        # Ensure both masked_value are list
        masked_value = np.atleast_1d(masked_value)

        # Define the new padding mask value
        pad_mask_value = max(
            masked_value) + 1 if pad_mask_value is None else pad_mask_value

        logger.info(
            f"Padding masked values of {masked_value} by {padding} element "
            f"with a new value {pad_mask_value}")

        # Loop through each dimension to determine any masked value
        if np.any([self._mask == val for val in masked_value]):
            # Create a boolean array. True when masked value is found,
            # False otherwise.
            bool_mask = np.isin(self._mask, masked_value, assume_unique=True)

            # Get the indices of the each True value.
            coords_x, coords_y = self._bool2int(bool_mask)

            # Numpy element-wise the coords
            coords_x0 = coords_x - padding
            coords_x1 = coords_x + padding + 1
            coords_y0 = coords_y - padding
            coords_y1 = coords_y + padding + 1
            coords_x0[coords_x0 < 0] = 0
            coords_y0[coords_y0 < 0] = 0

            # Loop through each x,y coordinates and pad around the central
            # value.
            for x0, x1, y0, y1 in zip(coords_x0, coords_x1, coords_y0,
                                      coords_y1):
                mask_pad[x0:x1, y0:y1] = pad_mask_value

            # Ensure the original masked value shown
            if not merge_masks:
                for val in masked_value:
                    mask_pad[self._mask == val] = val

            return mask_pad

        else:
            logger.warning(f"Couldn't find any elements matching the expected "
                           f"masked value: {masked_value}. Returning the "
                           f"original mask!")
            return self._mask

    def output_mask(self, fpath: str, extra_bands: list = None,
                    compress: str = 'LZW') -> None:
        """
        Outputs the mask in a GeoTIFF format with LZW compression.

        :param fpath: str
            The file location you want to save the mask too.
        :param extra_bands: list, optional
            Specify any extra bands you want to save along with the
            original mask as a list. Default is None.
        :param compress: str, optional
            Specify the compression method you want to save the output
            mask with. Available options are 'LZW', 'DEFLATE' and
            'ZSTD'. Default is 'LZW'.

        :return: None
        """
        compress = compress.upper()
        if compress not in self.compress_sup:
            raise ValueError(f"Compression method not supported. Supported "
                             f"methods are {' | '.join(self.compress_sup)}")

        # Build all the bands together.
        bands = [self._mask]
        bands += extra_bands if extra_bands is not None else []

        # Create the TIFF file
        ds = self.gdal_driver.Create(self.ensure_dir(fpath, isdir=False),
                                     *self._mask.shape[::-1],
                                     len(bands),
                                     gdal.GDT_Byte,
                                     options=(f'COMPRESS={compress}',))

        if ds.RasterCount != len(bands):
            raise IOError("Could not create the required number of bands to "
                          "make your GeoTIFF.")

        if ds is None:
            raise IOError("Could not create GeoTIFF. Please check the logs.")

        # Write each band the raster
        for i, band in enumerate(bands, 1):
            ds.GetRasterBand(i).WriteArray(band * 126)

        # Output the raster to file
        ds.FlushCache()
        ds = None
