"""Utilities for real-time data augmentation on image data.
"""
import io
import os
import warnings
from pathlib import Path

import numpy as np

try:
    import librosa
except ImportError:
    librosa = None
try:
    import soundfile as sf
except ImportError:
    sf = None

def validate_filename(filename, white_list_formats):
    """Check if a filename refers to a valid file.

    # Arguments
        filename: String, absolute path to a file
        white_list_formats: Set, allowed file extensions

    # Returns
        A boolean value indicating if the filename is valid or not
    """
    return (filename.lower().endswith(white_list_formats) and os.path.isfile(filename))


def save_audio(path,
             x,
             sr=16000,
             file_format=None,
             scale=True,
             **kwargs):
    """Saves an image stored as a Numpy array to a path or file object.

    # Arguments
        path: Path or file object.
        x: Numpy array.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        file_format: Optional file format override. If omitted, the
            format to use is determined from the filename extension.
            If a file object was used instead of a filename, this
            parameter should always be used.
        scale: Whether to rescale image values to be within `[0, 255]`.
        **kwargs: Additional keyword arguments passed to `PIL.Image.save()`.
    """
    sf.write(path, y, sr, format=file_format)


def load_audio(path, sr=16000, target_size=30000,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path (string), pathlib.Path object, or io.BytesIO stream to image file.
        grayscale: DEPRECATED use `color_mode="grayscale"`.
        color_mode: The desired image format. One of "grayscale", "rgb", "rgba".
            "grayscale" supports 8-bit images and 32-bit signed integer images.
            Default: "rgb".
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported.
            Default: "nearest".
        keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
        TypeError: type of 'path' should be path-like or io.Byteio.
    """
    if librosa is None:
        raise ImportError('Could not import librosa.')

    audio, sr = librosa.load(path, sr=sr)
    #if isinstance(path, io.BytesIO):
    #    audio, sr = librosa.load(path, sr=sr)
    #elif isinstance(path, (Path, bytes, str)):
    #    if isinstance(path, Path):
    #        path = str(path.resolve())
    #    with open(path, 'rb') as f:
    #        audio = librosa.load(io.BytesIO(f.read()))
    #else:
    #    raise TypeError('path should be path-like or io.BytesIO'
    #                   ', not {}'.format(type(path)))
    if target_size is not None:
        if len(audio) != target_size:
            length = len(audio)
            # Set back to input height / width
            # if crop_height / crop_width is not smaller.
            crop_length = min(length, target_size)
            if target_size < length:
                audio = audio[:target_size]
            else:
                audio = np.concatenate([audio, np.zeros(target_size-length)], axis=0)
    return audio


def list_audio(directory, ext=('wav', 'mp3', 'flac')):
    """Lists all pictures in a directory, including all subdirectories.

    # Arguments
        directory: string, absolute path to the directory
        ext: tuple of strings or single string, extensions of the pictures

    # Returns
        a list of paths
    """
    ext = tuple('.%s' % e for e in ((ext,) if isinstance(ext, str) else ext))
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if f.lower().endswith(ext)]


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.

    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean, follow symbolic links to subdirectories.

    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            if fname.lower().endswith(white_list_formats):
                yield root, fname


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.

    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean, follow symbolic links to subdirectories.

    # Returns
         classes: a list of class indices
         filenames: the path of valid files in `directory`, relative from
             `directory`'s parent (e.g., if `directory` is "dataset/class1",
            the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        all_files = list(_iter_valid_files(directory, white_list_formats,
                                           follow_links))
        num_files = len(all_files)
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = all_files[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames


def array_to_audio(x, scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format, either "channels_first" or "channels_last".
            Default: "channels_last".
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively.
            Default: True.
        dtype: Dtype to use.
            Default: "float32".

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if librosa is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 1:
        raise ValueError('Expected audio array to have rank 1. '
                         'Got array with shape: %s' % (x.shape,))

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
        return x


def audio_to_array(audio, dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    #x = np.asarray(audio, dtype=dtype)
    return audio
