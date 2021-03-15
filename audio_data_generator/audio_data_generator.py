"""Utilities for real-time data augmentation on image data.
"""
import warnings

import numpy as np

#from .dataframe_iterator import DataFrameIterator
from .directory_iterator import DirectoryIterator
from .numpy_array_iterator import NumpyArrayIterator


class AudioDataGenerator(object):
    """Generate batches of tensor image data with real-time data augmentation.
     The data will be looped over (in batches).

    # Arguments
        featurewise_center: Boolean.
            Set input mean to 0 over the dataset, feature-wise.
        samplewise_center: Boolean. Set each sample mean to 0.
        featurewise_std_normalization: Boolean.
            Divide inputs by std of the dataset, feature-wise.
        samplewise_std_normalization: Boolean. Divide each input by its std.
        zca_whitening: Boolean. Apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: Int. Degree range for random rotations.
        width_shift_range: Float, 1-D array-like or int
            - float: fraction of total width, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-width_shift_range, +width_shift_range)`
            - With `width_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `width_shift_range=[-1, 0, +1]`,
                while with `width_shift_range=1.0` possible values are floats
                in the interval `[-1.0, +1.0)`.
        height_shift_range: Float, 1-D array-like or int
            - float: fraction of total height, if < 1, or pixels if >= 1.
            - 1-D array-like: random elements from the array.
            - int: integer number of pixels from interval
                `(-height_shift_range, +height_shift_range)`
            - With `height_shift_range=2` possible values
                are integers `[-1, 0, +1]`,
                same as with `height_shift_range=[-1, 0, +1]`,
                while with `height_shift_range=1.0` possible values are floats
                in the interval `[-1.0, +1.0)`.
        brightness_range: Tuple or list of two floats. Range for picking
            a brightness shift value from.
        shear_range: Float. Shear Intensity
            (Shear angle in counter-clockwise direction in degrees)
        zoom_range: Float or [lower, upper]. Range for random zoom.
            If a float, `[lower, upper] = [1-zoom_range, 1+zoom_range]`.
        channel_shift_range: Float. Range for random channel shifts.
        fill_mode: One of {"constant", "nearest", "reflect" or "wrap"}.
            Default is 'nearest'.
            Points outside the boundaries of the input are filled
            according to the given mode:
            - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
            - 'nearest':  aaaaaaaa|abcd|dddddddd
            - 'reflect':  abcddcba|abcd|dcbaabcd
            - 'wrap':  abcdabcd|abcd|abcdabcd
        cval: Float or Int.
            Value used for points outside the boundaries
            when `fill_mode = "constant"`.
        horizontal_flip: Boolean. Randomly flip inputs horizontally.
        vertical_flip: Boolean. Randomly flip inputs vertically.
        rescale: rescaling factor. Defaults to None.
            If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided
            (after applying all other transformations).
        preprocessing_function: function that will be applied on each input.
            The function will run after the image is resized and augmented.
            The function should take one argument:
            one image (NumPy tensor with rank 3),
            and should output a NumPy tensor with the same shape.
        data_format: Image data format,
            either "channels_first" or "channels_last".
            "channels_last" mode means that the images should have shape
            `(samples, height, width, channels)`,
            "channels_first" mode means that the images should have shape
            `(samples, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        validation_split: Float. Fraction of images reserved for validation
            (strictly between 0 and 1).
        interpolation_order: int, order to use for
            the spline interpolation. Higher is slower.
        dtype: Dtype to use for the generated arrays.

    # Examples
    Example of using `.flow(x, y)`:

    ```python
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32, epochs=epochs)

    # here's a more "manual" example
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
    ```
    Example of using `.flow_from_directory(directory)`:

    ```python
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    ```

    Example of transforming images and masks together.

    ```python
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory(
        'data/images',
        class_mode=None,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'data/masks',
        class_mode=None,
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50)
    ```

    Example of using ```.flow_from_dataframe(dataframe, directory,
                                            x_col, y_col)```:

    ```python

    train_df = pandas.read_csv("./train.csv")
    valid_df = pandas.read_csv("./valid.csv")

    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory='data/train',
            x_col="filename",
            y_col="class",
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_dataframe(
            dataframe=valid_df,
            directory='data/validation',
            x_col="filename",
            y_col="class",
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
    ```
    """

    def __init__(self,
                 rescale=None,
                 preprocessing_function=None,
                 validation_split=0.0,
                 dtype='float32'):

        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype

        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)
        self._validation_split = validation_split

        self.mean = None
        self.std = None


    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='wav',
             ignore_class_split=False,
             subset=None):
        """Takes data & label arrays, generates batches of augmented data.

        # Arguments
            x: Input data. NumPy array of rank 4 or a tuple.
                If tuple, the first element
                should contain the images and the second element
                another NumPy array or a list of NumPy arrays
                that gets passed to the output
                without any modifications.
                Can be used to feed the model miscellaneous data
                along with the images.
                In case of grayscale data, the channels axis of the image array
                should have value 1, in case
                of RGB data, it should have value 3, and in case
                of RGBA data, it should have value 4.
            y: Labels.
            batch_size: Int (default: 32).
            shuffle: Boolean (default: True).
            sample_weight: Sample weights.
            seed: Int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            ignore_class_split: Boolean (default: False), ignore difference
                in number of classes in labels across train and validation
                split (useful for non-classification tasks)
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.

        # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array of image data
                (in the case of a single image input) or a list
                of NumPy arrays (in the case with
                additional inputs) and `y` is a NumPy array
                of corresponding labels. If 'sample_weight' is not None,
                the yielded tuples are of the form `(x, y, sample_weight)`.
                If `y` is None, only the NumPy array `x` is returned.
        """
        return NumpyArrayIterator(
            x,
            y,
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            ignore_class_split=ignore_class_split,
            subset=subset,
            dtype=self.dtype
        )

    def flow_from_directory(self,
                            directory,
                            target_size=30000,
                            sr=16000,
                            dim=1,
                            classes=None,
                            class_mode='categorical',
                            batch_size=32,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='wav',
                            follow_links=False,
                            subset=None,
                            ):
        """Takes the path to a directory & generates batches of augmented data.

        # Arguments
            directory: string, path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](
                https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
                Whether the images will be converted to
                have 1, 3, or 4 channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                - If None, no labels are returned
                  (the generator will only yield batches of image data,
                  which is useful to use with `model.predict_generator()`).
                  Please note that in case of class_mode None,
                  the data still needs to reside in a subdirectory
                  of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
                If set to False, sorts the data in alphanumeric order.
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
            keep_aspect_ratio: Boolean, whether to resize images to a target
                size without aspect ratio distortion. The image is cropped in
                the center with target aspect ratio before resizing.

        # Returns
            A `DirectoryIterator` yielding tuples of `(x, y)`
                where `x` is a NumPy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a NumPy array of corresponding labels.
        """
        return DirectoryIterator(
            directory,
            self,
            target_size=target_size,
            sr=sr,
            dim=dim,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            dtype=self.dtype
        )

    #def flow_from_dataframe(self,
    #                        dataframe,
    #                        directory=None,
    #                        x_col="filename",
    #                        y_col="class",
    #                        weight_col=None,
    #                        target_size=30000,
    #                        classes=None,
    #                        class_mode='categorical',
    #                        batch_size=32,
    #                        shuffle=True,
    #                        seed=None,
    #                        save_to_dir=None,
    #                        save_prefix='',
    #                        save_format='wav',
    #                        subset=None,
    #                        interpolation='nearest',
    #                        validate_filenames=True,
    #                        **kwargs):
    #    """Takes the dataframe and the path to a directory
    #     and generates batches of augmented/normalized data.

    #    **A simple tutorial can be found **[here](
    #                                http://bit.ly/keras_flow_from_dataframe).

    #    # Arguments
    #        dataframe: Pandas dataframe containing the filepaths relative to
    #            `directory` (or absolute paths if `directory` is None) of the
    #            images in a string column. It should include other column/s
    #            depending on the `class_mode`:
    #            - if `class_mode` is `"categorical"` (default value) it must
    #                include the `y_col` column with the class/es of each image.
    #                Values in column can be string/list/tuple if a single class
    #                or list/tuple if multiple classes.
    #            - if `class_mode` is `"binary"` or `"sparse"` it must include
    #                the given `y_col` column with class values as strings.
    #            - if `class_mode` is `"raw"` or `"multi_output"` it should contain
    #            the columns specified in `y_col`.
    #            - if `class_mode` is `"input"` or `None` no extra column is needed.
    #        directory: string, path to the directory to read images from. If `None`,
    #            data in `x_col` column should be absolute paths.
    #        x_col: string, column in `dataframe` that contains the filenames (or
    #            absolute paths if `directory` is `None`).
    #        y_col: string or list, column/s in `dataframe` that has the target data.
    #        weight_col: string, column in `dataframe` that contains the sample
    #            weights. Default: `None`.
    #        target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
    #            The dimensions to which all images found will be resized.
    #        color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb".
    #            Whether the images will be converted to have 1 or 3 color channels.
    #        classes: optional list of classes (e.g. `['dogs', 'cats']`).
    #            Default: None. If not provided, the list of classes will be
    #            automatically inferred from the `y_col`,
    #            which will map to the label indices, will be alphanumeric).
    #            The dictionary containing the mapping from class names to class
    #            indices can be obtained via the attribute `class_indices`.
    #        class_mode: one of "binary", "categorical", "input", "multi_output",
    #            "raw", sparse" or None. Default: "categorical".
    #            Mode for yielding the targets:
    #            - `"binary"`: 1D NumPy array of binary labels,
    #            - `"categorical"`: 2D NumPy array of one-hot encoded labels.
    #                Supports multi-label output.
    #            - `"input"`: images identical to input images (mainly used to
    #                work with autoencoders),
    #            - `"multi_output"`: list with the values of the different columns,
    #            - `"raw"`: NumPy array of values in `y_col` column(s),
    #            - `"sparse"`: 1D NumPy array of integer labels,
    #            - `None`, no targets are returned (the generator will only yield
    #                batches of image data, which is useful to use in
    #                `model.predict_generator()`).
    #        batch_size: size of the batches of data (default: 32).
    #        shuffle: whether to shuffle the data (default: True)
    #        seed: optional random seed for shuffling and transformations.
    #        save_to_dir: None or str (default: None).
    #            This allows you to optionally specify a directory
    #            to which to save the augmented pictures being generated
    #            (useful for visualizing what you are doing).
    #        save_prefix: str. Prefix to use for filenames of saved pictures
    #            (only relevant if `save_to_dir` is set).
    #        save_format: one of "png", "jpeg"
    #            (only relevant if `save_to_dir` is set). Default: "png".
    #        follow_links: whether to follow symlinks inside class subdirectories
    #            (default: False).
    #        subset: Subset of data (`"training"` or `"validation"`) if
    #            `validation_split` is set in `ImageDataGenerator`.
    #        interpolation: Interpolation method used to resample the image if the
    #            target size is different from that of the loaded image.
    #            Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
    #            If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
    #            supported. If PIL version 3.4.0 or newer is installed, `"box"` and
    #            `"hamming"` are also supported. By default, `"nearest"` is used.
    #        validate_filenames: Boolean, whether to validate image filenames in
    #            `x_col`. If `True`, invalid images will be ignored. Disabling this
    #            option can lead to speed-up in the execution of this function.
    #            Default: `True`.

    #    # Returns
    #        A `DataFrameIterator` yielding tuples of `(x, y)`
    #        where `x` is a NumPy array containing a batch
    #        of images with shape `(batch_size, *target_size, channels)`
    #        and `y` is a NumPy array of corresponding labels.
    #    """
    #    if 'has_ext' in kwargs:
    #        warnings.warn('has_ext is deprecated, filenames in the dataframe have '
    #                      'to match the exact filenames in disk.',
    #                      DeprecationWarning)
    #    if 'sort' in kwargs:
    #        warnings.warn('sort is deprecated, batches will be created in the'
    #                      'same order than the filenames provided if shuffle'
    #                      'is set to False.', DeprecationWarning)
    #    if class_mode == 'other':
    #        warnings.warn('`class_mode` "other" is deprecated, please use '
    #                      '`class_mode` "raw".', DeprecationWarning)
    #        class_mode = 'raw'
    #    if 'drop_duplicates' in kwargs:
    #        warnings.warn('drop_duplicates is deprecated, you can drop duplicates '
    #                      'by using the pandas.DataFrame.drop_duplicates method.',
    #                      DeprecationWarning)

    #    return DataFrameIterator(
    #        dataframe,
    #        directory,
    #        self,
    #        x_col=x_col,
    #        y_col=y_col,
    #        weight_col=weight_col,
    #        target_size=target_size,
    #        classes=classes,
    #        class_mode=class_mode,
    #        batch_size=batch_size,
    #        shuffle=shuffle,
    #        seed=seed,
    #        save_to_dir=save_to_dir,
    #        save_prefix=save_prefix,
    #        save_format=save_format,
    #        subset=subset,
    #        interpolation=interpolation,
    #        validate_filenames=validate_filenames,
    #        dtype=self.dtype
    #    )

    def standardize(self, x):
        """Applies the normalization configuration in-place to a batch of inputs.

        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network. If a copy of `x`
        would be created instead it would have a significant performance cost.
        If you want to apply this method without changing the input in-place
        you can call the method creating a copy before:

        standardize(np.copy(x))

        # Arguments
            x: Batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        return x

    def get_random_transform(self, audio_shape, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """

        if seed is not None:
            np.random.seed(seed)

        theta = 0
        tx = 0
        ty = 0
        channel_shift_intensity = None

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'channel_shift_intensity': channel_shift_intensity,
                                }

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        #params = self.get_random_transform(x.shape, seed)
        #return self.apply_transform(x, params)
        return x

    def fit(self, x,
            augment=False,
            rounds=1,
            seed=None):
        """Fits the data generator to some sample data.

        This computes the internal data stats related to the
        data-dependent transformations, based on an array of sample data.

        Only required if `featurewise_center` or
        `featurewise_std_normalization` or `zca_whitening` are set to True.

        When `rescale` is set to a value, rescaling is applied to
        sample data before computing the internal data stats.

        # Arguments
            x: Sample data. Should have rank 4.
             In case of grayscale data,
             the channels axis should have value 1, in case
             of RGB data, it should have value 3, and in case
             of RGBA data, it should have value 4.
            augment: Boolean (default: False).
                Whether to fit on randomly augmented samples.
            rounds: Int (default: 1).
                If using data augmentation (`augment=True`),
                this is how many augmentation passes over the data to use.
            seed: Int (default: None). Random seed.
       """
        x = np.asarray(x, dtype=self.dtype)
        if seed is not None:
            np.random.seed(seed)

        x = np.copy(x)
        if self.rescale:
            x *= self.rescale

        if augment:
            ax = np.zeros(
                tuple([rounds * x.shape[0]] + list(x.shape)[1:]),
                dtype=self.dtype)
            for r in range(rounds):
                for i in range(x.shape[0]):
                    ax[i + r * x.shape[0]] = self.random_transform(x[i])
            x = ax
