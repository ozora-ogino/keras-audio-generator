# Audio Data Generator on keras (beta)

This is Keras Data Generator for audio data.

You can use this almost same way with Keras official ImageDataGenerator.

So you can use this easily once you install.

## Description

AudioDataGenerator uses raw audio as inputs and outputs 1D, 2D or 3D data for inputs of Neural Network.

If you want AudioDataGenerator to output 3d data such as spctrogram, all you need to do is pass the function which extracts audio features to  ```preprocessing_function```.


### Installation 

```
$ git clone https://github.com/ozora-ogino/keras-audio-generator
$ cd keras-audio-generator/
$ sudo pip3 install -r requirements.txt
```

## Contribution

Contributions are more than welcome!
Please fork it and pull request.
