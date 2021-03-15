from setuptools import find_packages, setup

long_description = '''
Keras audio data generator is the data preprocessing
and data augmentation module of the Keras deep learning library.
It provides utilities for working with audio data
which keras official library doesn't provide.

Keras audio data generator is distributed under the MIT license.
'''

setup(name='keras_audio_generator',
      version='1.0.0',
      description='Keras audio data generator'
                  'for deep learning models',
      long_description=long_description,
      author='Ozora Ogino',
      url='https://github.com/ozora-ogino/keras-audio-generator',
      download_url='https://github.com/ozora-ogino/'
                   'keras-audio-generator/tarball/1.1.2',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'keras',
                        'soundfile',
                        'librosa',
                        ],
      extras_require={
          'tests': [ 'tensorflow'],
          'pep8': ['flake8'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
