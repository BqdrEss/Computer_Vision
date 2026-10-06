# Image Augmentation and CNN Classification

**Computer vision coursework**

This exercise explores image preprocessing and classification using Python, OpenCV, and TensorFlow/Keras. I worked with random rotations, background-color transformations, and a convolutional neural network with four output classes.

## Files

- [hada.ipynb](hada.ipynb): data loading, preprocessing experiments, CNN definition, training, and evaluation cells.
- [hada.py](hada.py): helper functions for rotation, background transformation, resizing, and an experimental edge-processing function.

## Status and setup

This is an unfinished coursework notebook. It expects local images under `images/training/` and `images/testing/`, with a subfolder for each class. The original dataset is not included.

Run Jupyter from this directory so that `import hada` and the relative data paths resolve. The notebook uses NumPy, Matplotlib, OpenCV, TensorFlow/Keras, Keras-Preprocessing, Pillow, and scikit-learn. Its original metadata records Python 3.8.10; package versions were not pinned.

The original code is retained. It has unresolved batch/image indexing and resize issues, a mismatch between loaded image dimensions and the model input shape, and an incorrect confusion-matrix call. End-to-end training has not been revalidated. Saved notebook outputs have been cleared.

## Source

Consolidated from my `hada` repository into this coursework collection. The original repository retains its commit history; the notebook cell source and Python helper are preserved here.

[Back to computer vision coursework](../README.md)
