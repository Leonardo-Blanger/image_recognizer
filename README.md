# image_recognizer
An image recognition library using TensorFlow and Keras.

Inside `tensorflow_scripts`, there is a set of scripts for formatting and compressing the data, and training and testing a CNN for face recognition implemented using TensorFlow.
The dataset used comes from the Extended Yale Faces Database B, publicly available at: http://vision.ucsd.edu/content/extended-yale-face-database-b-b

The `face_recognizer.py` file allows the addition of image classes, and an artificial generation of data, using image translations and rotations. It also provides an easy way to build a neural network using the `Keras.layers` classes, as well as training and validating it using the images previously added.
