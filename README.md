# Prediction Of Steering Angles For Self-Driving Cars Using Deep Learning

This is the code for training and running the Deep Learning Model to predict steering angles. For a detailed description of the work done, please refer to the PDF file Final_Report. 

## Requirements & Dependencies
This project was developed using a pre-built community AMI of the new AWS P2 instances (Go Deeper AMI), which have NVIDIA Tesla K80 GPUs and already come with much of the necessary tools installed. 
- Python 2.7
- [Numpy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Tensorflow](https://www.tensorflow.org/get_started/os_setup)
- [Keras](https://keras.io/) 1.1.0
- [Autopilot-TensorFlow](https://github.com/SullyChen/Autopilot-TensorFlow) is an open-source TensorFlow implementation of the NVIDIA paper, provided the main architecture for the model. 
- [udacity-driving-reader](https://github.com/rwightman/udacity-driving-reader) from rwightman, used to extract images from ROS bag files
- [GoDeeper](https://github.com/Miej/GoDeeper) is an AWS EC2 Community AMI with support for GPU acceleration, CUDA, CuDNN, TensorFlow, Keras, OpenCV


# Data
This dataset is based on driving data from San Mateo, CA to Half Moon Bay, CA (curvy and highway driving).

For Udacity Evaluation: The converted dataset can be [downloaded from here](https://we.tl/Rv5OjNjp5n) (link expires March 7th, 2017)

Otherwise, the dataset can be downloaded in the ROSBAG format from here: https://github.com/udacity/self-driving-car/tree/master/datasets/CH2 (both CH2_001 and CH2_002)

We used rwightman's [Udacity Reader docker tool](https://github.com/rwightman/udacity-driving-reader) 
to convert the images into JPGs.

Even though the imagery is from three different cameras (left, center, right), we only used the center images.

The code assumes the following directory structure for data:

```
- data
-- models
-- test
--- center
---- 1477431483393873340.jpg
-- train
--- center
---- 1477431802438024821.jpg 
```

Change `data_path` value in `config.py` to point to this data directory.

# Pre-processing

The raw images are of size 640 x 480. In our final model, we resized the images to 256 x 192, converted from RGB color format to grayscale, 
computed lag 1 differences between frames and used 2 consecutive differenced images.
For example, at time t we used [x_{t} - x_{t-1}, x_{t-1} - x_{t-2}] as input where x corresponds to the grayscale image. 
No future frames were used to predict the current steering angle.

To pre-process training data, run:

```
python preprocess_train_data.py
```

To pre-process test data, run:

```
python preprocess_test_data.py
```

These pre-processing scripts convert image sets to numpy arrays.

# Model

The main architecture for this model was inspired by the [NVIDIA's self-driving car paper](https://arxiv.org/abs/1604.07316)
The code includes 3 different models. To choose one of the models, change the model_name in config.py to either "nvidia1", "nvidia2", or "nvidia3".

To train different models, run:

```
python train.py
```

You can change these parameters in the `config.py` file:

* `--data` - alias for pre-processed data. There are multiple ways to pre-process the data (how many consecutive frames to use, image size, etc).
This parameter value gives us information what data set to use.
* `--num_channels` - number of channels the data has. For example, if you use 4 consecutive frames, then `num_channels` must be 4.
* `--img_height` - image height in pixels, default is 192.
* `--img_width` - image width in pixels, default is 256.
* `--model_name` - model definition file, see `models.py` for different models.
* `--val_part` - which part of the data to use as validation set.
* `--batch_size` - minibatch size, default is 32.
* `--num_epoch` - number epochs to train, default is 10.
* `--data_path` - folder path to pre-processed numpy arrays.


Once you have trained your models, you can choose the one with the best performance, copy it into the submissions folder and rename it to "final_model.hdf5". 

To predict steering angles from test data, run:

```
python predict.py
```

* Visualizing predicted steering angles

To visualize model predictions on test data, run:

```
python visualize.py
```

White circle shows the true angle, black circle shows the predicted angle.
You might need to change the variable `VisualizeConfig` in `config.py` to point to the location of phase 2 images.

These visualizations can help us understand the weaknesses of the model.
For example, human steering movements are smoother on straight road while the model zig-zags.

# Pointers and Acknowledgements

* Some of the model architectures are based on [NVIDIA's end-to-end self-driving car paper](https://arxiv.org/abs/1604.07316).
* rwightman's [docker tool](https://github.com/rwightman/udacity-driving-reader) was used to convert the round 2 data from ROSBAG to JPG.
* [Keras](https://github.com/fchollet/keras) was used to build neural network models.
