from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from config import TrainConfig

#Model with 5 CNNs and 4 FCLs.
def create_nvidia_model1():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())

	# Added a Dropout layer to help reduce overfitting. 
    model.add(Dropout(0.5))

    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

#Model with 6 CNNs and 4 FCLs
def create_nvidia_model2():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

#Model with 6 CNNs and 4 larger FCLs. 
def create_nvidia_model3():
    model = Sequential()

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", input_shape=(row, col, ch)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    print('Model is created and compiled..')
    return model

def my_train_generator():
    num_iters = X_train.shape[0] / batch_size
    while 1:
        #print "Shuffling data..."
        #train_idx_shf = np.random.permutation(X_train.shape[0])
        #X_train = X_train[train_idx_shf]
        #y_train = y_train[train_idx_shf]
        for i in range(num_iters):
            #idx = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
            idx = train_idx_shf[i*batch_size:(i+1)*batch_size]
            tmp = X_train[idx].astype('float32')
            tmp = tmp - X_train_mean
            tmp = tmp / 255.0
            yield tmp, y_train[idx]

def my_test_generator():
    num_iters = X_test.shape[0] / batch_size
    while 1:
        for i in range(num_iters):
            tmp = X_test[i*batch_size:(i+1)*batch_size].astype('float32')
            tmp = tmp - X_train_mean
            tmp = tmp / 255.0
            yield tmp, y_test[i*batch_size:(i+1)*batch_size]

            
if __name__ == "__main__":
    config = TrainConfig()

    ch = config.num_channels
    row = config.img_height
    col = config.img_width
    num_epoch = config.num_epoch
    batch_size = config.batch_size
    data_path = config.data_path
    
    print "Loading training data..."
    print "Data path: " + data_path + "X_train_" + config.data_name + ".npy"

    X_train1 = np.load(data_path + "X_train_" + config.data_name + "_part1.npy")
    y_train1 = np.load(data_path + "y_train_" + config.data_name + "_part1.npy")
    X_train2 = np.load(data_path + "X_train_" + config.data_name + "_part2.npy")
    y_train2 = np.load(data_path + "y_train_" + config.data_name + "_part2.npy")
    X_train3 = np.load(data_path + "X_train_" + config.data_name + "_part3.npy")
    y_train3 = np.load(data_path + "y_train_" + config.data_name + "_part3.npy")
    X_train4 = np.load(data_path + "X_train_" + config.data_name + "_part4.npy")
    y_train4 = np.load(data_path + "y_train_" + config.data_name + "_part4.npy")
    X_train5 = np.load(data_path + "X_train_" + config.data_name + "_part5.npy")
    y_train5 = np.load(data_path + "y_train_" + config.data_name + "_part5.npy")
    
    # use part4 as validation set
    if config.val_part == 4:
        X_train = np.concatenate((X_train1, X_train2, X_train3, X_train5), axis=0)
        y_train = np.concatenate((y_train1, y_train2, y_train3, y_train5), axis=0)
        X_test = X_train4
        y_test = y_train4
    # use part3 as validation set        
    elif config.val_part == 3:
        X_train = np.concatenate((X_train1, X_train2, X_train4, X_train5), axis=0)
        y_train = np.concatenate((y_train1, y_train2, y_train4, y_train5), axis=0)
        X_test = X_train3
        y_test = y_train3
        
    print "X_train shape:" + str(X_train.shape)
    print "X_test shape:" + str(X_test.shape)
    print "y_train shape:" + str(y_train.shape)
    print "y_test shape:" + str(y_test.shape)
    
    np.random.seed(1235)
    train_idx_shf = np.random.permutation(X_train.shape[0])
    
    print "Computing training set mean..."
    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    
    print "Saving training set mean..."
    np.save(config.X_train_mean_path, X_train_mean)
    
    print "Creating model..."
    if config.model_name == "nvidia1":
        model = create_nvidia_model1()
    elif config.model_name == "nvidia2":
        model = create_nvidia_model2()
    elif config.model_name == "nvidia3":
        model = create_nvidia_model3()
        
    print model.summary()

    # checkpoint
    filepath = data_path + "models/weights_" + config.data_name + "_" + config.model_name + "-{epoch:02d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]
    
    iters_train = X_train.shape[0]
    iters_train = iters_train - iters_train % batch_size
    iters_test = X_test.shape[0]
    iters_test = iters_test - iters_test % batch_size
    
    model.fit_generator(my_train_generator(),
        nb_epoch=num_epoch,
        samples_per_epoch=iters_train,
        validation_data=my_test_generator(),
        nb_val_samples=iters_test,
        callbacks=callbacks_list,
        nb_worker=1
    )
