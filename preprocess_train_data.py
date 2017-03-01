from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
from config import DataConfig
    
def make_hsv_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]
    
    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print "Processed " + str(i) + " images..."
        for j in range(num_channels):
            path0 = df['fullpath'].iloc[i - j - 1]
            path1 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, target_size=(row, col))
            img1 = load_img(data_path + path1, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)
	    #Calculate Lag 1 differences.
            img = img1[:, :, 2] - img0[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)
    
            X[i - num_channels, :, :, j] = img
    return X, np.array(df["angle"].iloc[num_channels:])
    
if __name__ == "__main__":
    config = DataConfig()
    data_path = config.data_path
    row, col = config.img_height, config.img_width

# Creates Numpy arrays from the image frames on the txt files.
# The txt files contain the timestamps for mostly curvy driving. 
# Straight driving was manually removed from the training dataset.
   
    print "Pre-processing train data..."
    for i in range(1, 6):
        X_train, y_train = make_hsv_grayscale_diff_data("data/train_part" + str(i) + ".txt", 4)
        np.save("{}/X_train_hsv_gray_diff_ch4_part{}".format(data_path, i), X_train)
        np.save("{}/y_train_hsv_gray_diff_ch4_part{}".format(data_path, i), y_train)
