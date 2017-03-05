class DataConfig(object):
    data_path = "/home/icarus/Desktop/steering_model/data/"
    data_name = "hsv_gray_diff_ch4"
    img_height = 192
    img_width = 256
    num_channels = 4
    
class TrainConfig(DataConfig):
    model_name = "nvidia1"
    batch_size = 32
    num_epoch = 25
    val_part = 3
    X_train_mean_path = "data/X_train_"+ DataConfig.data_name + "_"+ model_name + "_mean.npy"
    
class TestConfig(TrainConfig):
    model_path = "submissions/final_model.hdf5"
    angle_train_mean = -0.004179079

class VisualizeConfig(object):
    pred_path = "submissions/final_predictions.csv"
    true_path = "submissions/CH2_final_evaluation.csv"
    img_path = DataConfig.data_path + "test/center/*.jpg"

class StatsConfig(object):
    data_path = "/home/icarus/Desktop/steering_model/data/train/steering.csv"
