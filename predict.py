import numpy as np
from keras.models import load_model
import pandas as pd
import glob
from config import TestConfig

config = TestConfig()

data_path = config.data_path
data_name = config.data_name
val_part = config.val_part
model_path = config.model_path

test_path = "{}/X_test_{}.npy".format(data_path, data_name)
out_name = "submissions/final_predictions.csv"

print "Loading model..."
model = load_model(config.model_path)

print "Loading training data mean..."
X_train_mean = np.load(config.X_train_mean_path)

print "Reading test data..."
X_test = np.load(test_path)
X_test = X_test.astype('float32')
X_test -= X_train_mean
X_test /= 255.0

print "Predicting..."
preds = model.predict(X_test)
preds = preds[:, 0]

dummy_preds = np.repeat(config.angle_train_mean, config.num_channels)
preds = np.concatenate((dummy_preds, preds))

# join predictions with frame_ids
filenames = glob.glob("{}/test/center/*.jpg".format(data_path))
filenames = sorted(filenames)
frame_ids = [f.replace(".jpg", "").replace("{}/test/center/".format(data_path), "") for f in filenames]

print "Writing predictions..."
pd.DataFrame({"frame_id": frame_ids, "steering_angle": preds}).to_csv(out_name, index=False, header=True)

print "Done!"

# Calculate RMSE
# ------------------------------------------------------------------------------------
print "Calculating RMSE"
test = pd.read_csv("submissions/CH2_final_evaluation.csv")
predictions = pd.read_csv("submissions/final_predictions.csv")

t = test['steering_angle']
p = predictions['steering_angle']

length = predictions.shape[0]
print "Predicted angles: " + str(length)
sq = 0
mse = 0
for j in range(length):
    sqd = ((p[j] - t[j])**2)
    sq = sq + sqd
print(sq)
mse = sq/length
print(mse)
rmse = np.sqrt(mse)
print("model evaluated RMSE:", rmse)
