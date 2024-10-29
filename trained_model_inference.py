# !wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
# !unzip ModelNet10.zip > /dev/null

!pip install trimesh
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from numpy import save, load

tf.random.set_seed(1234)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('GPUs Available: ', len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATA_DIR = os.path.join("/content/ModelNet10")

def parse_dataset(num_points = 2048):
  train_points = []
  train_labels = []
  test_points = []
  test_labels = []
  class_map = {}
  folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))
  print(folders)

  for i, folder in enumerate(folders):
    print("Processing folder: ", folder)

    # Store folder name with ID, so that we can retrieve later
    class_map[i] = folder.split("/")[-1]

    # gather all files
    train_files = glob.glob(os.path.join(folder, "train/*"))
    test_files = glob.glob(os.path.join(folder, "test/*"))

    for f in train_files:
      train_points.append(trimesh.load(f).sample(num_points))
      train_labels.append(i)


    for f in test_files:
      test_points.append(trimesh.load(f).sample(num_points))
      test_labels.append(i)
  return(
      np.array(train_points),
      np.array(test_points),
      np.array(train_labels),
      np.array(test_labels),
      class_map
  )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, class_map = parse_dataset(NUM_POINTS)

def augment(points, label):
  # Jitter Points
  points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
  # Shuffle points
  points = tf.random.shuffle(points)
  return points, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).map(augment).batch(1)

# Load the saved model (no need to redefine the model)
loaded_model = tf.keras.models.load_model('model_name.h5')

# Plotting example predictions on 8 samples
# (Code remains the same as before for plotting)

# Evaluate the entire test dataset
loss, accuracy = loaded_model.evaluate(test_dataset)

# Print the evaluation results
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

