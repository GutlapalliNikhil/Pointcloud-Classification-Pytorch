import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def augment(points, label):
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float32)
    points = tf.random.shuffle(points)
    return points, label

def conv_bn(x, filters):
  x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
  x = layers.BatchNormalization(momentum=0.0)(x)
  return layers.Activation("relu")(x)

def dense_bn(x, filters):
  x = layers.Dense(filters)(x)
  x = layers.BatchNormalization(momentum=0.0)(x)
  return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
  def __init__(self, num_features, reg=0.001):
    self.num_features = num_features
    self.reg = reg
    self.eye = tf.eye(num_features)

  def __call__(self, x):
    x = tf.reshape(x, (-1, self.num_features, self.num_features))
    xxt = tf.tensordot(x, x, axes=(2, 2))
    xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
    return tf.reduce_sum(self.reg * tf.square(xxt - self.eye))

def tnet(input, num_features):

  # Initalise bias as the identity matrix
  bias = keras.initializers.Constant(np.eye(num_features).flatten())
  # reg = OrthogonalRegularizer(num_features)

  x = conv_bn(inputs, 32)
  x = conv_bn(x, 64)
  x = conv_bn(x, 512)
  x = layers.GlobalMaxPooling1D()(x)
  x = dense_bn(x, 256)
  x = dense_bn(x, 128)
  x = layers.Dense(
      num_features * num_features,
      kernel_initializer="zeros",
      bias_initializer=bias,

  )(x)
  feat_T = layers.Reshape((num_features, num_features))(x)
  # Apply affine transformationto input features
  return layers.Dot(axes=(2, 1))([input, feat_T])


def create_model(NUM_POINTS, NUM_CLASSES):

    inputs = keras.Input(shape=(NUM_POINTS, 3))
    
    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = keras.Model(inputs = inputs, outputs = outputs, name="pointnet")

    return model

def train_model(model, train_dataset, test_dataset, epochs):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["sparse_categorical_accuracy"],
    )

    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    loaded_model = tf.keras.models.load_model(filepath)
    return loaded_model

