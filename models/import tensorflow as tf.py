import tensorflow as tf
from tensorflow.keras.utils import plot_model

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Plot the model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Print the model summary
model.summary()
