import os
import tensorflow as tf


class CheckPoint(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        checkpoint_filename='model1_e{0:03d}-norm{1:.3f}-cps{2:.3f}-total{3:.3f}.hdf5'.\
            format(epoch+1, logs['normal_1'], logs['cps_1'], logs['total_1'])
        checkpoint_filepath = os.path.join(self.checkpoint_dir, checkpoint_filename)
        self.model.model1.save_weights(checkpoint_filepath)
        checkpoint_filename = 'model2_e{0:03d}-norm{1:.3f}-cps{2:.3f}-total{3:.3f}.hdf5'. \
            format(epoch + 1, logs['normal_2'], logs['cps_2'], logs['total_2'])
        checkpoint_filepath = os.path.join(self.checkpoint_dir, checkpoint_filename)
        self.model.model2.save_weights(checkpoint_filepath)