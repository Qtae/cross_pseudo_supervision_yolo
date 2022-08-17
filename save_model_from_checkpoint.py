import os
import tensorflow as tf
from datetime import datetime
from model import yolo_v4_more_tiny

if __name__ == '__main__':
    root_dir = 'D:/Public/qtkim/CPS/'
    print('==================save model==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 3
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = root_dir+'model/model(' + start_time + ')'
    checkpoint_path = root_dir+'checkpoints/2022_08_12-16_10_53/model2_e961-norm0.102-cps0.074-total0.213.hdf5'

    ##build model
    print('-----------------------build model------------------------')
    model = yolo_v4_more_tiny(input_layer, output_class=classes, input_size=640,
                                      stride=32, anchors=[80, 80, 80, 80, 80, 80])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          metrics=['accuracy'])
    model.load_weights(checkpoint_path)

    ##save
    model.save(model_save_dir)