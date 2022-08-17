import os
import tensorflow as tf
import numpy as np
from datetime import datetime
from model import CPSYolo
from data import load_dataset
from callbacks import CheckPoint


if __name__ == '__main__':
    root_dir = 'D:/Public/qtkim/CPS/'
    print('===============================training===============================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 3
    epochs = 3000
    batch_size = 8
    learning_rate = 0.001
    data_dir = root_dir+'data/train'
    model_save_dir = root_dir +'model/model(' + start_time + ')'
    check_point_save_dir = root_dir + 'checkpoints/' + start_time
    os.mkdir(check_point_save_dir)

    ##load dataset
    print('-----------------------------load dataset-----------------------------')
    train_dataset, valid_dataset, steps_per_epoch, validation_steps = load_dataset(data_dir, valid_ratio=0.1,
                                                                                   batch_size=batch_size)
    ##build model
    #model
    print('-----------------------------build model------------------------------')
    model = CPSYolo(640, 3, 32, [80, 80, 80, 80, 80, 80], 0.5, loss_score_thresh=0.5)
    metric = 'accuracy'
    monitor_metric_name = 'val_total_1'
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[metric],
                  batch=batch_size, warmup_epoch=5)
    #callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric_name,
                                                      patience=200, mode='auto')
    reduce_rl = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric_name, factor=0.9, patience=30,
                                                     cooldown=10, min_lr=0.00001, mode='auto', verbose=True)
    checkpoint = CheckPoint(checkpoint_dir=check_point_save_dir)
    callbacks_list = [reduce_rl, checkpoint]

    ##train
    print('--------------------------------train---------------------------------')
    history = model.fit(train_dataset,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=valid_dataset,
                        validation_steps=validation_steps,
                        callbacks=callbacks_list,
                        class_weight=None,
                        initial_epoch=0)

    ##save model
    print('-----------------------------save model-------------------------------')
    model.model1.save(model_save_dir)