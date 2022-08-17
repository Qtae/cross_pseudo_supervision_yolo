import tensorflow as tf
from loss import normal_yolo_loss, cps_yolo_loss
from iou import bbox_iou


class CPSYolo(tf.keras.Model):
    def __init__(self, input_size, output_class, stride, anchors, iou_loss_thresh, loss_score_thresh, cps_weight = 1.5):
        super(CPSYolo, self).__init__()
        input_layer = tf.keras.layers.Input([input_size, input_size, 1])
        self.model1 = yolo_v4_more_tiny(input_layer, output_class, input_size, stride, anchors)
        self.model2 = yolo_v4_more_tiny(input_layer, output_class, input_size, stride, anchors)
        self.output_class = output_class
        self.stride = stride
        self.iou_loss_thresh = iou_loss_thresh
        self.loss_score_thresh = loss_score_thresh
        self.cps_weight = cps_weight

    def compile(self, optimizer, metrics, batch, warmup_epoch):
        super(CPSYolo, self).compile(optimizer=optimizer, metrics=metrics)
        self.model1.compile(optimizer=optimizer, metrics=metrics)
        self.model2.compile(optimizer=optimizer, metrics=metrics)
        self.batch = batch
        self.warmup_epoch = warmup_epoch

    def train_step(self, data):
        x, y, u = data
        labels = y[0]
        bboxes = y[1]

        # model 1
        with tf.GradientTape() as tape1:
            # normal loss
            output_normal_1 = self.model1(x, training=True)
            ciou_loss_normal_1= 0
            conf_loss_normal_1 = 0
            prob_loss_normal_1 = 0

            conv_normal_1, pred_normal_1 = output_normal_1[0], output_normal_1[1]

            loss_items_normal_1 = normal_yolo_loss(pred_normal_1, conv_normal_1, labels, bboxes, STRIDES=self.stride,
                                                NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh, i=0)
            ciou_loss_normal_1 += loss_items_normal_1[0]
            conf_loss_normal_1 += loss_items_normal_1[1]
            prob_loss_normal_1 += loss_items_normal_1[2]

            normal_loss_1 = ciou_loss_normal_1 + conf_loss_normal_1 + prob_loss_normal_1

            # cps loss
            output_cps_2 = self.model2(u, training=True)
            ciou_loss_cps_1 = 0
            conf_loss_cps_1 = 0
            prob_loss_cps_1 = 0

            pred_cps_2 = output_cps_2[1]

            loss_items_cps_1 = cps_yolo_loss(pred_normal_1, conv_normal_1, pred_cps_2, STRIDES=self.stride,
                                             NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh,
                                             SCORE_THRESH=self.loss_score_thresh, i=0)

            ciou_loss_cps_1 += loss_items_cps_1[0]
            conf_loss_cps_1 += loss_items_cps_1[1]
            prob_loss_cps_1 += loss_items_cps_1[2]

            cps_loss_1 = ciou_loss_cps_1 + conf_loss_cps_1 + prob_loss_cps_1

            # total loss
            total_loss_1 = normal_loss_1 + self.cps_weight * cps_loss_1

        gradients = tape1.gradient(total_loss_1, self.model1.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model1.trainable_variables))

        # model 2
        with tf.GradientTape() as tape2:
            # normal loss
            output_normal_2 = self.model2(x, training=True)
            ciou_loss_normal_2 = 0
            conf_loss_normal_2 = 0
            prob_loss_normal_2 = 0

            conv_normal_2, pred_normal_2 = output_normal_2[0], output_normal_2[1]

            loss_items_normal_2 = normal_yolo_loss(pred_normal_2, conv_normal_2, labels, bboxes, STRIDES=self.stride,
                                                NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh, i=0)
            ciou_loss_normal_2 += loss_items_normal_2[0]
            conf_loss_normal_2 += loss_items_normal_2[1]
            prob_loss_normal_2 += loss_items_normal_2[2]

            normal_loss_2 = ciou_loss_normal_2 + conf_loss_normal_2 + prob_loss_normal_2

            # cps loss
            output_cps_1 = self.model1(u, training=True)
            ciou_loss_cps_2 = 0
            conf_loss_cps_2 = 0
            prob_loss_cps_2 = 0

            pred_cps_1 = output_cps_1[1]

            loss_items_cps_2 = cps_yolo_loss(pred_normal_2, conv_normal_2, pred_cps_1, STRIDES=self.stride,
                                             NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh,
                                             SCORE_THRESH=self.loss_score_thresh, i=0)

            ciou_loss_cps_2 += loss_items_cps_2[0]
            conf_loss_cps_2 += loss_items_cps_2[1]
            prob_loss_cps_2 += loss_items_cps_2[2]

            cps_loss_2 = ciou_loss_cps_2 + conf_loss_cps_2 + prob_loss_cps_2

            # total loss
            total_loss_2 = normal_loss_2 + self.cps_weight * cps_loss_2

        gradients = tape2.gradient(total_loss_2, self.model2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model2.trainable_variables))


        #self.model1.compiled_metrics.update_state(labels, pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"normal_1": normal_loss_1, "cps_1": cps_loss_1, "total_1": total_loss_1,
                        "normal_2": normal_loss_2, "cps_2": cps_loss_2, "total_2": total_loss_2})
        return results

    def test_step(self, data):
        x, y = data

        pred_result1 = self.model1(x, training=True)
        pred_result2 = self.model2(x, training=True)
        ciou_loss1 = conf_loss1 = prob_loss1 = 0
        ciou_loss1_cps = conf_loss1_cps = prob_loss1_cps = 0
        ciou_loss2 = conf_loss2 = prob_loss2 = 0
        ciou_loss2_cps = conf_loss2_cps = prob_loss2_cps = 0

        labels = y[0]
        bboxes = y[1]
        conv1, pred1 = pred_result1[0], pred_result1[1]
        conv2, pred2 = pred_result2[0], pred_result2[1]

        ##model 1
        #normal loss
        loss_items1 = normal_yolo_loss(pred1, conv1, labels, bboxes, STRIDES=self.stride,
                                      NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh, i=0)
        ciou_loss1 += loss_items1[0]
        conf_loss1 += loss_items1[1]
        prob_loss1 += loss_items1[2]

        normal_loss1 = ciou_loss1 + conf_loss1 + prob_loss1
        
        #cps loss
        loss_items1_cps = cps_yolo_loss(pred1, conv1, pred2, STRIDES=self.stride,
                                             NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh,
                                             SCORE_THRESH=self.loss_score_thresh, i=0)
        ciou_loss1_cps += loss_items1_cps[0]
        conf_loss1_cps += loss_items1_cps[1]
        prob_loss1_cps += loss_items1_cps[2]

        cps_loss1 = ciou_loss1_cps + conf_loss1_cps + prob_loss1_cps

        #total loss
        total_loss1 = normal_loss1 + cps_loss1

        ##model 2
        #normal loss
        loss_items2 = normal_yolo_loss(pred2, conv2, labels, bboxes, STRIDES=self.stride,
                                      NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh, i=0)
        ciou_loss2 += loss_items2[0]
        conf_loss2 += loss_items2[1]
        prob_loss2 += loss_items2[2]

        normal_loss2 = ciou_loss2 + conf_loss2 + prob_loss2

        #cps loss
        loss_items2_cps = cps_yolo_loss(pred2, conv2, pred1, STRIDES=self.stride,
                                             NUM_CLASS=self.output_class, IOU_LOSS_THRESH=self.iou_loss_thresh,
                                             SCORE_THRESH=self.loss_score_thresh, i=0)
        ciou_loss2_cps += loss_items2_cps[0]
        conf_loss2_cps += loss_items2_cps[1]
        prob_loss2_cps += loss_items2_cps[2]

        cps_loss2 = ciou_loss2_cps + conf_loss2_cps + prob_loss2_cps

        #total loss
        total_loss2 = normal_loss2 + cps_loss2

        #self.model1.compiled_metrics.update_state(labels, pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({"normal_1": normal_loss1, "cps_1": cps_loss1, "total_1": total_loss1,
                        "normal_2": normal_loss2, "cps_2": cps_loss2, "total_2": total_loss2,})
        return results

    def summary(self):
        self.model1.summary()
        self.model2.summary()


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def yolo_v4_more_tiny(input_layer, output_class, input_size, stride, anchors):
    route_1, conv = csp_darknet53_tiny(input_layer)

    conv = convolutional(conv, (1, 1, 512, 256))

    conv = convolutional(conv, (3, 3, 256, 512))

    conv_mbbox = convolutional(conv, (1, 1, 512, 3 * (output_class + 5)), activate=False, bn=False)  # 256

    bbox_tensors = []
    bbox_tensor = decode_train(conv_mbbox, input_size // stride, output_class, stride, anchors, 0)
    bbox_tensors.append(conv_mbbox)
    bbox_tensors.append(bbox_tensor)

    model = tf.keras.models.Model(input_layer, bbox_tensors)

    return model


def csp_darknet53_tiny(input_data):
    input_data = convolutional(input_data, (3, 3, 3, 32), downsample=True)
    input_data = convolutional(input_data, (3, 3, 32, 64), downsample=True)
    input_data = convolutional(input_data, (3, 3, 64, 64))

    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 32, 32))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 32, 32))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 32, 64))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 64, 128))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 64, 64))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 64, 64))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 64, 128))
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 128, 256))
    route = input_data
    input_data = route_group(input_data, 2, 1)
    input_data = convolutional(input_data, (3, 3, 128, 128))
    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 128, 128))
    input_data = tf.concat([input_data, route_1], axis=-1)
    input_data = convolutional(input_data, (1, 1, 128, 256))
    route_1 = input_data
    input_data = tf.concat([route, input_data], axis=-1)
    input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

    input_data = convolutional(input_data, (3, 3, 512, 512))

    return route_1, input_data


def decode_train(conv_output, output_size, NUM_CLASS, STRIDE, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = tf.reshape(conv_output,
                             (tf.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDE
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                                  padding=padding, use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn: conv = BatchNormalization()(conv)
    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv


def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)
