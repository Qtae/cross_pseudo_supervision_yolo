import tensorflow as tf
import numpy as np
from iou import bbox_ciou, bbox_iou


def normal_yolo_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]
    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    # giou_loss = respond_bbox * bbox_loss_scale * (1- giou)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                   bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :4])  # :4 <- :
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < IOU_LOSS_THRESH, tf.float32)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss

def cps_yolo_loss(pred1, conv1, pred2, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, SCORE_THRESH, i=0):
    conv_shape = tf.shape(conv1)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES * output_size
    conv1 = tf.reshape(conv1, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv1[:, :, :, :, 4:5]
    conv_raw_prob = conv1[:, :, :, :, 5:]
    pred_xywh = pred1[:, :, :, :, 0:4]
    pred_conf = pred1[:, :, :, :, 4:5]

    label_xywh = pred2[:, :, :, :, 0:4]
    respond_bbox = pred2[:, :, :, :, 4:5]
    label_prob = pred2[:, :, :, :, 5:]
    label_score = respond_bbox * label_prob
    label_max_prob = tf.reduce_max(label_score, axis=-1, keepdims=True)
    score_mask = label_max_prob >= SCORE_THRESH
    _, feath, featw, num_anchors, xywhcp = tf.keras.backend.int_shape(pred2)
    score_mask = tf.tile(score_mask, [1, 1, 1, 1, xywhcp])
    score_mask = tf.cast(score_mask, dtype=tf.float32)
    label_xywh = label_xywh * score_mask[:, :, :, :, 0:4]
    respond_bbox = respond_bbox * score_mask[:, :, :, :, 4:5]
    label_prob = label_prob * score_mask[:, :, :, :, 5:]

    ciou = tf.expand_dims(bbox_ciou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * ( respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                      logits=conv_raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    ciou_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return ciou_loss, conf_loss, prob_loss
