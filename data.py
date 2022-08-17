import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from iou import bbox_iou


def shuffle(train_set):
    if train_set is None or len(train_set) == 0: return [[], []]
    tmp = list(zip(train_set[0], train_set[1]))
    random.shuffle(tmp)
    image, label = zip(*tmp)
    return [list(image), list(label)]


def load_dataset(root_dir, valid_ratio=0.2, batch_size=4, is_train=True, normalize=True):
    img_dir = os.path.join(root_dir, 'image')
    lbl_dir = os.path.join(root_dir, 'label')
    ul_img_dir = os.path.join(root_dir, 'unlabeled')
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpeg', 'jpg', '.bmp'))]
    lbl_files = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.endswith(('.txt'))]
    ul_img_files = [os.path.join(ul_img_dir, f) for f in os.listdir(ul_img_dir) if f.endswith(('.png', '.jpeg',
                                                                                               'jpg', '.bmp'))]
    total_img_num = len(img_files)
    valid_idx = int(total_img_num * valid_ratio)
    dataset = [[], []]
    dataset[0].extend(img_files)
    dataset[1].extend(lbl_files)
    if is_train:
        dataset = shuffle(dataset)
        random.shuffle(ul_img_files)
    img_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640))
                for f in dataset[0][valid_idx:]]
    lbl_list = [load_yolo_label(f) for f in dataset[1][valid_idx:]]
    ul_img_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640))
                   for f in ul_img_files[valid_idx:]]
    img_list_val = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640))
                    for f in dataset[0][:valid_idx]]
    lbl_list_val = [load_yolo_label(f) for f in dataset[1][:valid_idx]]
    images = np.array(img_list, dtype=np.float32)
    ul_images = np.array(ul_img_list, dtype=np.float32)
    labels, bboxes = preprocess_true_boxes(lbl_list)
    images_val = np.array(img_list_val, dtype=np.float32)
    labels_val, bboxes_val = preprocess_true_boxes(lbl_list_val)
    if normalize:
        images = images / 255.
        images_val = images_val / 255.
        ul_images = ul_images / 255.
    if is_train:
        images = images[:, :, :, np.newaxis]
        images_val = images_val[:, :, :, np.newaxis]
        ul_images = ul_images[:, :, :, np.newaxis]

    img_num = images.shape[0]
    val_img_num = images_val.shape[0]
    steps_per_epoch = int(img_num / batch_size) + bool(img_num % batch_size)
    validation_steps = int(val_img_num / batch_size) + bool(val_img_num % batch_size)

    images = tf.data.Dataset.from_tensor_slices(images)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    bboxes = tf.data.Dataset.from_tensor_slices(bboxes)
    ul_images = tf.data.Dataset.from_tensor_slices(ul_images)
    train_dataset = tf.data.Dataset.zip((images, (labels, bboxes), ul_images))
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

    images_val = tf.data.Dataset.from_tensor_slices(images_val)
    labels_val = tf.data.Dataset.from_tensor_slices(labels_val)
    bboxes_val = tf.data.Dataset.from_tensor_slices(bboxes_val)
    valid_dataset = tf.data.Dataset.zip((images_val, (labels_val, bboxes_val)))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(AUTOTUNE)

    return train_dataset, valid_dataset, steps_per_epoch, validation_steps


def load_test_imageset(root_dir, normalize=True):
    img_dir = os.path.join(root_dir, 'image')
    img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpeg', 'jpg', '.bmp'))]
    img_list = [cv2.resize(cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32), (640, 640)) for f in img_files]
    images = np.array(img_list, dtype=np.float32)
    if normalize:
        images = images / 255.
    images = images[:, :, :, np.newaxis]
    return images


def load_yolo_label(label_fullpath):
    text = tf.io.read_file(label_fullpath)

    if text == '':
        bboxes = tf.constant([[0,0,0,0,-1]], dtype=tf.float32)
    else:
        boxes = tf.strings.to_number(tf.strings.split(text))
        boxes = tf.reshape(boxes, (len(boxes) // 5, 5))
        # class_num, center_x, center_y, half_w, half_h = tf.split(boxes, 5, 1)
        class_num, center_x, center_y, width_, height_ = tf.split(boxes, 5, 1)
        bboxes = tf.concat([center_x,
                            center_y,
                            width_,
                            height_,
                            class_num],
                           1)
        bboxes = tf.multiply(bboxes, tf.constant([640, 640, 640, 640, 1], dtype=tf.float32))
    return bboxes


def preprocess_true_boxes(lbl_bboxes_list):
    input_size = 640
    output_size = 20
    stride = np.array([32],np.int32)
    anchor = np.array([80],np.int32)
    anchor_num = 3
    class_num = 3

    label_res = []
    bboxes_res = []

    for lbl_bboxes in lbl_bboxes_list:
        label = [np.zeros((output_size, output_size, anchor_num, 5 + class_num,))]
        bboxes = [np.zeros((150, 5))]

        count_overlap = [np.zeros((output_size, output_size,), dtype=np.int32)]

        label, bboxes = preprocess_yolo_layer(label, lbl_bboxes, bboxes, count_overlap, class_num, input_size, output_size,
                                              anchor, anchor_num, stride, 0)

        feat_h = label[0].shape[0]
        feat_w = label[0].shape[1]
        label[0] = np.reshape(label[0], (feat_h, feat_w, 3, -1))
        bboxes[0] = np.reshape(bboxes[0], [-1, 5])

        label_res.append(label[0])
        bboxes_res.append(bboxes[0])

    return np.array(label_res, dtype=np.float32), np.array(bboxes_res, dtype=np.float32)


def preprocess_yolo_layer(label, bboxes, bboxes_xywhc, count_overlap, class_num, input_size, output_size, anchor,
                          anchor_num, stride, i):
    feat_h, feat_w, _, _ = label[i].shape
    bbox_count = np.zeros((1,))

    if np.cast[np.int](bboxes[0][4]) == -1:
        return label, bboxes_xywhc

    for bbox in bboxes:
        bbox_coor = bbox[:4]
        bbox_class_ind = np.cast[np.int](bbox[4])
        onehot = np.zeros(class_num, dtype=np.float)
        onehot[bbox_class_ind] = 1.0
        uniform_distribution = np.full(class_num, 1.0 / class_num)
        delta = 0.01
        smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution

        bbox_xywh = np.concatenate([bbox_coor], axis=-1,)
        bbox_xywh = np.minimum(bbox_xywh, input_size - 1)
        bbox_xywh = np.maximum(bbox_xywh, 0)
        bbox_xywh_scaled = (1.0 * bbox_xywh[np.newaxis, :] / stride[0])

        iou = []
        exist_positive = False

        bbox_xywh_scaled = np.minimum(bbox_xywh_scaled, output_size - 1)
        bbox_xywh_scaled = np.maximum(bbox_xywh_scaled, 0)

        anchors_xywh = np.zeros((anchor[0], 4), dtype=np.float32)
        anchors_xywh[:, 0:2] = (
                np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
        )

        anchors_xywh[:, 2:4] = (anchor[0] / stride[0]).astype(np.float32)

        iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)

        iou.append(iou_scale)
        iou_mask = iou_scale > 0.3

        if np.any(iou_mask):
            xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

            label_idx = count_overlap[i][yind, xind]
            label[i][yind, xind, label_idx:label_idx + 1, :] = 0
            label[i][yind, xind, label_idx:label_idx + 1, 0:4] = bbox_xywh
            label[i][yind, xind, label_idx:label_idx + 1, 4:5] = 1.0
            label[i][yind, xind, label_idx:label_idx + 1, 5:] = smooth_onehot
            count_overlap[i][yind, xind] += 1  # 한 grid 에 중심점 3개 초과시 에러날 것

            bbox_ind = int(bbox_count[i] % 150)
            bboxes_xywhc[i][bbox_ind, :4] = bbox_xywh
            bboxes_xywhc[i][bbox_ind, 4:5] = bbox_class_ind
            bbox_count[i] += 1

            exist_positive = True


        if not exist_positive:
            best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
            best_detect = int(best_anchor_ind / anchor_num)
            best_anchor = int(best_anchor_ind % anchor_num)
            xind, yind = np.floor(
                bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

            label_idx = count_overlap[best_detect][yind, xind]
            label[best_detect][yind, xind, label_idx:label_idx + 1, :] = 0
            label[best_detect][yind, xind, label_idx:label_idx + 1, 0:4] = bbox_xywh
            label[best_detect][yind, xind, label_idx:label_idx + 1, 4:5] = 1.0
            label[best_detect][yind, xind, label_idx:label_idx + 1, 5:] = smooth_onehot
            count_overlap[best_detect][yind, xind] += 1

            bbox_ind = int(
                bbox_count[best_detect] % 150
            )
            bboxes_xywhc[best_detect][bbox_ind, :4] = bbox_xywh
            bboxes_xywhc[i][bbox_ind, 4:5] = bbox_class_ind
            bbox_count[best_detect] += 1

    return label, bboxes_xywhc
