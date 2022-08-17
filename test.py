import os
import cv2
import tensorflow as tf
import numpy as np
from datetime import datetime
from data import load_test_imageset
from iou import bbox_iou

def nms(boxes, num_classes, iou_threshold=0.5, score_threshold=0.25, max_boxes_per_class=50):
    batch, feath, featw, num_anchors, xywhcp = tf.keras.backend.int_shape(boxes)

    boxes = np.array(boxes).reshape(1, -1, xywhcp)  # (batch, -1, xywhcp)
    boxes_conf = boxes[..., 4:5]
    boxes_classprob = boxes[..., 5:]

    boxes_conf = boxes_conf * boxes_classprob
    boxes_conf = np.max(boxes_conf, axis=-1, keepdims=True)
    conf_mask = boxes_conf >= score_threshold
    boxes_conf = boxes_conf[conf_mask]
    boxes_conf = np.reshape(boxes_conf, newshape=(1, boxes_conf.shape[0], 1))
    # print("boxes_conf:{}".format(boxes_conf))

    conf_mask = np.tile(conf_mask, (1, 1, 5 + num_classes))
    boxes = boxes[conf_mask]
    boxes = np.reshape(boxes, newshape=(1, len(boxes) // (5 + num_classes), 5 + num_classes))

    boxes_classes = np.reshape(np.argmax(boxes[..., 5:], axis=-1), newshape=(1, -1, 1))
    boxes_coord = boxes[..., :4]

    bboxes_coord = []
    bboxes_scores = []
    bboxes_classes = []
    for class_ind in range(num_classes):
        mask_class = boxes_classes[..., 0:1] == class_ind
        boxes_class = boxes_classes[mask_class]
        boxes_conf_class = boxes_conf[mask_class]

        mask_class = np.tile(mask_class, (1, 1, 4))
        boxes_coord_class = boxes_coord[mask_class]
        boxes_coord_class = np.reshape(boxes_coord_class, (1, -1, 4))

        # conf 내림차순 정렬
        sorted_idx = np.argsort(-boxes_conf_class)
        # sorted_idx = sorted_idx[::-1]

        boxes_class = np.reshape(boxes_class, newshape=(len(sorted_idx), 1))
        boxes_class = boxes_class[sorted_idx]
        # boxes_class = np.expand_dims(boxes_class, axis=0)

        boxes_conf_class = np.reshape(boxes_conf_class, newshape=(len(sorted_idx), 1))
        boxes_conf_class = boxes_conf_class[sorted_idx]
        # boxes_conf_class = np.expand_dims(boxes_conf_class, axis=0)

        boxes_coord_class = np.reshape(boxes_coord_class, newshape=(len(sorted_idx), 4))
        boxes_coord_class = boxes_coord_class[sorted_idx]
        # boxes_coord_class = np.expand_dims(boxes_coord_class, axis=0)

        best_conf_ind = 0
        num_process = boxes_class.shape[0]
        while best_conf_ind+1 < num_process:
            iou_scores = bbox_iou(boxes_coord_class[best_conf_ind:best_conf_ind + 1, :],
                                  boxes_coord_class[best_conf_ind + 1:, :])
            iou_mask = iou_scores < iou_threshold
            iou_mask = np.reshape(iou_mask, newshape=(-1, 1))

            boxes_class = np.vstack([boxes_class[:best_conf_ind + 1, :],
                                     np.expand_dims(boxes_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])
            boxes_conf_class = np.vstack([boxes_conf_class[:best_conf_ind + 1, :],
                                          np.expand_dims(boxes_conf_class[best_conf_ind + 1:, :][iou_mask], axis=-1)])

            iou_mask = np.tile(iou_mask, (1, 4))
            boxes_coord_class = np.vstack([boxes_coord_class[:best_conf_ind + 1, :],
                                           np.reshape(boxes_coord_class[best_conf_ind + 1:, :][iou_mask],
                                                      newshape=(-1, 4))])

            best_conf_ind += 1
            num_process, _ = np.array(boxes_coord_class).shape

        bboxes_coord.append(boxes_coord_class)
        bboxes_scores.append(boxes_conf_class)
        bboxes_classes.append(boxes_class)

    # max_bbox = max_boxes_per_class * num_classes
    bboxes_coord = np.vstack(bboxes_coord)
    bboxes_scores = np.vstack(bboxes_scores)
    bboxes_classes = np.vstack(bboxes_classes)

    return bboxes_coord, bboxes_scores, bboxes_classes

if __name__ == '__main__':
    root_dir = 'D:/Public/qtkim/CPS/'
    print('===============================testing===============================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    batch_size = 2
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_path = root_dir + 'model/model(2022_08_17-09_41_45)'
    data_dir = root_dir + 'data/test'

    ##load dataset
    print('-----------------------load dataset-----------------------')
    test_images = load_test_imageset(data_dir)

    ##load model
    print('-----------------------load model------------------------')
    model = tf.keras.models.load_model(model_path)

    ##test
    print('--------------------------test---------------------------')
    test_result_dir = root_dir + 'test_result/' + start_time
    os.mkdir(test_result_dir)

    i = 0
    while True:
        batch_images = test_images[0:1, :, :, :]
        _, pred_res = model.predict(batch_images)
        res_xywh, res_score, _ = nms(pred_res, 3)
        img = np.array(batch_images[0], dtype=np.float32) * 255
        for j, box in enumerate(res_xywh):
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.rectangle(img, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)),
                                (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), (0, 0, 255), 2)
            img = cv2.putText(img, str(res_score[j][0]), (int(box[0]+box[2]/2), int(box[1]+box[3]/2)),
                              cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imwrite(test_result_dir + '/' + str(i) + '_img.png', img)

        test_images = test_images[1:, :, :, :]
        i += 1
        if len(test_images) == 0:
            break
