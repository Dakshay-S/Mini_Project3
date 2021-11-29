import cv2
import numpy as np
import os

DETECTOR_CLASS_FILE = '../cnn_labels/gun_detection.names'
DETECTOR_CONFIG_FILE = '../cfgs/gun_detection.cfg'
DETECTOR_WEIGHTS_FILE = '../cnn_weights/gun_detection.weights'

DETECTOR_MIN_CONFIDENCE = 0.1
OBJ_MIN_CONFIDENCE = 0.1

# reduce BLOB_SIZE if GPU is not available
BLOB_SIZE = 480
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800

DETECTOR_LABELS = open(DETECTOR_CLASS_FILE).read().strip().split("\n")

COLORS = np.array([[0, 0, 255]], dtype='uint8')

gun_detector_DNN = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG_FILE, DETECTOR_WEIGHTS_FILE)

# if using CPU change to: cv2.dnn.DNN_BACKEND_OPENCV
gun_detector_DNN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

# if using CPU change to : cv2.dnn.DNN_TARGET_CPU
gun_detector_DNN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layers_name = gun_detector_DNN.getLayerNames()
detector_out_layers_name = [layers_name[i[0] - 1] for i in gun_detector_DNN.getUnconnectedOutLayers()]

net = gun_detector_DNN


def predict(img_name, dir_path):
    img_path = f'../testimgs/{img_name}'
    frame = cv2.imread(img_path)

    HEIGHT = frame.shape[0]
    WIDTH = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(detector_out_layers_name)

    rects = []
    probabilities = []
    classCodes = []

    for layer_output in layerOutputs:

        for detected_obj in layer_output:
            if detected_obj[4] < OBJ_MIN_CONFIDENCE:
                continue

            obj_confidence = detected_obj[5:]
            class_code = np.argmax(obj_confidence)
            max_confidence = obj_confidence[class_code]

            if max_confidence > DETECTOR_MIN_CONFIDENCE:
                rect = detected_obj[0:4] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                (centerX, centerY, width, height) = rect.astype("int")

                top_left_x = int(centerX - (width / 2))
                top_left_y = int(centerY - (height / 2))

                rects.append([top_left_x, top_left_y, int(width), int(height)])
                probabilities.append(float(max_confidence))
                classCodes.append(class_code)

    indices = cv2.dnn.NMSBoxes(rects, probabilities, DETECTOR_MIN_CONFIDENCE,
                               DETECTOR_MIN_CONFIDENCE)

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (rects[i][0], rects[i][1])
            (w, h) = (rects[i][2], rects[i][3])

            obj_type = DETECTOR_LABELS[classCodes[i]]

            bgr = [int(c) for c in COLORS[classCodes[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
            cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, bgr, 2)

    if SHOW_DETECTION:

        cv2.imshow("output", cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        cv2.waitKey(0)
    else:
        cv2.imwrite(dir_path + img_name, frame)


if __name__ == '__main__':
    imgs = os.listdir('../testimgs/')
    SHOW_DETECTION = False

    dir = '../testResults/'

    for img in imgs:
        predict(img, dir)
