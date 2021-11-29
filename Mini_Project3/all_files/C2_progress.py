import numpy as np
import cv2
from imutils.video import FPS


VIDEO_PATH = '../test_videos/shooting.mp4'

DETECTOR_CLASS_FILE = '../cnn_labels/gun_detection.names'
DETECTOR_CONFIG_FILE = '../cfgs/gun_detection.cfg'
DETECTOR_WEIGHTS_FILE = '../cnn_weights/gun_detection.weights'

DETECTOR_MIN_CONFIDENCE = 0.4

SHOW_DETECTION = True

# thread queue
DETECTOR_MAX_THREAD_Q_SZ = 15

DETECTOR_LABELS = open(DETECTOR_CLASS_FILE).read().strip().split("\n")

np.random.seed(10)
COLORS = np.array([[0, 0, 255]], dtype='uint8')

# reduce BLOB_SIZE if GPU is not available
BLOB_SIZE = 512
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800

gun_detector_DNN = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG_FILE, DETECTOR_WEIGHTS_FILE)

# if using CPU change to: cv2.dnn.DNN_BACKEND_OPENCV
gun_detector_DNN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

# if using CPU change to : cv2.dnn.DNN_TARGET_CPU
gun_detector_DNN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

video_source = cv2.VideoCapture(VIDEO_PATH)
# video_source = cv2.VideoCapture(0)

fps = FPS()

layers_name = gun_detector_DNN.getLayerNames()
detector_out_layers_name = [layers_name[i[0] - 1] for i in gun_detector_DNN.getUnconnectedOutLayers()]

# wont work for webcam
good_frame, f = video_source.read()
if good_frame:
    HEIGHT = f.shape[0]
    WIDTH = f.shape[1]
else:
    raise Exception("Can't read video")


# def all_processing(video_feed, net):
#     while True:
#         (valid, frame) = video_feed.read()
#         if not valid:
#             break
#
#         blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
#                                      swapRB=True, crop=False)
#
#         net.setInput(blob)
#         layerOutputs = net.forward(detector_out_layers_name)
#
#         rects = []
#         probabilities = []
#         classCodes = []
#
#         for layer_output in layerOutputs:
#
#             for detected_obj in layer_output:
#                 if detected_obj[4] < 0.3:
#                     continue
#
#                 obj_confidence = detected_obj[5:]
#                 class_code = np.argmax(obj_confidence)
#                 max_confidence = obj_confidence[class_code]
#
#                 if max_confidence > DETECTOR_MIN_CONFIDENCE:
#                     rect = detected_obj[0:4] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
#                     (centerX, centerY, width, height) = rect.astype("int")
#
#                     top_left_x = int(centerX - (width / 2))
#                     top_left_y = int(centerY - (height / 2))
#
#                     rects.append([top_left_x, top_left_y, int(width), int(height)])
#                     probabilities.append(float(max_confidence))
#                     classCodes.append(class_code)
#
#         indices = cv2.dnn.NMSBoxes(rects, probabilities, DETECTOR_MIN_CONFIDENCE,
#                                    DETECTOR_MIN_CONFIDENCE)
#
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 (x, y) = (rects[i][0], rects[i][1])
#                 (w, h) = (rects[i][2], rects[i][3])
#
#                 obj_type = DETECTOR_LABELS[classCodes[i]]
#
#                 if SHOW_DETECTION:
#                     bgr = [int(c) for c in COLORS[classCodes[i]]]
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
#                     class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
#                     cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.5, bgr, 2)
#
#         # todo: give as input to  gui
#         cv2.imshow("output", cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
#         fps.update()
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break


video_feed = video_source
net = gun_detector_DNN


fps = FPS().start()

while True:
    (valid, frame) = video_feed.read()
    if not valid:
        break

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
                                 swapRB=True, crop=False)

    net.setInput(blob)
    layerOutputs = net.forward(detector_out_layers_name)

    rects = []
    probabilities = []
    classCodes = []

    for layer_output in layerOutputs:

        for detected_obj in layer_output:
            if detected_obj[4] < 0.3:
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

            if SHOW_DETECTION:
                bgr = [int(c) for c in COLORS[classCodes[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
                class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
                cv2.putText(frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, bgr, 2)

    # todo: give as input to  gui
    cv2.imshow("output", cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    fps.update()

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
video_source.release()
