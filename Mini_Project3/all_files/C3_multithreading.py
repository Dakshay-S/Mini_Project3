import numpy as np
import cv2
from imutils.video import FPS
from queue import Queue
from threading import Thread


VIDEO_PATH = '../test_videos/shooting.mp4'

DETECTOR_CLASS_FILE = '../cnn_labels/gun_detection.names'
DETECTOR_CONFIG_FILE = '../cfgs/gun_detection.cfg'
DETECTOR_WEIGHTS_FILE = '../cnn_weights/gun_detection.weights'

DETECTOR_MIN_CONFIDENCE = 0.1
OBJ_MIN_CONFIDENCE = 0.3

SHOW_DETECTION = True

# thread queue
DETECTOR_MAX_THREAD_Q_SZ = 15
detector_frame_queue = Queue(DETECTOR_MAX_THREAD_Q_SZ)
detector_blob_queue = Queue(DETECTOR_MAX_THREAD_Q_SZ)


DETECTOR_LABELS = open(DETECTOR_CLASS_FILE).read().strip().split("\n")

np.random.seed(10)
COLORS = np.array([[0,0,255]], dtype='uint8')


# reduce BLOB_SIZE if GPU is not available
BLOB_SIZE = 416
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


def make_blob_from_frame(video_feed):
    while True:
        (valid, frame) = video_feed.read()
        if not valid:
            break

        detector_frame_queue.put(frame)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
                                     swapRB=True, crop=False)
        detector_blob_queue.put(blob)


def all_processing(net):
    # global tracker
    while True:
        try:
            curr_frame = detector_frame_queue.get(timeout=1)
            blob = detector_blob_queue.get(timeout=1)
        except:
            break

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

                if SHOW_DETECTION:
                    bgr = [int(c) for c in COLORS[classCodes[i]]]
                    cv2.rectangle(curr_frame, (x, y), (x + w, y + h), bgr, 2)
                    class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
                    cv2.putText(curr_frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bgr, 2)

        # del_y = 30
        # cv2.putText(curr_frame, f"frame_q : {detector_frame_queue.qsize()}", (20, 1 * del_y),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 2)
        # cv2.putText(curr_frame, f"blob_q : {detector_blob_queue.qsize()}", (20, 2 * del_y),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (255, 255, 255), 2)

        # todo: give as input to  gui
        cv2.imshow("output", cv2.resize(curr_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        fps.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


blob_thread = Thread(target=make_blob_from_frame, args=(video_source,), daemon=True)
blob_thread.start()

fps = FPS().start()
all_processing_thread = Thread(target=all_processing, args=(gun_detector_DNN,), daemon=True)
all_processing_thread.start()

blob_thread.join()
# reader_thread.join()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


cv2.destroyAllWindows()

video_source.release()