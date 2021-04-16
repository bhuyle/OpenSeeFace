from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

from utils import draw_bbox_maxmin, write_text
from utils import get_config

from src import detect_face_ssd

# deep_sort
from libs import preprocessing
from libs import nn_matching
from libs import Detection
from libs import Tracker
from utils import generate_detections as gdet
from libs import Detection as ddet
from collections import deque
# import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'


PROTOTXT = 'models/dnn/deploy.prototxt.txt'
MODEL = 'models/dnn/res10_300x300_ssd_iter_140000.caffemodel'

def setup_config(cfg):
    if cfg.MAIN.path_file_config_cam != None:
        cfg.merge_from_file(cfg.MAIN.path_file_config_cam)
    
    if cfg.MAIN.path_file_deep_sort != None: 
        cfg.merge_from_file(cfg.MAIN.path_file_deep_sort)

def setup_deep_sort(cfg):
    encoder = gdet.create_box_encoder(
            cfg.DEEPSORT.MODEL, batch_size=4)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", cfg.DEEPSORT.MAX_COSINE_DISTANCE, cfg.DEEPSORT.NN_BUDGET)
    tracker = Tracker(cfg, metric)

    return encoder, metric, tracker

def main(tracking=False):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    # cap.open("{}://{}:{}@{}:{}".format(proto, user, password, ip, port))

    # cnt = 0 
    while(True):
        start_time = time.time()

        ret, frame = cap.read()
        # frame = imutils.resize(frame, width=400)
    
        ################ PROCESS ##########################
        # detect
        list_face_bbox, list_score, list_classes = detect_face_ssd(frame, PROTOTXT, MODEL)
        for index in list_face_bbox:
            if index[2] > frame.shape[1]-10:
                index[2] = frame.shape[1]-20
        # tracking
        if tracking:
 
            features = encoder(frame, list_face_bbox)
            detections = [Detection(bbox, 1.0, cls, feature) for bbox, _, cls, feature in
                            zip(list_face_bbox, list_score, list_classes, features)]
            # run nms
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            # boxes = np.array([d for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(
                boxes, cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
            detections = [detections[i] for i in indices]
            if len(list_face_bbox) != 0:
                tracker.predict()
                tracker.update(detections)
            else:
                tracker.predict()

                # tracker.update(detections)
            id = []
            list_bbox = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                # bbox = track.mean[:4].copy()
                bbox = track.to_tlbr()
                id.append(track.track_id)
                list_bbox.append(bbox)
                # draw track
                # frame = draw_bbox_maxmin(frame, list_face_bbox, True, track.track_id)
            # print(list_bbox,list_face_bbox)
            list_bbox = np.array(list_bbox).astype('int32')
            for i,box in enumerate(list_bbox):
                if id == [] or len(id) != len(list_bbox):
                    break
                frame = draw_bbox_maxmin(frame, box, True, id[i])

        
        else:
            for face_bbox, face_score in zip(list_face_bbox, list_score):
                text = "{:.2f}%".format(face_score * 100)

                frame = draw_bbox_maxmin(frame, face_bbox)
                frame = write_text(frame, text, face_bbox[0], face_bbox[1]-3)

        # cal fps
        fps =  round(1.0 / (time.time() - start_time), 2)
        frame = write_text(frame, "fps: {}".format(fps), 5, 15)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def track_frame(list_face_bbox,frame):
    features = encoder(frame, list_face_bbox)
    detections = [Detection(bbox, 1.0, 'person', feature) for bbox,feature in
                    zip(list_face_bbox, features)]
    boxes = np.array([d.tlwh for d in detections])
        # boxes = np.array([d for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]
    if len(list_face_bbox) != 0:
        tracker.predict()
        tracker.update(detections)
    else:
        tracker.predict()
    id = []
    list_bbox = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        # bbox = track.mean[:4].copy()
        bbox = track.to_tlbr()
        id.append(track.track_id)
        list_bbox.append(bbox)

    list_bbox = np.array(list_bbox).astype('int32')
    for i,box in enumerate(list_bbox):
        if id == [] or len(id) != len(list_bbox):
            break
        frame = draw_bbox_maxmin(frame, box, True, id[i])
    return frame

cfg = get_config()
cfg.merge_from_file("configs/main_process.yaml")
setup_config(cfg)
encoder, metric, tracker = setup_deep_sort(cfg)

if __name__ == '__main__':
    cfg = get_config()
    cfg.merge_from_file("configs/main_process.yaml")

    # call setup_config
    setup_config(cfg)

    # setup to call cam
    proto = cfg.CAM.proto
    user = cfg.CAM.user 
    password = cfg.CAM.password
    ip = cfg.CAM.ip 
    port = cfg.CAM.port 

    # setup deep_sort
    encoder, metric, tracker = setup_deep_sort(cfg)

    # main(True)