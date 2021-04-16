import copy
import os
import sys
import argparse
import traceback
import gc
from tracker import *
from access_camera import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=15)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=0)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=5)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=1)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=1, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if not self.fh is None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if not self.fh is None:
            self.fh.flush()
        self.output.flush()
output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if not args.blackmagic_options is None:
        dshowcapture.set_options(args.blackmagic_options)
    if not args.priority is None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])


import numpy as np
import time
import cv2
import socket
import struct
import json
from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path


target_ip = args.ip
target_port = args.port

if args.faces >= 40:
    print("Transmission of tracking data over network is not supported with 40 or more faces.")

fps = 0
dcap = None
use_dshowcapture_flag = False
if os.name == 'nt':
    fps = args.fps
    dcap = args.dcap
    use_dshowcapture_flag = True if args.use_dshowcapture == 1 else False
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag)
if type(input_reader.reader) == VideoReader:
    fps = 0

log = None
out = None
first = True
height = 0
width = 0
tracker = None
sock = None
total_tracking_time = 0.0
tracking_time = 0.0
tracking_frames = 0
frame_count = 0

features = ["eye_l", "eye_r", "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l", "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r", "mouth_corner_updown_l", "mouth_corner_inout_l", "mouth_corner_updown_r", "mouth_corner_inout_r", "mouth_open", "mouth_wide"]

is_camera = args.capture == str(try_int(args.capture))

try:
    attempt = 0
    frame_time = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1. / float(fps)
    repeat = args.repeat_video != 0 and type(input_reader.reader) == VideoReader
    need_reinit = 0
    failures = 0
    source_name = input_reader.name

    ### Setup deep sort
    cfg = get_config()
    cfg.merge_from_file("configs/main_process.yaml")
    setup_config(cfg)
    encoder, metric, tracker = setup_deep_sort(cfg)

    while repeat or input_reader.is_open():
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(args.capture, args.raw_rgb, args.width, args.height, fps, use_dshowcapture=use_dshowcapture_flag, dcap=dcap)
            if input_reader.name != source_name:
                print(f"Failed to reinitialize camera and got {input_reader.name} instead of {source_name}.")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.02)
            continue
        if not input_reader.is_ready():
            time.sleep(0.02)
            continue

        ret, frame = input_reader.read()
        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif is_camera:
                attempt += 1
                if attempt > 30:
                    break
                else:
                    time.sleep(0.02)
                    if attempt == 3:
                        need_reinit = 1
                    continue
            else:
                break;

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
        try:
            start = time.time()
            inference_start = time.perf_counter()
            faces = tracker.predict_bboxonly(frame)

            if len(faces) > 0:
                inference_time = (time.perf_counter() - inference_start)
                total_tracking_time += inference_time
                tracking_time += inference_time / len(faces)
                tracking_frames += 1
            packet = bytearray()
            detected = False
            list_face_bbox = []
            for face_num, f in enumerate(faces):
                f = copy.copy(f)
                box = f.bbox
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                list_face_bbox.append([x1,y1,x1+x2,y1+y2])
                cv2.rectangle(frame,(x1,y1),(x1+x2,y1+y2),(255,255,0),2)
                f.id += args.face_id_offset
                detected = True
            frame = track_frame(list_face_bbox,frame)

            end = time.time()
            fps = str(int(1/(end-start)))
            cv2.putText(frame, fps,(5,25),
            cv2.FONT_HERSHEY_SIMPLEX,1,  # font scale
            (255, 0, 255),3)
            if args.visualize != 0:
                cv2.imshow('OpenSeeFace Visualization', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            failures = 0
        except Exception as e:
            if e.__class__ == KeyboardInterrupt:
                if args.silent == 0:
                    print("Quitting")
                break
            traceback.print_exc()
            failures += 1
            if failures > 30:
                break

        collected = False
        del frame

except KeyboardInterrupt:
    if args.silent == 0:
        print("Quitting")

input_reader.close()
if not out is None:
    out.release()
cv2.destroyAllWindows()