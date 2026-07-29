import numpy as np
import cv2
from utils import *
import os
import time
import argparse

parser = argparse.ArgumentParser(description="Advanced ADAS Lane & Vehicle Detection System")
parser.add_argument('--model_cfg', type=str, default='',
                    help='Path to YOLO config file (e.g. yolov3.cfg)')
parser.add_argument('--model_weights', type=str, default='',
                    help='Path to YOLO weights file (e.g. yolov3.weights)')
parser.add_argument('--onnx', type=str, default='',
                    help='Path to ONNX model file (e.g. yolov3.onnx)')
parser.add_argument('--video', type=str, default='',
                    help='Path to input video file')
parser.add_argument('--src', type=int, default=0,
                    help='Source camera index if no video file is provided')
parser.add_argument('--output_dir', type=str, default='',
                    help='Path to output directory for saving results')
parser.add_argument('--detect_interval', type=int, default=3,
                    help='Interval (in frames) for running YOLO object detection (default: 3 for high FPS)')
args = parser.parse_args()

print('----- ADAS System Info -----')
print('[i] Config file:      ', args.model_cfg if args.model_cfg else 'None')
print('[i] Weights file:     ', args.model_weights if args.model_weights else 'None')
print('[i] ONNX model file:  ', args.onnx if args.onnx else 'None')
print('[i] Input Video file: ', args.video if args.video else f'Camera #{args.src}')
print(f'[i] YOLO Interval:    Every {args.detect_interval} frames (Optimized for High FPS)')
print('###########################################################\n')

frameWidth = 640
frameHeight = 480

# Check and setup Object Detection
has_yolo = False
net = None
classes = []
output_layers = []
colors = []

coco_path = "coco.names"
if not os.path.isfile(coco_path):
    coco_path = os.path.join(os.path.dirname(__file__), "coco.names")

if os.path.isfile(coco_path):
    with open(coco_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
else:
    classes = [f"class_{i}" for i in range(80)]

if args.onnx and os.path.isfile(args.onnx):
    try:
        net = cv2.dnn.readNetFromONNX(args.onnx)
        colors = np.random.uniform(50, 255, size=(len(classes), 3))
        output_layers = net.getUnconnectedOutLayersNames()
        has_yolo = True
        print('[i] ONNX model loaded successfully!')
    except Exception as e:
        print(f'[!] Warning: Failed to load ONNX model: {e}')
elif args.model_cfg and args.model_weights:
    if os.path.isfile(args.model_cfg) and os.path.isfile(args.model_weights):
        try:
            net = cv2.dnn.readNet(args.model_weights, args.model_cfg)
            layers_names = net.getLayerNames()
            out_layers = net.getUnconnectedOutLayers()
            if len(out_layers.shape) == 1 or isinstance(out_layers[0], (int, np.integer)):
                output_layers = [layers_names[i - 1] for i in out_layers]
            else:
                output_layers = [layers_names[i[0] - 1] for i in out_layers]

            colors = np.random.uniform(50, 255, size=(len(classes), 3))
            has_yolo = True
            print('[i] YOLO Darknet model loaded successfully!')
        except Exception as e:
            print(f'[!] Warning: Failed to load YOLO Darknet model: {e}')
            print('[!] Continuing with lane detection only.')
    else:
        print('[!] Warning: Specified YOLO weights or config file not found.')
else:
    print('[!] Note: Object detection model not specified. Running Lane Detection only.')

font = cv2.FONT_HERSHEY_PLAIN
frame_id = 0

cameraFeed = False
if not args.video:
    cameraFeed = True

if cameraFeed:
    cameraNo = args.src
    intialTracbarVals = [24, 55, 12, 100]
    cap = cv2.VideoCapture(cameraNo)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    output_file = 'live_camera_Detection.avi'
else:
    intialTracbarVals = [42, 63, 14, 87]
    cap = cv2.VideoCapture(args.video)
    base_name = os.path.basename(args.video)
    output_file = (base_name[:-4] if len(base_name) > 4 else base_name) + '_Detection.avi'

if not cap.isOpened():
    print(f"\n[!] Error: Could not open video source ({'Camera #' + str(args.src) if cameraFeed else args.video}).")
    exit(1)

noOfArrayValues = 10
arrayCounter = 0
arrayCurve = np.zeros([noOfArrayValues])
initializeTrackbars(intialTracbarVals)

fps_val = cap.get(cv2.CAP_PROP_FPS)
if not fps_val or np.isnan(fps_val) or fps_val <= 0:
    fps_val = 20.0

output_path = os.path.join(args.output_dir, output_file) if args.output_dir else output_file
video_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps_val,
    (frameWidth, frameHeight)
)

screenshots_dir = "screenshots"
os.makedirs(screenshots_dir, exist_ok=True)

starting_time = time.time()
cached_detections = []
paused = False

print("\n================ INTERACTIVE CONTROLS ================")
print("  Press 'P' - Pause / Resume video stream")
print("  Press 'S' - Save current HUD screenshot")
print("  Press 'Q' - Exit program")
print("======================================================\n")

while True:
    if not paused:
        success, img = cap.read()
        if not success:
            print('[i] ==> Processing complete!')
            break

        img = cv2.resize(img, (frameWidth, frameHeight), None)
        frame_id += 1

        imgWarpPoints = img.copy()
        imgFinal = img.copy()
        imgCanny = img.copy()

        # 1. Lane Detection Pipeline
        imgUndis = undistort(img)
        imgThres, imgCanny, imgColor = thresholding(imgUndis)
        src = valTrackbars()
        imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
        imgWarpPoints = drawPoints(imgWarpPoints, src)
        imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)

        lane_curve = 0
        try:
            curverad = get_curve(imgFinal, curves[0], curves[1])
            lane_curve = np.mean([curverad[0], curverad[1]])
            imgFinal = draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

            # Rolling Average for Smoothing
            currentCurve = lane_curve // 50
            if int(np.sum(arrayCurve)) == 0:
                averageCurve = currentCurve
            else:
                averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]

            if abs(averageCurve - currentCurve) > 200:
                arrayCurve[arrayCounter] = averageCurve
            else:
                arrayCurve[arrayCounter] = currentCurve

            arrayCounter = (arrayCounter + 1) % noOfArrayValues
        except Exception:
            pass

        imgFinal = drawLines(imgFinal, lane_curve)

        # 2. Optimized Object Detection (Run every detect_interval frames)
        frame = img.copy()
        height, width, _ = frame.shape

        if has_yolo:
            should_run_yolo = (frame_id % max(1, args.detect_interval) == 0) or (frame_id == 1)
            if should_run_yolo:
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                ALLOWED_VEHICLES = {'car', 'truck', 'bus', 'motorbike', 'bicycle', 'person'}
                cached_detections = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.45:
                            class_name = classes[class_id].lower() if class_id < len(classes) else ""
                            if class_name in ALLOWED_VEHICLES:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w_box = int(detection[2] * width)
                                h_box = int(detection[3] * height)
                                x = int(center_x - w_box / 2)
                                y = int(center_y - h_box / 2)

                                # Exclude full-width bonnet/hood detections near the bottom edge
                                if y + h_box > height * 0.95 and w_box > width * 0.7:
                                    continue

                                cached_detections.append((x, y, w_box, h_box, confidence, class_id))

                # Non-Maximum Suppression
                if cached_detections:
                    boxes = [[d[0], d[1], d[2], d[3]] for d in cached_detections]
                    confidences = [d[4] for d in cached_detections]
                    class_ids = [d[5] for d in cached_detections]
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.3)
                    
                    filtered_detections = []
                    if len(indexes) > 0:
                        for i in indexes.flatten():
                            filtered_detections.append((boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], confidences[i], class_ids[i]))
                    cached_detections = filtered_detections

            # Render bounding boxes on frame & final HUD
            for (x, y, w_box, h_box, confidence, class_id) in cached_detections:
                class_name = classes[class_id] if class_id < len(classes) else "object"
                label = f"{class_name.upper()}: {confidence*100:.1f}%"
                color_tuple = colors[class_id % len(colors)]
                bgr_color = (int(color_tuple[0]), int(color_tuple[1]), int(color_tuple[2]))

                frame = drawFancyBoundingBox(frame, x, y, w_box, h_box, label, bgr_color)
                imgFinal = drawFancyBoundingBox(imgFinal, x, y, w_box, h_box, label, bgr_color)

        # 3. Render Dashboard HUD
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time if elapsed_time > 0 else 0
        imgFinal = drawHUD(imgFinal, lane_curve, fps)

        # Stacked debug pipeline view
        imgStacked = stackImages(0.7, ([imgUndis, frame],
                                      [imgColor, imgCanny],
                                      [imgWarp, imgSliding]
                                      ))

        video_writer.write(imgFinal)

    # Display Windows
    cv2.imshow("Image (Object Detection)", frame)
    cv2.imshow("Pipeline (Multi-View)", imgStacked)
    cv2.imshow("Result (ADAS Dashboard HUD)", imgFinal)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("[i] Exiting program...")
        break
    elif key == ord('p') or key == ord('P'):
        paused = not paused
        status_msg = "PAUSED" if paused else "RESUMED"
        print(f"[i] Video playback {status_msg}")
    elif key == ord('s') or key == ord('S'):
        ts = int(time.time())
        ss_filename = os.path.join(screenshots_dir, f"adas_hud_{ts}.jpg")
        cv2.imwrite(ss_filename, imgFinal)
        print(f"[!] Saved HUD Screenshot to: {ss_filename}")

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print('==> All done!')
print('***********************************************************')
