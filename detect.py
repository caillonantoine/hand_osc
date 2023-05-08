import logging
from time import sleep, time

import cv2
import mediapipe as mp
from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

logging.basicConfig(level=logging.INFO)

vid = cv2.VideoCapture(1)

model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

client = udp_client.SimpleUDPClient("127.0.0.1", port=1893)


def get_ms_counter():
    start = time()

    def counter():
        ellapsed = time() - start
        return int(1000 * ellapsed)

    return counter


def live_hand_detection(landmarker: HandLandmarker):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    counter = get_ms_counter()

    logging.info("starting detection")
    while True:
        _, img = cam.read()
        img = cv2.flip(img, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        landmarker.detect_async(mp_image, counter())
        sleep(.01)


def detection_callback(result: HandLandmarkerResult, output_image: mp.Image,
                       timestamp_ms: int):
    if result.hand_landmarks:
        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY)
        for hand, landmarks in zip(result.handedness, result.hand_landmarks):
            left_right = hand[0].category_name.lower()
            for i, landmark in enumerate(landmarks):
                msg = osc_message_builder.OscMessageBuilder(address="/hand")
                msg.add_arg(left_right)
                msg.add_arg(i)
                msg.add_arg(landmark.x)
                msg.add_arg(landmark.y)
                msg.add_arg(landmark.z)
                bundle.add_content(msg.build())
        client.send(bundle.build())


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=detection_callback,
    num_hands=2,
)

logging.info("landmarker ready")
with HandLandmarker.create_from_options(options) as landmarker:
    live_hand_detection(landmarker=landmarker)
