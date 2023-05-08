import logging
import pickle as pk
from time import time

import cv2
import mediapipe as mp
import numpy as np
from absl import app, flags
from pythonosc import osc_bundle_builder, osc_message_builder, udp_client

logging.basicConfig(level=logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_integer("device", default=0, help="camera to use")
flags.DEFINE_string("address", default="127.0.0.1", help="ip to send osc to")
flags.DEFINE_integer("port", default=1893, help="port to send osc to")
flags.DEFINE_bool("show", default=False, help="show camera output")


def center_hand(hand):
    """centers, normalize and reproject hand on a 
    position invariant orthogonal basis"""
    
    center = hand[0]
    thumb_base = hand[2]
    major_base = hand[9]
    major_tip = hand[12]

    ax1 = thumb_base - center
    ax2 = major_base - center
    ax3 = major_tip - center

    scale = np.linalg.norm(ax2)

    # normalize axes
    ax1 = ax1 / np.linalg.norm(ax1, ord=2)
    ax2 = ax2 / np.linalg.norm(ax2, ord=2)
    ax3 = ax3 / np.linalg.norm(ax3, ord=2)

    # orthogonalize ax1 w.r.t ax2
    ax1 = ax1 - np.dot(ax1, ax2) * ax2
    ax1 = ax1 / np.linalg.norm(ax1, ord=2)

    # orthogonalize ax3 w.r.t ax1 and ax2
    ax3 = ax3 - np.dot(ax1, ax3) * ax1
    ax3 = ax3 / np.linalg.norm(ax3, ord=2)
    ax3 = ax3 - np.dot(ax2, ax3) * ax2
    ax3 = ax3 / np.linalg.norm(ax3, ord=2)

    # create basis
    basis = np.stack([ax1, ax2, ax3], 0)

    # project and scale hand
    new_hand = np.einsum("oi,li->lo", basis, hand - center)
    new_hand =  new_hand / scale

    return new_hand



def main(argv):
    model_path = 'hand_landmarker.task'

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    client = udp_client.SimpleUDPClient(FLAGS.address, port=FLAGS.port)

    def get_ms_counter():
        start = time()

        def counter():
            ellapsed = time() - start
            return int(1000 * ellapsed)

        return counter

    def live_hand_detection(landmarker: HandLandmarker):
        cam = cv2.VideoCapture(FLAGS.device)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        counter = get_ms_counter()

        logging.info("starting detection")
        while True:
            _, img = cam.read()
            img = cv2.flip(img, 1)
            if FLAGS.show:
                cv2.imshow("frame", img)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
            landmarker.detect_async(mp_image, counter())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def detection_callback(result: HandLandmarkerResult,
                           output_image: mp.Image, timestamp_ms: int):
        if result.hand_landmarks:
            bundle = osc_bundle_builder.OscBundleBuilder(
                osc_bundle_builder.IMMEDIATELY)
            for hand, landmarks in zip(result.handedness,
                                       result.hand_landmarks):
                left_right = hand[0].category_name.lower()
                hand = list(map(lambda x: [x.x, x.y, x.z], landmarks))
                hand = np.asarray(hand)
                hand = center_hand(hand)
                for i, landmark in enumerate(hand):
                    msg = osc_message_builder.OscMessageBuilder(
                        address="/hand")
                    msg.add_arg(left_right)
                    msg.add_arg(i)
                    msg.add_arg(landmark[0])
                    msg.add_arg(landmark[1])
                    msg.add_arg(landmark[2])
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


if __name__ == "__main__":
    app.run(main)
