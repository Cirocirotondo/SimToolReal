import numpy as np

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs

import time
import datetime

import json
import argparse

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pose_estimation.transform_utils import (
    pose_to_homogeneous,
)


# TODO: where to do calibration of multiple cameras/robots
def get_board(export_image=False):
    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )
    # Generate the ChArUco board
    SQUARE_LENGTH = 0.028
    MARKER_LENGHT = 0.01265
    NUMBER_OF_SQUARES_VERTICALLY = 11
    NUMBER_OF_SQUARES_HORIZONTALLY = 8

    board = cv2.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGHT,
        dictionary=charuco_marker_dictionary,
    )

    if export_image:
        image_name = f"ChArUco_Marker_{NUMBER_OF_SQUARES_HORIZONTALLY}x{NUMBER_OF_SQUARES_VERTICALLY}.png"
        charuco_board_image = board.generateImage(
            [
                i * 100
                for i in (NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY)
            ]
        )

        cv2.imwrite(image_name, charuco_board_image)

    return board, charuco_marker_dictionary


def board_to_json(board, dict_name):
    marker_data = {"dictionary": dict_name, "markers": []}

    for i, corner in enumerate(board.getObjPoints()):
        marker_data["markers"].append(
            {
                "id": int(board.getIds()[i]),  # Marker ID
                "corners": corner.tolist(),  # Convert NumPy array to list
            }
        )

    return marker_data


def get_larger_board(export_image=False):
    charuco_marker_dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )
    # Generate the ChArUco board
    SQUARE_LENGTH = 0.0617
    MARKER_LENGHT = 0.04216
    NUMBER_OF_SQUARES_VERTICALLY = 5
    NUMBER_OF_SQUARES_HORIZONTALLY = 4

    board = cv2.aruco.CharucoBoard(
        size=(NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY),
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGHT,
        dictionary=charuco_marker_dictionary,
    )

    if export_image:
        image_name = f"ChArUco_Marker_{NUMBER_OF_SQUARES_HORIZONTALLY}x{NUMBER_OF_SQUARES_VERTICALLY}.png"
        charuco_board_image = board.generateImage(
            [
                i * 100
                for i in (NUMBER_OF_SQUARES_HORIZONTALLY, NUMBER_OF_SQUARES_VERTICALLY)
            ]
        )

        cv2.imwrite(image_name, charuco_board_image)

        serialized_board = board_to_json(board, "DICT_4X4_250")

        # Save to JSON
        with open("calibration/calibration_board_large.json", "w") as f:
            json.dump(serialized_board, f, indent=4)

    return board, charuco_marker_dictionary


def main():
    parser = argparse.ArgumentParser(description="Camera calibration with ArUco markers.")
    parser.add_argument(
        "--serial_number",
        type=str,
        help="Serial number of the RealSense camera to use.",
        required=False,
    )
    args = parser.parse_args()

    # board that we are using for calibration
    # board, charuco_marker_dictionary = get_board(True)
    board, charuco_marker_dictionary = get_larger_board(False)
    # board, charuco_marker_dictionary = get_even_larger_board(False)

    # Initialize webcam
    # cap = cv2.VideoCapture(0)

    pipeline = rs.pipeline()
    config = rs.config()

    if args.serial_number is not None:
        serial_number = args.serial_number
    else:
        realsense_ctx = rs.context()
        devices_serial_numbers = []
        for i in range(len(realsense_ctx.devices)):
            detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
            devices_serial_numbers.append(detected_camera)
            print(detected_camera)

        if len(devices_serial_numbers) > 1:
            print(f"Detected {len(devices_serial_numbers)} cameras, need to specify one.")
            return 0
        else:
            serial_number = devices_serial_numbers[0]
            print(f"Only detected one camera, running calibration for {serial_number}.")

    config.enable_device(serial_number)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # color_sensor = profile.get_device().query_sensors()[0]
    # color_sensor.set_option(rs.option.enable_auto_exposure, True)
    # Get the sensor once at the beginning. (Sensor index: 1)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]

    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 15000.000)
    # sensor.set_option(rs.option.enable_auto_exposure, True)
    # sensor.set_option(rs.option.enable_auto_white_balance, True)
    # sensor.set_option(rs.option.sharpness, 100)

    # Get camera intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    # Create camera matrix from intrinsics
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Get distortion coefficients
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

    # capture a couple of images, and get the position of a
    # fixed board.
    estimate_pose = False

    print("Press 'c' to use the current frame for calibration.")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        # Capture frame
        # ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, charuco_marker_dictionary)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray, board
            )
            if retval > 0:
                cv2.drawChessboardCorners(frame, (5, 4), charuco_corners, True)

            if retval > 4:  # Need at least 4 corners
                rvec = np.zeros((3, 1), dtype=np.float32)
                tvec = np.zeros((3, 1), dtype=np.float32)

                success = aruco.estimatePoseCharucoBoard(
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids,
                    board=board,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                )

                if success:
                    cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                    )

        cv2.imshow("ArUco Markers", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and ids is not None and retval > 0:
            estimate_pose = True
            break

        # Break loop with 'q' key
        if key == ord("q"):
            break

        # Rate limiting
        time.sleep(0.01)

    # compute the relative pose of the camera to the board.
    if estimate_pose:
        cv2.drawFrameAxes(
            frame,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            0.1,
        )

        # Aruco boards have the z-axis pointing in a weird direction.
        # We rotate it to make the z positive.
        R_x_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Rotation matrix for 90 degrees around the z-axis
        R_z_90 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # Combined transformation (first rotate around x, then around z)
        T_board_to_world = R_z_90 @ R_x_180

        orig_rvec = rvec
        orig_tvec = tvec

        hom_board_in_camera_frame = pose_to_homogeneous(orig_rvec, orig_tvec)
        hom_cam_pose_in_world_frame = T_board_to_world @ np.linalg.inv(
            hom_board_in_camera_frame
        )

        print("Board pose in camera frame.")
        print(hom_board_in_camera_frame)

        print("Camera pose in world frame.")
        print(hom_cam_pose_in_world_frame)

        print("Press 'enter' to close and export the calibration.")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure the calibration folder exists
        if not os.path.exists("calibration"):
            os.makedirs("calibration")

        np.save(f"calibration/{serial_number}_{timestamp}_homogenous_transform.npy", hom_cam_pose_in_world_frame)
        np.save(f"calibration/{serial_number}_homogenous_transform.npy", hom_cam_pose_in_world_frame)

        print("Calibration successful. Camera parameters saved.")

        cv2.imshow("ArUco Markers", frame)
        cv2.waitKey(0)  # Ensure the window stays open
        cv2.destroyAllWindows()

    # we assume knowledge of the board pose wrt. robot base


if __name__ == "__main__":
    main()
