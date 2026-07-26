import numpy as np

import cv2
import cv2.aruco as aruco

import pyrealsense2 as rs

import zmq
import copy

import json

import sys
import os
import time
import datetime

import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pose_estimation.camera_calibration import get_larger_board
from pose_estimation.transform_utils import (
    pose_to_homogeneous,
    seven_d_to_homogeneous,
    scalar_last_to_scalar_first,
)

from scipy.spatial.transform import Rotation as R


ADAPTER_TO_EE_POSE = np.array([
    [1.0, 0.0,  0.0, 0.046],
    [0.0, 0.0, -1.0, 0.046],
    [0.0, 1.0,  0.0, 0.050],
    [0.0, 0.0,  0.0, 1.000],
])

DEFAULT_CAMERA_CONFIG_PATH = (
    "/home/duplo/git/robohand/src/tag-pose-estimation/config/"
    "camera_config/ur5_realsense_config.json"
)


def average_pose_estimates(transforms):
    # Extract translations and rotations
    translations = np.array([T[:3, 3] for T in transforms])
    rotations = np.array(
        [R.from_matrix(T[:3, :3]).as_quat() for T in transforms]
    )  # (x, y, z, w)

    # Average translation
    avg_translation = np.mean(translations, axis=0)

    # Quaternions q and -q represent the same rotation. Align their signs
    # before averaging to avoid cancellation.
    reference_quaternion = rotations[0]
    rotations = np.array([
        quaternion
        if np.dot(reference_quaternion, quaternion) >= 0
        else -quaternion
        for quaternion in rotations
    ])

    # Average quaternion (normalize sum of quaternions)
    avg_quat = np.mean(rotations, axis=0)
    avg_quat /= np.linalg.norm(avg_quat)

    # Convert back to rotation matrix
    avg_rotation = R.from_quat(avg_quat).as_matrix()

    # Assemble averaged transform
    avg_transform = np.eye(4)
    avg_transform[:3, :3] = avg_rotation
    avg_transform[:3, 3] = avg_translation

    return avg_transform


def compute_pose_dispersion(transforms, average_transform):
    translation_errors = np.array([
        np.linalg.norm(transform[:3, 3] - average_transform[:3, 3])
        for transform in transforms
    ])

    average_rotation = R.from_matrix(average_transform[:3, :3])
    rotation_errors_deg = np.array([
        np.degrees(
            (average_rotation.inv() * R.from_matrix(transform[:3, :3])).magnitude()
        )
        for transform in transforms
    ])

    return {
        "translation_mean_m": float(np.mean(translation_errors)),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_mean_deg": float(np.mean(rotation_errors_deg)),
        "rotation_max_deg": float(np.max(rotation_errors_deg)),
    }


def update_camera_extrinsic_path(
    camera_config_path,
    camera_serial_number,
    extrinsic_calibration_path,
):
    with open(camera_config_path, "r") as config_file:
        camera_config = json.load(config_file)

    cameras = camera_config.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError(
            f"Camera config {camera_config_path} does not contain a "
            "'cameras' list."
        )

    matching_cameras = [
        camera
        for camera in cameras
        if str(camera.get("serial_number")) == str(camera_serial_number)
    ]
    if len(matching_cameras) != 1:
        raise ValueError(
            f"Expected exactly one camera with serial "
            f"{camera_serial_number} in {camera_config_path}, found "
            f"{len(matching_cameras)}."
        )

    absolute_extrinsic_path = os.path.abspath(extrinsic_calibration_path)
    matching_cameras[0]["extrinsic_calibration_file"] = (
        absolute_extrinsic_path
    )

    temporary_path = f"{camera_config_path}.tmp"
    with open(temporary_path, "w") as config_file:
        json.dump(camera_config, config_file, indent=2)
        config_file.write("\n")
    os.replace(temporary_path, camera_config_path)

    return absolute_extrinsic_path


def load_ee_calibration_board(ee_calibration_board_path):
    def load_single_aruco_board(board_config):
        with open(board_config, "r") as f:
            board_data = json.load(f)

        # load which dict from config
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, board_data["dictionary"])
        )

        # load ids and corners from config
        marker_ids_list = []
        marker_corners_list = []

        for marker in board_data["markers"]:
            marker_ids_list.append(marker["id"])
            marker_corners_list.append(marker["corners"])

        board = cv2.aruco.Board(
            objPoints=np.array(marker_corners_list, np.float32),
            dictionary=aruco_dict,
            ids=np.array(marker_ids_list),
        )

        return board

    return load_single_aruco_board(ee_calibration_board_path)


def load_world_charuco_board(world_board_path):
    with open(world_board_path, "r") as f:
        board_data = json.load(f)

    required_keys = (
        "squares_x",
        "squares_y",
        "square_length_m",
        "marker_length_m",
        "dictionary",
    )
    missing_keys = [key for key in required_keys if key not in board_data]
    if missing_keys:
        raise ValueError(
            f"World board config {world_board_path} is missing required "
            f"ChArUco fields: {', '.join(missing_keys)}"
        )

    try:
        dictionary_id = getattr(cv2.aruco, board_data["dictionary"])
    except AttributeError as exc:
        raise ValueError(
            f"Unknown ArUco dictionary {board_data['dictionary']!r} in "
            f"{world_board_path}"
        ) from exc

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        size=(int(board_data["squares_x"]), int(board_data["squares_y"])),
        squareLength=float(board_data["square_length_m"]),
        markerLength=float(board_data["marker_length_m"]),
        dictionary=dictionary,
    )

    return board, dictionary


def perturb_quaternion(q, angle_std_deg=5.0):
    """
    Apply a small random rotational perturbation to a quaternion.

    Parameters:
    - q: array-like of shape (4,) — original quaternion (x, y, z, w)
    - angle_std_deg: standard deviation of perturbation angle in degrees

    Returns:
    - perturbed quaternion as a numpy array (x, y, z, w)
    """
    # Generate small random rotation vector (axis-angle), with angle ~ N(0, angle_std_deg)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)  # normalize to get random direction
    angle_rad = np.random.normal(0, np.deg2rad(angle_std_deg))
    delta_rotvec = axis * angle_rad

    # Convert original quaternion to scipy Rotation
    r_orig = R.from_quat(q)  # scipy expects [x, y, z, w]
    r_delta = R.from_rotvec(delta_rotvec)

    # Apply perturbation
    r_perturbed = r_delta * r_orig
    return r_perturbed.as_quat()  # returns [x, y, z, w]


def main(
    name,
    zmq_ip="127.0.0.1",
    zmq_controller_port=5555,
    zmq_state_est_port=5556,
    camera_serial_number=None,
    camera_config_path=DEFAULT_CAMERA_CONFIG_PATH,
    world_board_path=None,
    ee_board_path="./calibration/robot_calibration_board.json",
):
    print("setting up robot pose estimation socket")
    robot_pose_context = zmq.Context()
    robot_pose_socket = robot_pose_context.socket(zmq.SUB)
    robot_pose_socket.setsockopt(zmq.CONFLATE, 1)  # Keep only the latest message
    robot_pose_socket.connect(f"tcp://{zmq_ip}:{zmq_state_est_port}")
    robot_pose_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("setting up robot control socket")
    controller_context = zmq.Context()
    controller_publisher = controller_context.socket(zmq.PUB)
    controller_publisher.bind(f"tcp://{zmq_ip}:{zmq_controller_port}")

    # Board that defines the world frame. Keep the historical hard-coded board
    # as the default for backwards compatibility.
    if world_board_path is None:
        board, charuco_marker_dictionary = get_larger_board(False)
        print("Using the legacy built-in world ChArUco board.")
    else:
        board, charuco_marker_dictionary = load_world_charuco_board(
            world_board_path
        )
        print(f"Using world board from {world_board_path}")

    # Board mounted on the end effector.
    ee_board = load_ee_calibration_board(ee_board_path)
    ee_marker_dictionary = ee_board.getDictionary()
    print(f"Using end-effector board from {ee_board_path}")

    # Initialize webcam
    # cap = cv2.VideoCapture(0)

    pipeline = rs.pipeline()
    config = rs.config()

    if camera_serial_number:
        config.enable_device(camera_serial_number)

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # color_sensor = profile.get_device().query_sensors()[0]
    # color_sensor.set_option(rs.option.enable_auto_exposure, True)
    # Get the sensor once at the beginning. (Sensor index: 1)

    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]

    # Set the exposure anytime during the operation
    # sensor.set_option(rs.option.exposure, 10000.000)
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

    # move robot around a bit
    # take pictures and save robot ee-pose along with it
    # compute basepose from it, and export
    robot_start_state = None

    while True:
        # update real state
        try:
            robot_start_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
            print("received robot state")
        except zmq.Again:
            pass

        if robot_start_state is not None:
            break

    robot_ee_start_pose = robot_start_state["pos"]

    # fill list with poses that we are going to do
    desired_ee_poses = []

    offset = 0.1
    dirs = [
        (offset / 2, offset / 2),
        (offset / 2, -offset / 2),
        (-offset / 2, -offset / 2),
        (-offset / 2, offset / 2),
    ]

    # square in xy
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[1] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:])
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    # square in z
    for i in range(4):
        pose = copy.deepcopy(robot_ee_start_pose)
        pose[0] += dirs[i][0]
        pose[2] += dirs[i][1]

        perturbed_quat = perturb_quaternion(pose[3:])
        pose[3:] = perturbed_quat

        desired_ee_poses.append(pose)

    robot_base_poses = []
    camera_world_poses = []

    curr_pose_idx = 0
    while True:
        if curr_pose_idx >= len(desired_ee_poses):
            break

        robot_state = None
        while True:
            command = {"target_ee_pose": list(desired_ee_poses[curr_pose_idx])}
            # print(f"Sending target pose: {command}")
            controller_publisher.send_json(command)

            try:
                robot_state = robot_pose_socket.recv_json(flags=zmq.NOBLOCK)
                # print("received robot state")
            except zmq.Again:
                pass

            if robot_state is None:
                continue

            position_error = np.linalg.norm(
                np.array(robot_state["pos"][:3])
                - np.array(desired_ee_poses[curr_pose_idx][:3])
            )
            q1_inv = R.from_quat(robot_state["pos"][3:]).inv()
            q_rel = q1_inv * R.from_quat(
                desired_ee_poses[curr_pose_idx][3:]
            )  # relative rotation from q1 to q2

            # Get angle difference in radians:
            angle_diff = q_rel.magnitude()

            if position_error < 1e-2 and angle_diff < 1e-2:
                curr_pose_idx += 1

                time.sleep(0.5)
                break
            # else:
            #     print(f"Pose diff too big: {position_error}")

            time.sleep(0.01)

        # take picture, estimate board pose, and ee-pose
        max_img_taking_attempts = 10
        for _ in range(max_img_taking_attempts):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())

            # Capture frame
            # ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            world_corners, world_ids, _ = aruco.detectMarkers(
                gray, charuco_marker_dictionary
            )

            if world_ids is not None:
                aruco.drawDetectedMarkers(frame, world_corners, world_ids)

                # cv2.imshow("Camera image", frame)

                # while True:
                #     key = cv2.waitKey(1) & 0xFF
                #     if key == ord("c"):
                #         break

                # world frame calibration
                retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                    world_corners, world_ids, gray, board
                )
                if retval > 0:
                    chessboard_size = tuple(
                        int(size) - 1 for size in board.getChessboardSize()
                    )
                    cv2.drawChessboardCorners(
                        frame, chessboard_size, charuco_corners, True
                    )

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

                        R_x_180 = np.array(
                            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                        )

                        # Rotation matrix for 90 degrees around the z-axis
                        R_z_90 = np.array(
                            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                        )

                        # Combined transformation (first rotate around x, then around z)
                        T_board_to_world = R_z_90 @ R_x_180

                        orig_rvec = rvec
                        orig_tvec = tvec

                        hom_board_in_camera_frame = pose_to_homogeneous(
                            orig_rvec, orig_tvec
                        )
                        hom_cam_pose_in_world_frame = T_board_to_world @ np.linalg.inv(
                            hom_board_in_camera_frame
                        )
                    else:
                        print("Unable to estimate the charuco board pose")
                        continue
                else:
                    print("Unable to find the charuco-corners of the world-frame-board")
                    continue

                # ee calibration
                # detect ee board
                print("Estimating ee-pose")
                ee_corners, ee_ids, _ = aruco.detectMarkers(
                    gray, ee_marker_dictionary
                )
                if ee_ids is None:
                    print("Unable to find the end-effector board markers")
                    continue
                aruco.drawDetectedMarkers(frame, ee_corners, ee_ids)

                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    ee_corners,
                    ee_ids,
                    ee_board,
                    camera_matrix,
                    dist_coeffs,
                    None,
                    None,
                )

                if retval:
                    cv2.drawFrameAxes(
                        frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1
                    )

                    ee_adapter_pose_world_frame = (
                        hom_cam_pose_in_world_frame @ pose_to_homogeneous(rvec, tvec)
                    )

                    ee_pose_world_frame = (
                        ee_adapter_pose_world_frame @ ADAPTER_TO_EE_POSE
                    )

                    # robot_state[pos] is [pos][quat], where quat is xyzw
                    tmp = scalar_last_to_scalar_first(robot_state["pos"])

                    # tmp = copy.deepcopy(robot_state["pos"])
                    # tmp[3] = robot_state["pos"][6]
                    # tmp[4] = robot_state["pos"][3]
                    # tmp[5] = robot_state["pos"][4]
                    # tmp[6] = robot_state["pos"][5]
                    ee_pose_robot_frame = seven_d_to_homogeneous(np.array(tmp))

                    robot_base_pose = ee_pose_world_frame @ np.linalg.inv(
                        ee_pose_robot_frame
                    )

                    cv2.imshow("Camera image", frame)

                    use_this_estimate = False
                    print("Press 'y' if this is acceptable, 'n' otherwise.")
                    while True:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("y"):
                            use_this_estimate = True
                            break

                        if key == ord("n"):
                            break

                    if use_this_estimate:
                        robot_base_poses.append(robot_base_pose)
                        camera_world_poses.append(hom_cam_pose_in_world_frame)

                    break
                else:
                    print("Unable to locate end effector.")

            else:
                print("Was not able to identify corners.")

    command = {"target_ee_pose": list(robot_ee_start_pose)}
    # print(f"Sending target pose: {command}")
    controller_publisher.send_json(command)

    print("Computed base pose:")
    if len(robot_base_poses) > 1:
        # Average robot-base and camera poses from the same accepted
        # observations so both outputs share exactly the same world frame.
        estimated_base_pose = average_pose_estimates(robot_base_poses)
        estimated_camera_pose = average_pose_estimates(camera_world_poses)

        base_dispersion = compute_pose_dispersion(
            robot_base_poses, estimated_base_pose
        )
        camera_dispersion = compute_pose_dispersion(
            camera_world_poses, estimated_camera_pose
        )

        print(estimated_base_pose)
        print("Robot-base pose dispersion:", base_dispersion)
        print("Camera extrinsic pose dispersion:", camera_dispersion)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # we export this twice, once as a timed version to have access to a specific version,
        # and secondly to a generic version, which we generally use in our files.
        output_filepath = f"./calibration/base_pose_robot_{name}.npy"
        timed_output_filepath = f"./calibration/{timestamp}_base_pose_robot_{name}.npy"

        camera_id = camera_serial_number or "camera"
        camera_output_filepath = (
            f"./calibration/camera_extrinsic_{camera_id}.npy"
        )
        timed_camera_output_filepath = (
            f"./calibration/{timestamp}_camera_extrinsic_{camera_id}.npy"
        )

        np.save(output_filepath, estimated_base_pose)
        np.save(timed_output_filepath, estimated_base_pose)
        np.save(camera_output_filepath, estimated_camera_pose)
        np.save(timed_camera_output_filepath, estimated_camera_pose)

        updated_extrinsic_path = update_camera_extrinsic_path(
            camera_config_path,
            camera_serial_number,
            camera_output_filepath,
        )

        metadata_filepath = f"./calibration/{timestamp}_calibration_summary.txt"
        with open(metadata_filepath, "w") as metadata_file:
            metadata_file.write(
                f"Accepted observations: {len(robot_base_poses)}\n\n"
            )
            metadata_file.write("T_world_robot_base:\n")
            metadata_file.write(np.array2string(estimated_base_pose))
            metadata_file.write("\n\nRobot-base pose dispersion:\n")
            metadata_file.write(f"{base_dispersion}\n\n")
            metadata_file.write("T_world_camera:\n")
            metadata_file.write(np.array2string(estimated_camera_pose))
            metadata_file.write("\n\nCamera extrinsic pose dispersion:\n")
            metadata_file.write(f"{camera_dispersion}\n")

        print(
            f"Saved robot-base calibration to {output_filepath} and "
            f"{timed_output_filepath}"
        )
        print(
            f"Saved camera extrinsic calibration to {camera_output_filepath} "
            f"and {timed_camera_output_filepath}"
        )
        print(
            f"Updated extrinsic_calibration_file in {camera_config_path} to "
            f"{updated_extrinsic_path}"
        )
        print(f"Saved calibration summary to {metadata_filepath}")
    else:
        print("Did not find a sufficient number of observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robot base calibration with ArUco markers."
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Name of the robot under which the calibration should be saved.",
        required=True,
    )
    parser.add_argument(
        "-s", "--serial_number",
        type=str,
        help="Serial number of the camera that should be used for calibration.",
        required=True,
    )
    # parser.add_argument(
    #     "--export_camera_pose",
    #     type=bool,
    #     default=False,
    #     help="Robot config path.",
    #     required=True,
    # )
    parser.add_argument(
        "-r", "--robot_config_path",
        type=str,
        help="Robot config path.",
        required=True,
    )
    parser.add_argument(
        "--camera_config_path",
        type=str,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help=(
            "Camera JSON to update with the generated extrinsic calibration "
            f"path. Defaults to {DEFAULT_CAMERA_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--world_board_path",
        type=str,
        default=None,
        help=(
            "Path to a ChArUco world-board JSON containing squares_x, "
            "squares_y, square_length_m, marker_length_m, and dictionary. "
            "If omitted, the legacy built-in 4x5 board is used."
        ),
    )
    parser.add_argument(
        "--ee_board_path",
        type=str,
        default="./calibration/robot_calibration_board.json",
        help=(
            "Path to the ArUco board JSON mounted on the end effector. "
            "Defaults to ./calibration/robot_calibration_board.json."
        ),
    )
    args = parser.parse_args()

    try:
        with open(args.robot_config_path) as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Config file not found.", file=sys.stderr)
        sys.exit(1)

    camera_serial_number = args.serial_number

    main(
        args.name,
        zmq_controller_port=config["socket_port"],
        zmq_state_est_port=config["publisher_port"],
        camera_serial_number=camera_serial_number,
        camera_config_path=args.camera_config_path,
        world_board_path=args.world_board_path,
        ee_board_path=args.ee_board_path,
    )
