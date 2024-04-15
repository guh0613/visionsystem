import cv2
import numpy as np
from modules.aruco_module import ArUcoDetector
from modules.camera import Camera
from modules.kalman_smooth import KalmanFilter


class VisionSystem:
    def __init__(self, camera_params, marker_length):
        self.camera_matrix, self.dist_coeffs = camera_params
        self.marker_length = marker_length
        self.detector = ArUcoDetector(cv2.aruco.DICT_6X6_250, self.camera_matrix, self.dist_coeffs, self.marker_length)
        self.kalman_filter = KalmanFilter(state_dim=6, measure_dim=6)
        self.camera = Camera()

    def capture_and_process(self):
        """捕获一张图像，识别aruco码，获取坐标."""
        frame = self.camera.capture_frame()
        corners, ids, rejected = self.detector.detect_markers(frame)
        world_positions = []
        if ids is not None:
            rvecs, tvecs = self.detector.estimate_pose(corners, ids)
            self.detector.update_world_origin()
            for rvec, tvec in zip(rvecs, tvecs):
                measure = np.vstack((rvec, tvec))
                self.kalman_filter.predict()
                self.kalman_filter.update(measure)
                pose = self.detector.get_marker_world_position(rvec, tvec)
                world_positions.append(pose)
        smoothed_states = self.kalman_filter.smooth()
        return world_positions, smoothed_states, frame, corners, tvecs, rvecs

    def set_world_origin(self, image, origin_id):
        """Set a specific marker as the origin of the world coordinate system."""
        self.detector.set_origin_by_marker(image, origin_id)

    def get_axis(self, image, rvec, tvec, corners):
        """绘制整个坐标轴,并标出原点，各个marker的位置"""
        self.detector.draw_markers(image, corners, rvec, tvec)
        return image


# 调用一下
if __name__ == "__main__":
    # 摄像头参数，手头没摄像头，瞎写一个
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])
    dist_coeffs = np.zeros((4, 1))
    system = VisionSystem((camera_matrix, dist_coeffs), marker_length=1)

    positions, states, image, corners, tvecs, rvecs = system.capture_and_process()
    print("Detected positions:", positions)
    print("Smoothed states:", states)

    # 生成一张在原有的捕获的图像上绘制好坐标系的新图像，标注出原点以及标注marker的位置
    image = system.get_axis(image, rvecs, tvecs, corners)
    cv2.imshow('识别结果', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







