import cv2
import cv2.aruco as aruco
import numpy as np


class ArUcoDetector:
    def __init__(self, dictionary_name, camera_matrix, dist_coeffs, marker_length):
        """ 初始化 ArUco 检测器
        Args:
            dictionary_name (int): 使用的 ArUco 字典
            camera_matrix (numpy.ndarray): 相机内参矩阵
            dist_coeffs (numpy.ndarray): 相机畸变系数
            marker_length (float): ArUco 标记的物理尺寸
        """
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary_name)
        self.parameters = aruco.DetectorParameters()
        # 调整参数以提高识别率
        self.parameters.adaptiveThreshConstant = 7
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 10
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length
        self.transformation_matrix_to_world = np.eye(4)  # 初始化为单位矩阵
        self.all_tvecs = []

    # 预处理，但目前发现一预处理就识别不出来了，先暂时不动了
    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return gray

    def detect_markers(self, image):
        """ 检测图像中的所有 ArUco 标记
        Args:
            image (numpy.ndarray): 预处理后的图像
        Returns:
            tuple: 返回角点、标记ID和拒绝的标记
        """
        processed_image = image
        corners, ids, rejected = aruco.detectMarkers(processed_image, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            print(f"Detected markers: {ids.flatten()}")


        return corners, ids, rejected

    def estimate_pose(self, corners, ids):
        """ 估计标记的姿态
        Args:
            corners (list): 标记的角点
            ids (numpy.ndarray): 标记的ID
        Returns:
            tuple: 返回旋转向量和平移向量
        """
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix,
                                                              self.dist_coeffs)
            self.all_tvecs.extend(tvecs)
            return rvecs, tvecs
        return None, None

    def transform_to_world_coordinates(self, rvecs, tvecs):
        """ 将检测到的标记转换为世界坐标系
        Args:
            rvecs (numpy.ndarray): 旋转向量
            tvecs (numpy.ndarray): 平移向量
        Returns:
            list: 返回标记在世界坐标系中的位置和姿态
        """
        world_poses = []
        for rvec, tvec in zip(rvecs, tvecs):
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = tvec.flatten()
            world_pose = np.dot(self.transformation_matrix_to_world, transform_matrix)
            world_poses.append(world_pose)
        return world_poses

    def set_origin_by_marker(self, image, origin_id):
        """ 设置坐标系原点为指定ID的标记
        Args:
            image (numpy.ndarray): 输入图像
            origin_id (int): 作为原点的标记ID
        """
        corners, ids = self.detect_markers(image)
        rvecs, tvecs = self.estimate_pose(corners, ids)
        if ids is not None:
            for id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
                if id == origin_id:
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    transform_matrix = np.eye(4)
                    transform_matrix[:3, :3] = rotation_matrix
                    transform_matrix[:3, 3] = tvec.flatten()
                    # 计算逆变换矩阵，用于将该标记设置为坐标原点
                    self.transformation_matrix_to_world = np.linalg.inv(transform_matrix)
                    break

    def get_marker_world_position(self, rvec, tvec):
        """计算单个标记在世界坐标系中的位置
        Args:
            rvec (numpy.ndarray): 标记的旋转向量
            tvec (numpy.ndarray): 标记的平移向量
        Returns:
            numpy.ndarray: 标记在世界坐标系中的位置
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = tvec.flatten()
        world_position = np.dot(self.transformation_matrix_to_world, transform_matrix)
        return world_position[:3, 3]

    def update_world_origin(self):
        """ 更新世界坐标系原点为所有标记的平均位置 """
        if self.all_tvecs:
            average_tvec = np.mean(np.array(self.all_tvecs), axis=0).flatten()
            self.transformation_matrix_to_world[:3, 3] = -average_tvec
            self.all_tvecs = []

    def draw_markers(self, image, corners, rvecs, tvecs):
        """ 在图像上绘制标记的边界、中心点和标记的世界坐标 """
        for i, corner in enumerate(corners):
            # 绘制多边形轮廓
            int_corners = np.int32(corner)
            cv2.polylines(image, [int_corners], isClosed=True, color=(0, 255, 0), thickness=2)

            # 计算标记的中心点并确保是整数类型的元组
            center = tuple(np.int32(np.mean(corner, axis=1)).ravel())
            cv2.circle(image, center, 5, (0, 0, 255), -1)  # 在中心位置绘制红色的点

            # 计算标记的世界坐标
            world_position = self.get_marker_world_position(rvecs[i], tvecs[i])
            world_pos_str = f"({world_position[0]:.2f}, {world_position[1]:.2f}, {world_position[2]:.2f})"

            # 在标记的中心下方添加坐标文本
            cv2.putText(image, world_pos_str, (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 处理相机矩阵以确保其为浮点类型
            camera_matrix_float = np.array(self.camera_matrix, dtype=np.float64)

            origin_world_position = np.array([0, 0, 0], dtype=np.float32).reshape(1, 1, 3)
            origin_world_position = self.get_marker_world_position(np.zeros((3, 1)), origin_world_position[0])
            origin_image_point, _ = cv2.projectPoints(origin_world_position,
                                                      np.zeros((3, 1)), np.zeros((3, 1)),
                                                      camera_matrix_float, self.dist_coeffs)
            origin_image_point = tuple(np.int32(origin_image_point[0, 0, :]))
            cv2.circle(image, origin_image_point, 5, (255, 255, 0), -1)  # 在原点位置绘制蓝色点
            cv2.putText(image, "(0,0,0)", (origin_image_point[0] + 10, origin_image_point[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)


