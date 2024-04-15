import cv2


class Camera:
    def __init__(self, device_index=0):
        """初始化摄像头
        Args:
            device_index (int, optional): 摄像头设备的索引号，默认为 0.
        """
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise Exception("Camera could not be opened")

        # 检查分辨率是否设置成功
        print("分辨率: ", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def capture_frame(self):
        """捕获一帧图像
        Returns:
            ndarray: 返回捕获的图像帧
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from camera")
        return frame

    def release(self):
        """释放摄像头资源"""
        self.cap.release()

    def __del__(self):
        """确保摄像头资源被正确释放"""
        self.release()

# Usage
if __name__ == "__main__":
    camera = Camera()
    frame = camera.capture_frame()
    cv2.imshow("Camera", frame)
    cv2.waitKey(0)
    camera.release()
