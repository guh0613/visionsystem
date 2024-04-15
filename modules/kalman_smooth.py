import numpy as np


class KalmanFilter:
    def __init__(self, state_dim, measure_dim, control_dim=0):
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.control_dim = control_dim

        # 状态向量和协方差矩阵
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)

        # 系统噪声和测量噪声
        self.process_noise = np.eye(state_dim) * 0.01
        self.measurement_noise = np.eye(measure_dim) * 0.1

        # 状态转移矩阵和测量矩阵
        self.state_transition = np.eye(state_dim)
        self.measurement_matrix = np.random.random((measure_dim, state_dim))

        # 控制矩阵（如果有控制输入的话）
        if control_dim > 0:
            self.control_matrix = np.random.random((state_dim, control_dim))
        else:
            self.control_matrix = None

        # 用于平滑的变量
        self.states = []
        self.covariances = []

    def predict(self, control_input=None):
        # 预测状态
        if self.control_matrix is not None and control_input is not None:
            self.state = self.state_transition @ self.state + self.control_matrix @ control_input
        else:
            self.state = self.state_transition @ self.state

        # 预测协方差
        self.covariance = self.state_transition @ self.covariance @ self.state_transition.T + self.process_noise
        self.states.append(self.state.copy())
        self.covariances.append(self.covariance.copy())

    def update(self, measurement):
        measurement = np.reshape(measurement, (self.measure_dim, 1))
        # 卡尔曼增益
        K = self.covariance @ self.measurement_matrix.T @ np.linalg.inv(
            self.measurement_matrix @ self.covariance @ self.measurement_matrix.T + self.measurement_noise)

        # 更新状态
        self.state = self.state + K @ (measurement - self.measurement_matrix @ self.state)

        # 更新协方差
        self.covariance = (np.eye(self.state_dim) - K @ self.measurement_matrix) @ self.covariance

    def smooth(self):
        n = len(self.states)
        smoothed_states = [self.states[-1]]
        smoothed_covariances = [self.covariances[-1]]

        for i in range(n - 2, -1, -1):
            # 计算平滑增益
            P_k = self.covariances[i]
            P_kplus1 = self.covariances[i + 1]
            F_k = self.state_transition
            K_k = P_k @ F_k.T @ np.linalg.inv(P_kplus1)

            # 平滑状态
            x_kplus1_smoothed = smoothed_states[0]
            x_k_smoothed = self.states[i] + K_k @ (x_kplus1_smoothed - F_k @ self.states[i])

            # 平滑协方差
            P_k_smoothed = P_k + K_k @ (smoothed_covariances[0] - P_kplus1) @ K_k.T

            # 将平滑的结果插入列表前部
            smoothed_states.insert(0, x_k_smoothed)
            smoothed_covariances.insert(0, P_k_smoothed)

        return smoothed_states

