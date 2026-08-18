import math
import numpy as np
from scipy.interpolate import CubicSpline

class SplineExpert:
    def __init__(self):
        self.MAX_STEERING_ANGLE_DEG = 30.0
        self.previous_steering_angle = 0.0 

    def calculate_action(self, raw_lidar_array):
        """
        1000개의 라이다 원본 데이터를 받아 (조향각 정규화 값, 스로틀) 튜플을 반환합니다.
        """
        # 데이터가 비어있으면 안전하게 직진(0.0)과 기본 스로틀 반환
        if raw_lidar_array is None or len(raw_lidar_array) == 0:
            return 0.0

        # 1. 라이다 데이터 필터링 (0 초과 0.6m 이하만 유효)
        lidar_data = [r if math.isfinite(r) and 0.0 < r <= 0.6 else 0.0 for r in raw_lidar_array]

        # 2. 직교좌표(x, y) 변환
        x_coords, y_coords = self.get_lidar_data_as_xy(lidar_data)
        
        # 3. 좌우 점 분할 (인덱스 범위 고정)
        left_points = [(x_coords[i], y_coords[i]) for i in range(100, 400) if lidar_data[i] > 0]
        right_points = [(x_coords[i], y_coords[i]) for i in range(600, 900) if lidar_data[i] > 0]

        # 유효한 점이 없으면 이전 조향각 유지 혹은 직진
        if not left_points and not right_points:
            return 0.0
        
        # 4. 클러스터링 및 병합
        left_points_agg = self.aggregate_nearby_points(left_points) if left_points else []
        right_points_agg = self.aggregate_nearby_points(right_points) if right_points else []
        
        # 5. 스플라인 곡선 생성
        left_spline = self.create_spline_curve(left_points_agg)
        right_spline = self.create_spline_curve(right_points_agg)
        
        # 6. 중심 곡선 계산
        center_spline = self.calculate_center_spline(left_spline, right_spline)

        # 7. 최종 조향각 계산 및 정규화 (-1.0 ~ 1.0)
        steering_angle_deg = self._compute_angle_from_spline(center_spline)
        normalized_steering = self._normalize_steering(steering_angle_deg)
        
        return -normalized_steering

    def _normalize_steering(self, angle_deg):
        ratio = angle_deg / self.MAX_STEERING_ANGLE_DEG
        return max(-1.0, min(1.0, ratio))

    def aggregate_nearby_points(self, points, distance_threshold=0.05):
        if not points:
            return []
        
        clusters, current_cluster = [], [points[0]]
        
        for i in range(1, len(points)):
            dist = math.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
            if dist > distance_threshold:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = [points[i]]
            else:
                current_cluster.append(points[i])
                
        if current_cluster:
            clusters.append(current_cluster)
            
        return [
            (sum(p[0] for p in c) / len(c), sum(p[1] for p in c) / len(c)) if len(c) > 1 else c[0]
            for c in clusters
        ]

    def get_lidar_data_as_xy(self, lidar_data):
        angle_step = 360.0 / len(lidar_data)
        angles = np.radians(np.arange(len(lidar_data)) * angle_step)
        dists = np.array(lidar_data)
        return (dists * np.cos(angles)).tolist(), (dists * np.sin(angles)).tolist()

    def create_spline_curve(self, points):
        if not points or len(points) < 2:
            return None 
        
        x_points = np.array([p[0] for p in points])
        y_points = np.array([p[1] for p in points])
        
        sorted_indices = np.argsort(x_points)
        x_points, y_points = x_points[sorted_indices], y_points[sorted_indices]
        
        if len(np.unique(x_points)) < 2:
             return None

        return CubicSpline(x_points, y_points)

    def calculate_center_spline(self, left_spline, right_spline):
        if left_spline is None or right_spline is None:
            return None 

        x_start = max(left_spline.x[0], right_spline.x[0])
        x_end = min(left_spline.x[-1], right_spline.x[-1])
        
        if x_end <= x_start: 
            return None
        
        x_mid = np.linspace(x_start, x_end, 50)
        
        try:
            y_center = (left_spline(x_mid) + right_spline(x_mid)) / 2
        except ValueError:
            return None

        center_points = list(zip(x_mid, y_center))
        if len(center_points) < 2:
            return None
            
        return self.create_spline_curve(center_points)
    
    def _compute_angle_from_spline(self, center_spline):
        if center_spline is None:
            return self.previous_steering_angle

        x_points = center_spline.x
        y_points = center_spline(x_points) 

        if len(x_points) == 0:
            return self.previous_steering_angle
        
        target_angle_rad = math.atan2(y_points[-1], x_points[-1])
        final_steering_angle = math.degrees(target_angle_rad)
        
        final_steering_angle = max(-self.MAX_STEERING_ANGLE_DEG, min(self.MAX_STEERING_ANGLE_DEG, final_steering_angle))
        self.previous_steering_angle = final_steering_angle
            
        return final_steering_angle