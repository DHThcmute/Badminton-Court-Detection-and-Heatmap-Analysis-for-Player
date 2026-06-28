import cv2
import numpy as np

class CourtFlattener:
    def __init__(self, target_height=480):
        self.target_height = target_height
        self.target_width = int(target_height * 0.4552238806)
        
        self.dst_pts = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype=np.float32)
        
        # Thêm biến lưu giữ ma trận phối cảnh
        self.current_matrix = None

    def flatten(self, frame, court_corners):
        if court_corners is None:
            self.current_matrix = None
            return np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        
        src_pts = court_corners.reshape(4, 2).astype(np.float32)
        self.current_matrix = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        
        warped = cv2.warpPerspective(frame, self.current_matrix, (self.target_width, self.target_height))
        return warped

    def map_point(self, point_2d):
        """Đưa tọa độ (x,y) từ ảnh gốc sang không gian phẳng 2D"""
        if self.current_matrix is None or point_2d is None:
            return None
        
        # Định dạng lại ma trận điểm để OpenCV có thể tính toán
        pts = np.array([[[point_2d[0], point_2d[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.current_matrix)
        
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))