import cv2
import numpy as np
import os

class StaticCourtHeatmap:
    def __init__(self, court_image_path='court.png', target_height=800):
        self.target_height = target_height
        self.target_width = int(target_height * 0.4552238806)
        
        #ratio
        if os.path.exists(court_image_path):
            self.court_bg = cv2.imread(court_image_path)
            self.court_bg = cv2.resize(self.court_bg, (self.target_width, self.target_height))
        else:
            print(f"[!] Không tìm thấy '{court_image_path}', sẽ dùng nền đen thay thế.")
            self.court_bg = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

       #matrix lưu cộng dồn
        self.accumulation_matrix = np.zeros((self.target_height, self.target_width), dtype=np.float32)

    def add_point_scaled(self, point_2d, src_width, src_height):
        if point_2d is not None:
            x_src, y_src = point_2d
            
            # Nội suy tọa độ sang không gian đồ họa tĩnh
            x_mapped = int(x_src * (self.target_width / src_width))
            y_mapped = int(y_src * (self.target_height / src_height))
            
            if 0 <= x_mapped < self.target_width and 0 <= y_mapped < self.target_height:
                self.accumulation_matrix[y_mapped, x_mapped] += 1.0

    def generate_heatmap(self):
        #gaussianize heatpoints
        blurred = cv2.GaussianBlur(self.accumulation_matrix, (91, 91), 0)

        normalized = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        color_heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Để làm heatmap "mờ" đi (trong suốt), chúng ta cần trộn nó với nền sân.
        # Trọng số alpha (cho heatmap) sẽ quyết định độ trong suốt, giá trị càng nhỏ càng mờ.
        alpha = 0.8 # Độ trong suốt của heatmap, bạn có thể điều chỉnh giá trị này (ví dụ: 0.5, 0.6)
        blended = cv2.addWeighted(self.court_bg, 1 - alpha, color_heatmap, alpha, 0)

        # Chúng ta chỉ áp dụng hiệu ứng trộn màu ở những vùng "nóng" để tránh làm cả sân bị ám màu.
        _, mask = cv2.threshold(normalized, 5, 255, cv2.THRESH_BINARY)
        final_out = np.where(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) > 0, blended, self.court_bg)
        return final_out