import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image

class AIFeetTracker:
    """
    Một lớp để theo dõi vị trí bàn chân trong video cầu lông bằng mô hình ResNet18.

    Attributes:
        device (torch.device): Thiết bị (CPU hoặc GPU) để chạy mô hình.
        model (torch.nn.Module): Mô hình PyTorch đã được huấn luyện.
        transform (transforms.Compose): Chuỗi các phép biến đổi ảnh đầu vào.
        target_size (tuple[int, int]): Kích thước mục tiêu để resize ảnh đầu vào.
    """
    def __init__(self, checkpoint_path: str, target_size: tuple[int, int] = (416, 416)):
        """
        Khởi tạo AIFeetTracker.

        Args:
            checkpoint_path (str): Đường dẫn đến file checkpoint của mô hình.
            target_size (tuple[int, int], optional): Kích thước ảnh đầu vào cho mô hình. Mặc định là (416, 416).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size

        # Khởi tạo mô hình
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2) # Output là 2 giá trị (x, y)
        self.model = self.model.to(self.device)

        # Tải trọng số đã huấn luyện
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval() # Chuyển mô hình sang chế độ đánh giá (inference)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Các giá trị mean/std này là tiêu chuẩn cho ImageNet.
            # Cần đảm bảo rằng quá trình training cũng sử dụng chúng.
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Tiền xử lý một frame trước khi đưa vào mô hình."""
        # Tiền xử lý giống hệt lúc Train
        # Lưu ý: Việc chuyển sang ảnh xám rồi lại convert sang RGB có thể không tối ưu.
        # Nếu mô hình được train trên ảnh xám, nên sử dụng 1 kênh đầu vào và chuẩn hóa khác.
        # Nếu mô hình được train trên ảnh màu, không nên chuyển sang ảnh xám ở đây.
        processed = cv2.resize(frame, self.target_size)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

        # Chuyển từ ảnh xám (1 kênh) sang ảnh RGB (3 kênh) bằng cách lặp lại kênh
        pil_image = Image.fromarray(processed).convert("RGB")
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return input_tensor

    def predict(self, frame: np.ndarray) -> tuple[int, int]:
        """
        Dự đoán tọa độ bàn chân từ một frame ảnh.

        Args:
            frame (np.ndarray): Frame ảnh đầu vào từ video (định dạng BGR của OpenCV).

        Returns:
            tuple[int, int]: Tọa độ (x, y) dự đoán trên ảnh gốc.
        """
        # Lấy kích thước thật để scale ngược
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / self.target_size[0]
        scale_y = orig_h / self.target_size[1]

        input_tensor = self._preprocess_frame(frame)

        # Thực hiện dự đoán
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Scale tọa độ dự đoán về kích thước ảnh gốc
        pred_x_scaled = outputs[0, 0].item()
        pred_y_scaled = outputs[0, 1].item()
        real_x = int(pred_x_scaled * scale_x)
        real_y = int(pred_y_scaled * scale_y)

        return (real_x, real_y)