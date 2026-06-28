import cv2
import numpy as np

def order_points(pts):

    # Reshape from (4, 1, 2) to (4, 2)
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point has the smallest sum (x+y)
    # The bottom-right point has the largest sum (x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point has the smallest difference (y-x)
    # The bottom-left has the largest difference (y-x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect.reshape(4, 1, 2).astype(np.int32)

class CourtDetector:
    def __init__(self):
        #is stable?
        self.last_good_white_corners = None
        self.last_good_green_corners = None
        self.is_roi_locked = False
        self.locked_roi_display_corners = None # For drawing the final locked ROI
        #delta dich chuyen d/dt
        self.MAX_CORNER_SHIFT = 50.0

    def _detect_court_boundary(self, frame, white_detection_roi_mask=None):

        output_frame = frame.copy()
        green_court_corners = None
        roi_for_white_detection = None

        if white_detection_roi_mask is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            close_kernel = np.ones((15, 15), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel)

            # h, w = green_mask.shape[:2]
            # debug_w = 400
            # debug_h = int(h * (debug_w / w))
            # resized_green_mask = cv2.resize(green_mask, (debug_w, debug_h))
            # cv2.imshow("Debug - Green Mask", resized_green_mask)
            # # -----------------------------------------
            
            roi_for_white_detection = green_mask

            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if green_contours:
                largest_green_contour = max(green_contours, key=cv2.contourArea)
                perimeter = cv2.arcLength(largest_green_contour, True)
                epsilon = 0.04 * perimeter
                approx_corners = cv2.approxPolyDP(largest_green_contour, epsilon, True)

                if len(approx_corners) == 4:
                    green_court_corners = order_points(approx_corners)
                    cv2.drawContours(output_frame, [green_court_corners], -1, (255, 0, 0), 2)
        else:
            roi_for_white_detection = white_detection_roi_mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, white_lines_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        if roi_for_white_detection is None:
            return output_frame, None, green_court_corners
        active_roi = cv2.bitwise_and(white_lines_mask, roi_for_white_detection)

        # h, w = active_roi.shape[:2]
        # debug_w = 400
        # debug_h = int(h * (debug_w / w))
        # resized_active_roi = cv2.resize(active_roi, (debug_w, debug_h))
        # cv2.imshow("Debug - White Mask", resized_active_roi)
        # # ----------------------------------------------------

        dilate_kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(active_roi, dilate_kernel, iterations=2)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5000:
                perimeter = cv2.arcLength(largest_contour, True)
                epsilon = 0.04 * perimeter
                approx_corners = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx_corners) == 4:
                    sorted_corners = order_points(approx_corners)
                    tl, tr, br, bl = sorted_corners.reshape(4, 2)
                    avg_top_y = int((tl[1] + tr[1]) / 2)
                    avg_bottom_y = int((bl[1] + br[1]) / 2)
                    stabilized_pts = np.array([
                        [tl[0], avg_top_y], [tr[0], avg_top_y],
                        [br[0], avg_bottom_y], [bl[0], avg_bottom_y]
                    ], dtype=np.float32)
                    stabilized_corners = order_points(stabilized_pts.reshape(4, 1, 2))
                    cv2.drawContours(output_frame, [stabilized_corners], -1, (0, 255, 0), 3)
                    return output_frame, stabilized_corners, green_court_corners
                else:
                    cv2.drawContours(output_frame, [largest_contour], -1, (0, 0, 255), 2)

        return output_frame, None, green_court_corners

    def process_frame(self, frame):

        processed_frame = frame.copy()

        if not self.is_roi_locked:
            #search
            processed_frame, detected_white_corners, detected_green_corners = self._detect_court_boundary(frame, None)

            if detected_green_corners is not None:
                if self.last_good_green_corners is None:
                    self.last_good_green_corners = detected_green_corners
                else:
                    avg_distance = np.mean(np.linalg.norm(detected_green_corners - self.last_good_green_corners, axis=2))
                    if avg_distance < self.MAX_CORNER_SHIFT:
                        alpha = 0.25
                        self.last_good_green_corners = (alpha * detected_green_corners + (1 - alpha) * self.last_good_green_corners).astype(np.int32)

            if detected_white_corners is not None:
                if self.last_good_white_corners is None:
                    self.last_good_white_corners = detected_white_corners
                else:
                    avg_distance = np.mean(np.linalg.norm(detected_white_corners - self.last_good_white_corners, axis=2))
                    if avg_distance < self.MAX_CORNER_SHIFT:
                        alpha = 0.25
                        self.last_good_white_corners = (alpha * detected_white_corners + (1 - alpha) * self.last_good_white_corners).astype(np.int32)

            if self.last_good_white_corners is not None:
                print("Stable white boundary found. Locking ROI.")
                self.locked_roi_display_corners = self.last_good_white_corners.reshape(4, 2).astype(np.int32)
                self.is_roi_locked = True

        # --- Prepare the final display frame ---
        output_frame = frame.copy()
        status_text = "STATUS: ROI LOCKED" if self.is_roi_locked else "STATUS: SEARCHING FOR COURT"
        cv2.putText(output_frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        if self.is_roi_locked:
            overlay = output_frame.copy()
            cv2.fillPoly(overlay, [self.locked_roi_display_corners], (255, 0, 0))
            alpha = 0.3
            output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)
            cv2.polylines(output_frame, [self.locked_roi_display_corners], isClosed=True, color=(255, 100, 100), thickness=2)
        elif self.last_good_green_corners is not None:
            cv2.drawContours(output_frame, [self.last_good_green_corners], -1, (255, 0, 0), 2)

        if not self.is_roi_locked and self.last_good_white_corners is not None:
            cv2.drawContours(output_frame, [self.last_good_white_corners], -1, (0, 255, 0), 3)

        final_frame_to_show = output_frame if (self.is_roi_locked or self.last_good_white_corners is not None) else processed_frame
        
        court_corners = self.locked_roi_display_corners if self.is_roi_locked else None
        return final_frame_to_show, court_corners