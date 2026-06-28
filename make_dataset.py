import cv2
import os
import csv
import time

# --- CẤU HÌNH ---
VIDEO_PATH = 'badlong.mp4'
DATASET_DIR = 'dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
CSV_FILE = os.path.join(DATASET_DIR, 'labels.csv')

# ĐÓNG DẤU TIẾN TRÌNH THEO TÊN VIDEO ĐỂ KHÔNG BỊ LẪN LỘN
video_filename = os.path.basename(VIDEO_PATH)
PROGRESS_FILE = os.path.join(DATASET_DIR, f'progress_{video_filename}.txt')

FRAME_SKIP = 10  
TARGET_SIZE = (416, 416) 

current_frame_display = None
clicked_coords = None
is_clicked = False

def mouse_callback(event, x, y, flags, param):
    global clicked_coords, is_clicked, current_frame_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_coords = (x, y)
        is_clicked = True
        cv2.circle(current_frame_display, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Data Annotator', current_frame_display)

def main():
    global current_frame_display, is_clicked, clicked_coords

    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)

    file_exists = os.path.isfile(CSV_FILE)
    csv_file = open(CSV_FILE, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not file_exists:
        csv_writer.writerow(['image_name', 'x', 'y'])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {VIDEO_PATH}")
        return

    #khoi phuc last time
    start_frame = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                start_frame = int(content)
                
    if start_frame > 0:
        print(f"[*] Đang tua {video_filename} lại đến khung hình số {start_frame}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        print(f"[*] Bắt đầu làm dữ liệu {video_filename} từ đầu video.")

    cv2.namedWindow('Data Annotator')
    cv2.setMouseCallback('Data Annotator', mouse_callback)

    frame_count = start_frame
    saved_count = 0


    print("- CHUỘT TRÁI: Ghi nhận tọa độ và TỰ ĐỘNG nhảy frame.")
    print("- Nhấn phím 'Space': BỎ QUA frame này.")
    print("- Nhấn phím 'q': LƯU TIẾN TRÌNH VÀ THOÁT.")
    
    exit_program = False

    while cap.isOpened() and not exit_program:
        ret, frame = cap.read()
        if not ret:
            print(f"Đã hết video {video_filename}.")
            break

        frame_count += 1
        
        if frame_count % FRAME_SKIP != 0:
            continue

        processed = cv2.resize(frame, TARGET_SIZE)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

        display_frame = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        image_to_save = processed.copy()
        
        current_frame_display = display_frame.copy()
        is_clicked = False
        clicked_coords = None

        while True:
            cv2.imshow('Data Annotator', current_frame_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                exit_program = True
                break
                
            elif key == ord(' '):
                with open(PROGRESS_FILE, 'w') as f:
                    f.write(str(frame_count))
                break
                
            elif is_clicked:
                img_name = f"frame_{int(time.time() * 1000)}.jpg"
                img_path = os.path.join(IMAGES_DIR, img_name)
                
                cv2.imwrite(img_path, image_to_save)
                
                #continue write csv
                csv_writer.writerow([img_name, clicked_coords[0], clicked_coords[1]])
                csv_file.flush()
                
                with open(PROGRESS_FILE, 'w') as f:
                    f.write(str(frame_count))
                
                saved_count += 1
                print(f"Đã lưu: {img_name} -> Tiến trình {video_filename}: Frame {frame_count}")
                break

    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(frame_count))

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nHoàn thành! Add thêm được {saved_count} tọa độ.")

if __name__ == '__main__':
    main()