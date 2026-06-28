import cv2
from court_detect import CourtDetector
from be_rectangle import CourtFlattener
from ai_tracker import AIFeetTracker
from heatmap import StaticCourtHeatmap

def main():
    video_path = 'badminton4.mp4'
 
    cap = cv2.VideoCapture(video_path)
 
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở file video tại {video_path}")
        return
 
    print("Starting.")

    court_detector = CourtDetector()
    court_flattener = CourtFlattener(target_height=480)
    ai_tracker = AIFeetTracker(checkpoint_path='feet_detector_checkpoint.pth')
    
    # generate static heatmap
    static_heatmap = StaticCourtHeatmap(court_image_path='court.png', target_height=800)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
 
        feet_coord_orig = ai_tracker.predict(frame)
        processed_frame, court_corners = court_detector.process_frame(frame)
        warped_court = court_flattener.flatten(frame, court_corners)
        
        feet_coord_flat = court_flattener.map_point(feet_coord_orig)
        
        if feet_coord_flat is not None:
            #transfer + scale
            static_heatmap.add_point_scaled(
                feet_coord_flat, 
                court_flattener.target_width, 
                court_flattener.target_height
            )
            
            # cv2.circle(warped_court, feet_coord_flat, 5, (255, 255, 255), -1)
            cv2.circle(warped_court, feet_coord_flat, 6, (0, 255, 255), 2)
        
        resized_display = cv2.resize(processed_frame, (854, 480))
        combined_display = cv2.hconcat([resized_display, warped_court])
 
        cv2.imshow('Badminton Analysis', combined_display)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    
    final_heatmap = static_heatmap.generate_heatmap()
    
    cv2.imshow("Final Tactical Heatmap", final_heatmap)
    cv2.imwrite("final_tactical_heatmap.jpg", final_heatmap)
    
    print("Đã lưu kết quả thành 'final_tactical_heatmap.jpg'")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()