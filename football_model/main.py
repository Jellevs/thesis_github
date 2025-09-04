import supervision as sv
import cv2
from trackers import Tracker
from video_utils import save_video, read_video, save_images
import time

def main():
    
    # Read video and get frames
    frames = read_video(input_video_path=input_video_path)
    
    # Initialize YOLO model and Tracker
    tracker = Tracker(model_path)

    # Calculate detections
    detections = tracker.calculate_detections(frames,
                                              conf=0.1)
    
    # Add trackers
    detections_with_tracking, ball_detections = tracker.track_objects(detections)

    # Annotate frames
    annotated_frames = tracker.draw_annotations(frames, detections_with_tracking, ball_detections)

    # Save video
    save_images(output_folder="output_images", 
                annotated_frames=annotated_frames)


    save_video(output_video_path="output_videos", 
               input_video_path=input_video_path, 
               annotated_frames=annotated_frames)


if __name__ == "__main__":
    input_video_path = "../input_videos_images/video_1.mp4"
    model_path = "../saved_models/best_100_epochs.pt"
    main()