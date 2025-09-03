import supervision as sv
import cv2
from trackers import Tracker

def main():
    
    # Get frames
    frame_generator = sv.get_video_frames_generator(video_path)
    frames = [frame for frame in frame_generator]
    
    # Run model and Tracker
    tracker = Tracker(model_path)
    detections = tracker.calculate_detections(frames)
    detections_with_tracking = tracker.track_objects(detections)
    annotated_frames = tracker.draw_annotations(frames, detections_with_tracking)

    sv.plot_image(annotated_frames[10])

if __name__ == "__main__":
    video_path = "../input_videos_images/video_1.mp4"
    model_path = "../saved_models/best_100_epochs.pt"
    main()