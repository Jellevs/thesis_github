from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path, verbose=True)
        self.tracker = sv.ByteTrack()

        # Annotators
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#FFFFFF"),
            base=25,
            height=21,
            outline_thickness=1
        )

    def calculate_detections(self, frames):
        batch_size = 16

        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.3)
            detections += detections_batch
         
        return detections


    def track_objects(self, detections):
        BALL_ID = 0

        tracked_frames = []
        ball_detections = []
        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)

            ball_detection = detection_supervision[detection_supervision.class_id == BALL_ID]
            all_detections = detection_supervision[detection_supervision.class_id != BALL_ID]

            # Tracker objects
            detections_with_tracking = self.tracker.update_with_detections(detections=all_detections)
            tracked_frames.append(detections_with_tracking)
            ball_detections.append(ball_detection)

        return tracked_frames, ball_detections


    def draw_annotations(self, frames, tracks_detections, ball_detections):
        annotated_frames = []
        for frame_num, frame in enumerate(frames):
            annotated_frame = frame.copy()
            annotated_frame = self.ellipse_annotator.annotate(
                scene = annotated_frame,
                detections = tracks_detections[frame_num]
            )
            annotated_frame = self.triangle_annotator.annotate(
                scene = annotated_frame,
                detections = ball_detections[frame_num]
            )
            annotated_frames.append(annotated_frame)
        return annotated_frames

    



