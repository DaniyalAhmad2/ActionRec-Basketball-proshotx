import csv
import sys
import os
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

def main():
    """
    Usage: python script.py <annotations_csv> <pose_model.task>

    Where:
      - annotations_csv is your CSV containing rows like:
           video_name,event_name,start_frame_num,end_frame_num,lstm_category
      - pose_model.task is the MediaPipe Pose Landmarker model asset path.
    """
    # if len(sys.argv) < 3:
    #     print("Usage: python script.py <annotations_csv> <pose_model.task>")
    #     sys.exit(1)

    annotation_csv_path = "shot_phase.csv"
    model_asset_path = "pose_landmarker.task"

    # -------------------------------
    # 1. Read all intervals from the CSV (only 1 video)
    # -------------------------------
    intervals = []
    video_name = "Front-60fps.mp4"  # We'll detect the video name from the first row

    with open(annotation_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header row
        for row in reader:
            # row format: [video_name, event_name, start_frame_num, end_frame_num, lstm_category]
            if video_name is None:
                video_name = row[0]  # Store the video name from the first row

            event_name = row[1]
            start_frame = int(row[2])
            end_frame = int(row[3])
            intervals.append((start_frame, end_frame, event_name))

    # Sort intervals by start_frame
    intervals.sort(key=lambda x: x[0])

    # -------------------------------
    # 2. Open the single video once
    # -------------------------------
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_name}'")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------------------
    # 3. Prepare output directories
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)
    annotated_video_path = os.path.join("outputs", "annotated_video.mp4")
    csv_output_path = os.path.join("outputs", "all_data.csv")

    # Create a VideoWriter for the annotated result
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

    # -------------------------------
    # 4. Initialize MediaPipe PoseLandmarker
    # -------------------------------
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    landmarker_opts = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False  # We only need landmarks
    )
    detector = vision.PoseLandmarker.create_from_options(landmarker_opts)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # -------------------------------
    # 5. Prepare a single CSV for ALL frames
    # -------------------------------
    # We'll store up to 17 landmarks: (x1,y1, x2,y2, ..., x17,y17)
    NUM_LANDMARKS = 17
    fieldnames = ["frame_number"]
    for i in range(1, NUM_LANDMARKS + 1):
        fieldnames.append(f"x{i}")
        fieldnames.append(f"y{i}")
    fieldnames.append("label")

    out_csv_file = open(csv_output_path, "w", newline="")
    writer = csv.DictWriter(out_csv_file, fieldnames=fieldnames)
    writer.writeheader()
    classes_ = {"LowPocket" : 0, "HighPocket" : 1, "Ball_RELEASE" : 2, "Release" : 3, "PostShot" : 4 }
    def get_label_for_frame(frame_idx):
        """
        Returns the first event_name whose [start, end] interval contains frame_idx.
        If no interval matches, returns 'Background'.
        """
        for (start_f, end_f, evt_name) in intervals:
            if start_f <= frame_idx <= end_f:
                return classes_[evt_name], evt_name
        return 5, "Background"

    # -------------------------------
    # 6. Process every frame in the video
    # -------------------------------
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        label, str_label = get_label_for_frame(frame_idx)

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect pose landmarks
        detection_result = detector.detect(mp_image)

        # Prepare a row for CSV
        row_dict = {"frame_number": frame_idx, "label": label}
        # Initialize all x_i,y_i to 0.0 in case no pose is found
        for i in range(1, NUM_LANDMARKS + 1):
            row_dict[f"x{i}"] = 0.0
            row_dict[f"y{i}"] = 0.0

        # Make a copy of the frame for annotation
        annotated_frame = frame.copy()

        # If we have at least one pose, fill in landmark data + draw them
        if len(detection_result.pose_landmarks) > 0:
            pose_landmarks = detection_result.pose_landmarks[0]

            # Write up to the first 17 landmarks
            for i in range(min(NUM_LANDMARKS, len(pose_landmarks))):
                lx = pose_landmarks[i].x
                ly = pose_landmarks[i].y
                row_dict[f"x{i+1}"] = lx
                row_dict[f"y{i+1}"] = ly

            # Draw landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=l.x, y=l.y, z=l.z) 
                for l in pose_landmarks
            ])
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_landmarks_proto,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Put label in top-right corner
        text_size, baseline = cv2.getTextSize(str_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = width - text_size[0] - 10
        text_y = 40
        cv2.putText(
            annotated_frame,
            str_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Write annotated frame and CSV row
        out_writer.write(annotated_frame)
        cv2.imshow("annotated_frame",annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            # Ignore 'q', continue playing the video
            continue
        writer.writerow(row_dict)

        frame_idx += 1

    # Cleanup
    cap.release()
    out_writer.release()
    out_csv_file.close()
    cv2.destroyAllWindows()

    print("\nDone!")
    print(f" - CSV with all frames: {csv_output_path}")
    print(f" - Annotated video: {annotated_video_path}")

if __name__ == "__main__":
    main()
