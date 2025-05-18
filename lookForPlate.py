import cv2
from ultralytics import YOLO
import supervision as sv
import os
import argparse
import time

def parse_arguments() -> argparse.Namespace:
    parse = argparse.ArgumentParser(description="Live Lisence Plate Reaction")
    parse.add_argument("--webcam-resolution", nargs=2, type=int, default=(1280, 720))
    args = parse.parse_args()
    return args

def main():
    args = parse_arguments()
    cap = cv2.VideoCapture(1)
    frame_width, frame_height = args.webcam_resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    model_path = os.path.join("C:\\Users\Vettri\OneDrive\Desktop\\StealingLicensePlates\\runs\detect\\train25\weights\\best.pt")
    model = YOLO(model_path)
    acc = 0
    confidence_threshold = 0.85

    box = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )




    while True:
        ret, frame = cap.read()
        results = model(frame)

        detections = sv.Detections.from_yolov8(results[0])
        label = [
            f"{model.names[1]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]

        detections_list = list(detections)
        annotated_frame = box.annotate(
            scene=frame,
            detections=detections_list,
            labels=label
        )
        #"""

        # Check if there are detected objects and if any of them meet the confidence threshold
        if results[0].boxes.xyxy.numel() > 0:
            for i in range(results[0].boxes.xyxy.shape[0]):
                confidence = results[0].boxes.conf[i]  # Assuming 'conf' holds the confidence values

                if confidence >= confidence_threshold:
                    acc = acc + 1
                    x1, y1, x2, y2 = results[0].boxes.xyxy[i]
                    # Draw bounding box and save image
                    output_filename = f"detectedPlate{acc}.png"
                    cv2.imwrite(output_filename,
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5))

        #"""

        cv2.imshow("Lisence Plate Detector", annotated_frame)

        if cv2.waitKey(30) == 27:
            break

if __name__ == "__main__":
    main()
