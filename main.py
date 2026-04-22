"""Smoke test: confirm YOLOv8n loads and runs on the bundled bus image."""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")
    results = model("https://ultralytics.com/images/bus.jpg")
    for result in results:
        for box in result.boxes:
            print(result.names[int(box.cls[0])], box.xyxy.tolist())


if __name__ == "__main__":
    main()
