from ultralytics import YOLO

def train():

    model = YOLO("runs/detect/train/weights/last.pt") #where it was left off, if you want to start from the beginning, use "yolov8m.pt"

    model.train(
        data=r"C:\Users\sahil\Desktop\projects\ccp vehcle detection\training data test 1.v1i.yolov8\data.yaml",
        epochs=70,
        imgsz=640,
        batch=16,
        name="train",
        exist_ok=True,
        resume=True
    )

if __name__ == "__main__":
    train()
