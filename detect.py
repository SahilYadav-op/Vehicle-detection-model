from ultralytics import YOLO

model = YOLO(r"C:\Users\sahil\Desktop\projects\ccp vehcle detection\runs\detect\train\weights\best.pt")

model.predict(
    source=r"C:\Users\sahil\Desktop\projects\ccp vehcle detection\clean_images\0001864.jpg",
    show=True,
    conf=0.25
)
