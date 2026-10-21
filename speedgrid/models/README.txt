yolox_s.onnx - YOLOX-S object detector (COCO, 80 classes), ONNX export from the
OpenCV model zoo (opencv/opencv_zoo, models/object_detection_yolox,
object_detection_yolox_2022nov.onnx). YOLOX is by Megvii, Apache License 2.0.
Used by speedgrid/vehicles.py to find other vehicles (car, motorbike, bus, truck).

vehicle3d.onnx / vehicle3d.json - optional, made by "Train Vehicle Model.bat"
(training/train_vehicle3d.py): ResNet-18 predicting a vehicle's size and
observation angle, trained on KITTI 3D object (CC BY-NC-SA 3.0). Used by
speedgrid/box3d.py when present.
