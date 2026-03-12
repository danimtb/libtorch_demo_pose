from ultralytics import YOLO

# Load the pretrained YOLOv8 nano pose model
model = YOLO('yolov8n-pose.pt')

# Export the model to TorchScript format
model.export(format='torchscript')
