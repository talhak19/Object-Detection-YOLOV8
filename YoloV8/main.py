from ultralytics import YOLO
 
model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
 
model.train(data='file.yaml',
   imgsz=640,
   epochs=15,
   batch=8,
   name='yolov8n_custom')  # train the model
print("\n-------------------------------------\n")
results = model.val()  # evaluate model performance on the validation set
print(results)

print("\n-------------------------------------\n")
model.predict(source="test/images/chair1.jpg")  # predict on an image

print("\n-------------------------------------\n")
model.export(format="onnx")  # export the model to ONNX format