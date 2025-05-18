import ultralytics as ul

# Load a model
model = ul.YOLO("yolov8n.yaml")  # build a new model from scratch


# Use the model
model.train(data="https://app.roboflow.com/ds/RpkJHh8LE5?key=v20nPhGHqE", epochs=250)  # train the model
