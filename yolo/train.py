from ultralytics import YOLO
 
if __name__=="__main__":
    # Load a model
    model_yaml=r"C:\Users\57704\Desktop\tri_pre\ultralytics\ultralytics\cfg\models\11\yolo11.yaml"
    data_yaml=r"C:\Users\57704\Desktop\tri_pre\data.yaml"
    pre_model=r"C:\Users\57704\Desktop\tri_pre\yolo11n.pt"
 
    model = YOLO(model_yaml,task='detect').load(pre_model)  # build from YAML and transfer weights
 
    # Train the model
    results = model.train(data=data_yaml, epochs=15, imgsz=640,batch=4,workers=2)