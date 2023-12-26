import cv2
import inference
import supervision as sv

import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("DATASET_API_KEY")

# !pip install roboflow 

from roboflow import Roboflow
rf = Roboflow(api_key="DATASET_API_KEY")
# rf = Roboflow(api_key="DATASET_API_KEY")
project = rf.workspace("mohamed-traore-2ekkp").project("table-extraction-with-null-images")
dataset = project.version(6).download("yolov8")


annotator = sv.BoxAnnotator()

inference.Stream(
    # source="webcam", # or rtsp stream or camera id
    model="rock-paper-scissors-sxsw/11", # from Universe

    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    
    on_prediction=lambda predictions, image: (
        print(predictions), # now hold up your hand
        
        cv2.imshow(
            "Prediction", 
            annotator.annotate(
                scene=image, 
                detections=sv.Detections.from_roboflow(predictions)
            )
        ),
        cv2.waitKey(1)
    )
)
