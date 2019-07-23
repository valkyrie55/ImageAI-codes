#In imageAI we have  RetinaNet, YOLOv3 and TinyYOLOv3 for objet detection

from imageai.Detection import ObjectDetection # import the ImageAI object detection class
import os

working_path = os.getcwd()

res = ObjectDetection()  #create an object of the ObjectDetection class 
res.setModelTypeAsYOLOv3()
# https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
res.setModulePath(os.path.join(working_path,"yolo.h5")) #set the model path to the YOLOv3 model file 
res.loadModel()

detection = res.detectObjectsFromImage(input_image = os.path.join(working_path,image_1.jpg),output_image_path = os.path.join(working_path,image_2.jpg), minimum_percentage_probability = 30)

#Display
for obj in detection:
	print(obj["name"]," : ", obj["percentage_probability"]," : ",obj["box_points"])

# we can also use RetinaNet which is appropriate for high-performance and high-accuracy tasks
# https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5
# res = ObjectDetection()
# res.setModelTypeAsRetinaNet()
# res.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# res.loadModel()

# or TinyYOLOv3 which is optimized for speed and embedded devices
# https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny
# res = ObjectDetection()
# res.setModelTypeAsTinyYOLOv3()
# res.setModelPath( os.path.join(execution_path , "yolo-tiny.h5"))
# res.loadModel()