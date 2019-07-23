#Multiple image prediction
#
from imageai.Prediction import ImagePrediction
import os
working_path = os.getcwd()

multiple_prediction = ImagePrediction()
multiple_prediction.setModelTypeResNet()
multiple_prediction.setModelPath(os.path.join(working_path,
	"resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
multiple_prediction.loadModel(prediction_speed="fast")

all_images = []

all_files = os.listdir(working_path)
for file in all_files:
	if(file.endswith(".jpg") or file.endswith(".png")):
		all_images.append(file)

final_array = multiple_prediction.predictMultipleImages(all_images, result_count_per_image = 5)

#Display the output
for each_result in final_array:
	pred , percent = each_result["pred"],each_result["percent"]
    for i in range(len(pred)):
    	print(pred[i]," : ", percent[i])
