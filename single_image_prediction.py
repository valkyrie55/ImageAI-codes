from imageai.Prediction import ImagePrediction
import os

working_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath(os.path.join(working_path,"resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel(prediction_speed="fast")

pred, prob = prediction.predictImage(os.path.join(working_path, "sh.jpg"), result_count = 5)
for eachPrediction, eachProbability in zip(pred, prob):
	print(eachPrediction," : ", eachProbability)