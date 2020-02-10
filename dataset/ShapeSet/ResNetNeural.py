from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

model = ResNet50(weights = 'imagenet')

