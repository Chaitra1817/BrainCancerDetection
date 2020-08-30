# Making prediction about user data

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

# load json and create model
json_file = open('/content/drive/My Drive/Brain Tumor CNN/brain_tumor_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Brain Tumor CNN/brain_tumor_model.h5")

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale = 1./255)
user_data = test_datagen.flow_from_directory('/content/drive/My Drive/Brain Tumor CNN/User-Data/test',
                                            target_size = (64, 64),
                                            batch_size = 1,
                                            class_mode = 'binary')
 
# evaluate loaded model on user data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(user_data)
print(score)
#if score[1] == 0:
 # print('The convolutional neural network predicts that that this immage doesn't show signs of hemorrhage.')
#else:
 # print('The convolutional neural network predicts that this image shows signs of a brain-hemorrhage!')
