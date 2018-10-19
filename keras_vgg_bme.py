'''This script goes along the blog post
    "Building powerful image classification models using very little data"
    from blog.keras.io.
    It uses data that can be downloaded at:
    https://www.kaggle.com/c/dogs-vs-cats/data
    In our setup, we:
    - created a data/ folder
    - created train/ and validation/ subfolders inside data/
    - created cats/ and dogs/ subfolders inside train/ and validation/
    - put the cat pictures index 0-999 in data/train/cats
    - put the cat pictures index 1000-1400 in data/validation/cats
    - put the dogs pictures index 12500-13499 in data/train/dogs
    - put the dog pictures index 13500-13900 in data/validation/dogs
    So that we have 1000 training examples for each class, and 400 validation examples for each class.
    In summary, this is our directory structure:
    ```
    data/
    train/
    dogs/
    dog001.jpg
    dog002.jpg
    ...
    cats/
    cat001.jpg
    cat002.jpg
    ...
    validation/
    dogs/
    dog001.jpg
    dog002.jpg
    ...
    cats/
    cat001.jpg
    cat002.jpg
    
    ...
    ```
    '''

from keras import applications
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras.models
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
from keras.callback import ModelCheckpoint
from leras.callback import TensorBoard
import os.path


base_model = applications.VGG16(weights = 'imagenet', include_top = False)
print('model loaded')
# dataset information
img_width, img_height = 100, 100
train_data_dir = '/data/train'
test_data_dir = '/data/test'
num_train = 1000
num_test = 250
num_classes = 4
# training parameters
top_layer_weight_path = 'top.hdf5'
fine_tuned_weight_path = 'tuned.hdf5'
# epoch number
top_layer_epoch = 50
fine_tune_epoch = 50
batch_size = 32
# build top model
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(.5)(x)
Predictions = Dense(num_classes, activation = 'softmax' )(x)
model = Model(input = base_model.input, output = Predictions)
if os.path.exists(top_layer_weight_path):
    model.load_weights(top_layer_weight_path)
    print("chechpoint" + top_layer_weight_path + "loaded")
# only train the top layers
for layer in base_model.layers:
    layer.trainable = False
# compile the model
model.compile(optimizer = rmsprop,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
# prepare the train data
train_datagen = ImageDataGenerator(featurewise_center=False, # set the mean to zero
                                   rotation_range=20 # mostly rotate 20 degree
                                   horizontal_flip=True
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
# prepare the test data
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train.datagen.flow_from_directory(
                                                    train_data_dir,
                                                    target_size = (img_height, img_width),
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical'
                                                    )
test_generator = test_datagen.flow_from_directory(
                                                  test_data_dir,
                                                  target_size = (img_height,img_width),
                                                  batch_size = batch_size,
                                                  class_mode = 'categorical'
                                                  )
# save our model each epoch
mc_top = Modelcheckpoint(top_layer_weight_path, monitor = 'val_acc', verbose = 0, save_best_only = True,
                         save_weights_only = False,mode = 'auto', period = 1)
# save the tensorboard logs
tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

# train the top layers on the new data
model.fit_generator(
                    train_generator,
                    samples_per_epoch = num_train // batch_size,
                    epochs = top_layer_epoch,
                    test_data = test_generator,
                    nb_val_samples = num_test // batch_size,
                    callbacks = [mc_top, tb]
                    )
# visualize name of layers

for i, layers in enumerate(base_model.layers):
    print(i,layer.name)
# save the model
mc_fit = ModelCheckpoint(fine_tuned_weight_path, monitor='val_acc',
                         verbose=0, save_best_only=True,
                         save_weights_only=False,
                         mode='auto', period=1)
if os.path.exists(fine_tuned_weight_path):
    model.load_weights(fine_tuned_weight_path)
    print('laod path from {-0}' .format(fine_tuned_weight_path))
# freeze the basic layer
for layer in model.layers[end-5:end]:
    layer.trainable = True
for lyaer in model.layers[:end-4]:
    layer.trainable = False
model.compile(optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit_generator(train_generator,
                    samples_per_epoch = num_train // batch_size,
                    epochs = fine_tune_epoch,
                    validation_data = test_generator,
                    nb_val_samples = num_test//batch_size,
                    callbacks = [mc_fit,tb]
                    )
model.save_weights(fine_tuned_weight_path)
print('Saved trained model at %s ' % model_path)
# score the trained model
#socres = model.evaluate()



































