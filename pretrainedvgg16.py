import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import applications
import os
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

model = applications.VGG16(include_top=False, weights='imagenet')
model_pre = Model(model.inputs, model.output)


# inside directory there has to be many folders for each classes

data_dir = "Train/"
batch_size=128
image_size=100
datagen_top = ImageDataGenerator(rescale=1. / 255,rotation_range=9,horizontal_flip=True)
generator_top = datagen_top.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

nb_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    labels = generator_top.classes
    print("Shape of Labels and num of classes ",labels.shape,num_classes)
   
    predict_size_data = int(math.ceil(nb_samples / args.batch_size))
    print("Extracting Features")
    
bottleneck_features = model_pre.predict_generator(
        generator_top, predict_size_data, verbose=1)
np.save('bottleneck_features_train_Inception_ResnetV1.npy',bottleneck_features)
    np.save('train_labels.npy',labels)

    #bottleneck_features = np.load('bottleneck_features_train_Inception_ResnetV1.npy')
    #labels = np.load('train_labels.npy')


 # for i in tqdm(range(0,bottleneck_features.shape[0])):
 #        X=bottleneck_features[i]
 #        y=X.reshape(X.shape[0]*X.shape[1],X.shape[2])   #Shape (14*14, 512)
 #        print(y.shape)
reshaped=bottleneck_features.reshape(-1,bottleneck_features.shape[1]*bottleneck_features.shape[2]*bottleneck_features.shape[3])

print('Training classifier')
modelSVC = SVC(kernel = 'linear',probability=True, verbose=True)
        modelSVC.fit(bottleneck_features, labels)

from sklearn.externals import joblib

classifier_filename_exp="modelsvm.pkl"

with open(classifier_filename_exp, 'wb') as outfile:
            joblib.dump(modelSVC,outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)



=========


#elif (args.mode=='CLASSIFY'):
datagen_top = ImageDataGenerator(rescale=1. / 255,rotation_range=9,horizontal_flip=True)
generator_top = datagen_top.flow_from_directory(
data_dir,
target_size=(image_size, image_size),
batch_size=batch_size,
class_mode='categorical',
shuffle=False)

nb_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)
labels = generator_top.classes
print("Shape of Labels and num of classes ",labels.shape,num_classes)

predict_size_data = int(math.ceil(nb_samples / args.batch_size))
print("Extracting Features")

bottleneck_features = model_pre.predict_generator(
generator_top, predict_size_data, verbose=1)

np.save('test.npy',bottleneck_features)
np.save('test_labels.npy',labels)

reshaped=bottleneck_features.reshape(-1,bottleneck_features.shape[1]*bottleneck_features.shape[2]*bottleneck_features.shape[3])



print("test labels:")

    #np.save("testlabels.npy",labels)
    #np.save('bottleneck_features_cov_test.npy',bottleneck_features)
print('Testing classifier')
with open(classifier_filename_exp, 'rb') as infile:
    modelSVM = joblib.load(infile)
print('Loaded classifier model from file "%s"' % classifier_filename_exp)     
print("Predicting Images")
predictions = modelSVM.predict_proba(bottleneck_features)
best_class_indices = np.argmax(predictions, axis=1)
best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

arr=[]
for i in range(len(best_class_indices)):
    arr.append((predictions[i],best_class_indices[i],labels[i]))
print(arr)
# pickle.dump(arr, open('classification_data.p','wb'))
accuracy = 100*np.mean(np.equal(best_class_indices, labels))
print('Total Accuracy: %.3f' % accuracy)
    '''
    train_data = np.load('bottleneck_features_train.npy')
    train_data = train_data.reshape(train_data.shape[0],-1)
    train_data = scale(train_data)
    print("Shape of train data ",train_data.shape)