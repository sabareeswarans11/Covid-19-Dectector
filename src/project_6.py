#!/usr/bin/env python
# coding: utf-8

# In[1]:


from builtins import range, input
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import cv2
from glob import glob
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from PyQt5 import QtCore, QtGui, QtWidgets
import sys


# In[2]:


#reshape the input image with respect to cnn model (here:resnet50 and inceptionv3)
IMAGE_SIZE = [224, 224]

# training config:
epochs = 10         #user specified 
batch_size = 32

#import the dataset from local machine
covid_path = '/Users/sabareeswarans/Desktop/AI projects/pro6_dataset/COVID19_Data/positive'
noncovid_path = '/Users/sabareeswarans/Desktop/AI projects/pro6_dataset/COVID19_Data/negative'

# To grab images from path .jpg or jpeg
positive_files = glob(covid_path + '/*')
negative_files = glob(noncovid_path + '/*')


# In[3]:


# Visualize file variable contents
print("First 3 Covid Files: ",positive_files[0:3])
print("Total Count: ",len(positive_files))
print("First 3 NonCovid Files: ",negative_files[0:3])
print("Total Count: ",len(negative_files))


# In[4]:


# Fetch Images and Class Labels from Files
covid_labels = []
noncovid_labels = []

covid_images=[]
noncovid_images=[]

for i in range(len(positive_files)):
  image = cv2.imread(positive_files[i]) # read file 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
  image = cv2.resize(image,(224,224)) # resize as per model
  covid_images.append(image) # append image
  covid_labels.append('CT_COVID') #append class label
for i in range(len(negative_files)):
  image = cv2.imread(negative_files[i])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image,(224,224))
  noncovid_images.append(image)
  noncovid_labels.append('CT_NonCOVID')


# In[5]:


# diaplaying some random input as images.
def plot_images(images, title):
    nrows, ncols = 5, 8
    figsize = [10, 6]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, facecolor=(1, 1, 1))

    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i])
        axi.set_axis_off()

    plt.suptitle(title, fontsize=24)
    plt.tight_layout(pad=0.2, rect=[0, 0, 1, 0.9])
    plt.show()
plot_images(covid_images, 'Positive COVID-19 CT Scan')
plot_images(noncovid_images, 'Negative COVID-19 CT Scan')


# In[6]:


# Convert to array and Normalize to interval of [0,1]
covid_images = np.array(covid_images) / 255
noncovid_images = np.array(noncovid_images) / 255


# In[7]:


# Split into training(80%) and testing sets(20%) for both types of images
covid_x_train, covid_x_test, covid_y_train, covid_y_test = train_test_split(
    covid_images, covid_labels, test_size=0.2)
noncovid_x_train, noncovid_x_test, noncovid_y_train, noncovid_y_test = train_test_split(
    noncovid_images, noncovid_labels, test_size=0.2)

# Merge sets for both types of images
X_train = np.concatenate((noncovid_x_train, covid_x_train), axis=0)
X_test = np.concatenate((noncovid_x_test, covid_x_test), axis=0)
y_train = np.concatenate((noncovid_y_train, covid_y_train), axis=0)
y_test = np.concatenate((noncovid_y_test, covid_y_test), axis=0)

# Make labels into categories - either 0 or 1, for resnet ,inception model
y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)

print("Shape of X_train",X_train.shape)
print("Shape of y_train",y_train.shape)
print("Shape of X_test",X_test.shape)
print("Shape of y_test",y_test.shape)


# In[8]:


plot_images(covid_x_train, 'X_train')
plot_images(covid_x_test, 'X_test')
# y_train and y_test contain class lables 0 and 1 representing COVID and NonCOVID for X_train and X_test


# In[9]:


#structure and architecture of resnet50 (TASK 3)
print("ResNet Architecture")
Image("./pro6_image/resnet_structure.png")


# In[10]:


print("Inception Module")
Image("./pro6_image/inception_module.png")


# In[11]:


print("InceptionV3 Structure")
Image("./pro6_image/Inceptionv3_structure.png")


# In[12]:


# Building Model ResNet50
resnet = ResNet50(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224,224, 3)))

outputs_res = resnet.output
outputs_res = Flatten(name="flatten")(outputs_res)
outputs_res = Dropout(0.5)(outputs_res)
outputs_res = Dense(2, activation="softmax")(outputs_res)

model_res = Model(inputs=resnet.input, outputs=outputs_res)

for layer in resnet.layers:
    layer.trainable = False

model_res.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


# In[13]:


#ResNet50 Model summary  (Task 3 structure of the model)
model_res.summary()


# In[14]:


#inception model with optimizer = adam (TASK 2)
inception = InceptionV3(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs_inc_1 = inception.output
outputs_inc_1 = Flatten(name="flatten")(outputs_inc_1)
outputs_inc_1 = Dropout(0.5)(outputs_inc_1)
outputs_inc_1 = Dense(2, activation="softmax")(outputs_inc_1)

model_inc_1 = Model(inputs=inception.input, outputs=outputs_inc_1)

for layer in inception.layers:
    layer.trainable = False


model_inc_1.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)


# In[15]:


#inception model with optimizer = sgd(TASK 2)
inception_2 = InceptionV3(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs_inc_2 = inception_2.output
outputs_inc_2 = Flatten(name="flatten")(outputs_inc_2)
outputs_inc_2 = Dropout(0.5)(outputs_inc_2)
outputs_inc_2 = Dense(2, activation="softmax")(outputs_inc_2)

model_inc_2 = Model(inputs=inception_2.input, outputs=outputs_inc_2)

for layer in inception.layers:
    layer.trainable = False


model_inc_2.compile(
        loss='categorical_crossentropy', 
        optimizer='sgd', 
        metrics=['accuracy']
)


# In[16]:


#inception model with optimizer = rmsprop(TASK 2)
inception_3 = InceptionV3(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs_inc_3 = inception_3.output
outputs_inc_3 = Flatten(name="flatten")(outputs_inc_3)
outputs_inc_3 = Dropout(0.5)(outputs_inc_3)
outputs_inc_3 = Dense(2, activation="softmax")(outputs_inc_3)

model_inc_3 = Model(inputs=inception_3.input, outputs=outputs_inc_3)

for layer in inception.layers:
    layer.trainable = False


model_inc_3.compile(
        loss='categorical_crossentropy', 
        optimizer='rmsprop', 
        metrics=['accuracy']
)


# In[17]:


#inception Model summary  (Task 3 structure of the model)
model_inc_1.summary()
model_inc_2.summary()
model_inc_3.summary()


# In[18]:


#Generate the batches of tensor image data with real time augmentation 
train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[19]:


#training and calculating accuracy using ResNet50 with adam optimizer
training_adam_resnet = model_res.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[20]:


#Graph view -->Model Accuracy of ResNET50 with Adam
plt.plot(training_adam_resnet.history['accuracy'])
plt.plot(training_adam_resnet.history['val_accuracy'])

plt.title('Model Accuracy of ResNET50 with Adam')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/resnet_accuracy.png')
plt.show()

res_acc= np.array(training_adam_resnet.history['accuracy'])
print( 'Training Accuracy of ResNet50 with Adam :%.2f' %(res_acc[-1]*100),'%')

res_vacc = np.array(training_adam_resnet.history['val_accuracy'])
print('Testing Accuracy of ResNet50 with Adam :%.2f' % (res_vacc[-1]*100), '%')


# In[21]:


#Graph view -->Model Loss of ResNET50 with Adam
plt.plot(training_adam_resnet.history['loss'])
plt.plot(training_adam_resnet.history['val_loss'])

plt.title('Model Loss of ResNET50 with Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/resnet_loss.png')
plt.show()


# In[22]:


#training and calculating accuracy using inception with adam optimizer
training_adam_inc = model_inc_1.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[23]:


#Graph view -->Model Accuracy of inception with Adam
plt.plot(training_adam_inc.history['accuracy'])
plt.plot(training_adam_inc.history['val_accuracy'])

plt.title('Model Accuracy of inception with Adam')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_accuracy_adam.png')
plt.show()


inc_acc_adam = np.array(training_adam_inc.history['accuracy'])
print('Training Accuracy of Inception with Adam :%.2f' % (inc_acc_adam[-1]*100), '%')

inc_vacc_adam = np.array(training_adam_inc.history['val_accuracy'])
print('Testing Accuracy of Inception with Adam :%.2f' % (inc_vacc_adam[-1]*100), '%')


# In[24]:


#Graph view -->Model Loss of inception with Adam
plt.plot(training_adam_inc.history['loss'])
plt.plot(training_adam_inc.history['val_loss'])

plt.title('Model Loss of inception with Adam')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_loss_adam.png')
plt.show()


# In[25]:


#training and calculating accuracy using inception with sgd optimizer
training_sgd_inc = model_inc_2.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[26]:


#Graph view -->Model Accuracy of inception with SGD
plt.plot(training_sgd_inc.history['accuracy'])
plt.plot(training_sgd_inc.history['val_accuracy'])

plt.title('Model Accuracy of inception with SGD')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_accuracy_sgd.png')
plt.show()

inc_acc_sgd = np.array(training_sgd_inc.history['accuracy'])
print('Training Accuracy of Inception with SGD :%.2f' % (inc_acc_sgd[-1]*100), '%')

inc_vacc_sgd = np.array(training_sgd_inc.history['val_accuracy'])
print('Testing Accuracy of Inception with SGD  :%.2f' % (inc_vacc_sgd[-1]*100), '%')


# In[27]:


#Graph view -->Model Loss of inception with SGD
plt.plot(training_sgd_inc.history['loss'])
plt.plot(training_sgd_inc.history['val_loss'])

plt.title('Model Loss of inception with SGD')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_loss_sgd.png')
plt.show()


# In[28]:


#training and calculating accuracy using inception with rmsprop optimizer
training_rmsprop_inc = model_inc_3.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[29]:


# Graph view -->Model Accuracy of inception with rmsprop
plt.plot(training_rmsprop_inc.history['accuracy'])
plt.plot(training_rmsprop_inc.history['val_accuracy'])

plt.title('Model Accuracy of inception with rmsprop')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_accuracy_rms.png')
plt.show()

inc_acc_rms = np.array(training_rmsprop_inc.history['accuracy'])
print('Training Accuracy of Inception with Rmsprop :%.2f' % (inc_acc_rms[-1]*100), '%')

inc_vacc_rms = np.array(training_rmsprop_inc.history['val_accuracy'])
print('Testing Accuracy of Inception with Rmsprop :%.2f' % (inc_vacc_rms[-1]*100), '%')


# In[30]:


# Graph view -->Model Loss of inception with rmsprop
plt.plot(training_rmsprop_inc.history['loss'])
plt.plot(training_rmsprop_inc.history['val_loss'])

plt.title('Model Loss of inception with rmsprop')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('./pro6_accuracy_image/inception_loss_rms.png')
plt.show()


# In[31]:


from prettytable import PrettyTable

r =[["ResNet50 with Adam optimizer",'%.2f' %(res_acc[-1]*100),'%.2f' %(res_vacc[-1]*100)],
    ["Inception With Adam optimizer",'%.2f' %(inc_acc_adam[-1]*100),'%.2f' %(inc_vacc_adam[-1]*100)],[" Inception with SGD optimizer",'%.2f' %(inc_acc_sgd[-1]*100),'%.2f' %(inc_vacc_sgd[-1]*100)],
    ["Inception with Rmsprop ",'%.2f' %(inc_acc_rms[-1]*100),'%.2f' %(inc_vacc_rms[-1]*100)] ]
t= PrettyTable(["Different DCCN Model With Differnt Optimizers for Covid19-ct Images ", "Training Accuracy"," Testing Accuracy"])

for rec in r:
    t.add_row(rec)

print(t)


# In[32]:


#Save the trained model in local machine (which will be useful for GUI TASK4) 
model_res.save('./pro6_savedModel/resnet.h5')
model_inc_1.save('./pro6_savedModel/inception_adam.h5')
model_inc_2.save('./pro6_savedModel/inception_sgd.h5')
model_inc_3.save('./pro6_savedModel/inception_rmsprop.h5')


# In[33]:


#figure out the best model from the above and load for testing samples and also for GUI testing .
best_model = load_model('./pro6_savedModel/inception_sgd.h5')
y_pred = best_model.predict(X_test, batch_size=batch_size)

# Convert to Binary classes 
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)


# In[34]:


#Testing the model
prediction=y_pred[200:210]
for index, probability in enumerate(prediction):
  if probability[1] > 0.5:
        plt.title('%.2f' % (probability[1]*100) + '% COVID')
  else:
        plt.title('%.2f' % ((1-probability[1])*100) + '% NonCOVID')
  plt.imshow(X_test[index])
  plt.show()


# In[35]:


#confusion_matrix for best model
def plot_confusion_matrix(normalize):
  classes = ['COVID','NonCOVID']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='plasma',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)
print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')


# In[36]:


#TASK 4 GUI 
# This allows the user to import the image from local machine and with the saved model(already trained model) it performs testing.
image_size = 224
class GUI_covid19_detector(object):
    def retranslateUi(self, MainScreen):
        _translate = QtCore.QCoreApplication.translate
        MainScreen.setWindowTitle(_translate("MainScreen", "COVID-19 Detector"))
        self.predictedLabel.setText(_translate("MainScreen", "Your Covid Test Result:"))
        self.browseImageBtn.setText(_translate("MainScreen", "Browse Image"))
        self.predictBtn.setText(_translate("MainScreen", "Start Testing"))
        
    def setupUi(self, GuiWindow):            
        GuiWindow.setObjectName("MainScreen")
        GuiWindow.resize(547, 592)
        GuiWindow.setMaximumSize(QtCore.QSize(547, 700))
        self.centralwidget = QtWidgets.QWidget(GuiWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.InputImage = QtWidgets.QLabel(self.centralwidget)
        self.InputImage.setGeometry(QtCore.QRect(20, 40, 280, 350))
        self.InputImage.setText("")
        self.InputImage.setObjectName("InputImage")
        
        self.predictedLabel = QtWidgets.QLabel(self.centralwidget)
        self.predictedLabel.setGeometry(QtCore.QRect(40, 400, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.predictedLabel.setFont(font)
        self.predictedLabel.setObjectName("predictedLabel")
        
        
        self.probabilityLabel = QtWidgets.QLabel(self.centralwidget)
        self.probabilityLabel.setGeometry(QtCore.QRect(40, 450, 361, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.probabilityLabel.setFont(font)
        self.probabilityLabel.setObjectName("probabilityLabel")

        self.browseImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.browseImageBtn.setGeometry(QtCore.QRect(380, 20, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.browseImageBtn.setFont(font)
        self.browseImageBtn.setObjectName("browseImageBtn")
        self.browseImageBtn.clicked.connect(self.browseImage)

        self.predictBtn = QtWidgets.QPushButton(self.centralwidget)
        self.predictBtn.setGeometry(QtCore.QRect(380, 80, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(11)
        self.predictBtn.setFont(font)
        self.predictBtn.setObjectName("predictBtn")
        self.predictBtn.clicked.connect(self.prediction)

        
        GuiWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(GuiWindow)
        QtCore.QMetaObject.connectSlotsByName(GuiWindow)

  
    def browseImage(self):
        fm = QtWidgets.QFileDialog.getOpenFileName(None,"OpenFile")
        filename = fm[0]
        self.image = cv2.imread(filename)        
        self.InputImage.setPixmap(QtGui.QPixmap(filename))
        self.InputImage.setScaledContents(True)
    
    def prediction(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (image_size,image_size))
        self.image = np.array(self.image) / 255
        self.image = np.expand_dims(self.image, axis=0)
        print("Testing using Inception DCNN Model")        
        try:
            inc_pred = best_model.predict(self.image)
            probability = inc_pred[0]
            if probability[0] > 0.5: 
                self.predictedLabel.setText("Your Covid Test Result: COVID POSITIVE")
                self.probabilityLabel.setText("COVID-19 Positive with Probability: " + str('%.2f' %(probability[0]*100)+'%'))
            else:
                self.predictedLabel.setText("Your Covid Test Result: COVID NEGATIVE")
                self.probabilityLabel.setText("COVID-19 Negative with Probability: " + str('%.2f' %((1-probability[0])*100)+'%'))
            print("Testing completed")
        except:
            msgError = QtWidgets.QMessageBox()
            msgError.setIcon(QtWidgets.QMessageBox.Critical)
            msgError.setWindowTitle("Error")
            msgError.setText("Oops!! Error")
            msgError.exec_()


# In[37]:


# run the gui by calling the main function below.
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainScreen = QtWidgets.QMainWindow()
    obj = GUI_covid19_detector()
    obj.setupUi(MainScreen)
    MainScreen.show()
    sys.exit(app.exec_())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:




