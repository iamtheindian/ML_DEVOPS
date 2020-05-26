#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing useful libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Dense, Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.models import Model  ,Sequential


# In[2]:


img_rows,img_cols=(64,64)


# In[3]:


#now creation of model
model=Sequential()
model.add(          Convolution2D(filters=32,
                                  kernel_size=(3,3),
                                  activation='relu',
                                  input_shape=(img_rows,img_cols,3)
         ))
print(model.summary())


# In[4]:


#now adding more layers in this model
model.add(MaxPooling2D())
model.add(Convolution2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
print(model.summary())


# In[5]:


model.add(Dense(48,activation='relu',name='First MOdel Dense'))
#now output layer 
model.add(Dense(1,activation='sigmoid',name='Output Dense'))


# In[6]:


model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[7]:


print(model.summary())


# In[8]:


#data genrator
train_datagen=ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_data_dir = '/workstation/cell_images/'
validation_data_dir = '/workstation/valid/'
train_batchsize = 10
val_batchsize = 8
train_generator=train_datagen.flow_from_directory(train_data_dir,
                                                  target_size=(img_rows,img_cols),
                                                  batch_size=train_batchsize,
                                                  class_mode='binary'
                                                 )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        class_mode='binary',
        batch_size=val_batchsize,
        shuffle=False)


# In[9]:

#checkpoints and data training
checkpoint = ModelCheckpoint("/workstation/maleria_detection.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)
callbacks = [earlystop, checkpoint]
print("MODEL CREATION STARTED")
cl_cat=train_generator.class_indices
print(cl_cat)
print(validation_generator.class_indices)
history=model.fit_generator(train_generator,
                    epochs=2,
                    validation_data=validation_generator,
                    callbacks=callbacks
)
model.save('/workstation/maleria_detection.h5')


# In[20]:


print(history.history['val_accuracy'][-1])


# In[21]:


acc=int(history.history['val_accuracy'][-1]*100)


# In[24]:


with open('/workstation/des_acc.txt','w+') as f:
    f.write(str(acc)+'\n')


# In[25]:


print("all task completed")


# In[ ]:




