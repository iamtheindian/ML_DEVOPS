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
from keras.models import Model  ,Sequential,load_model

from random import randint 


# In[2]:


model=load_model('maleria_detection.h5')


# In[3]:


img_rows,img_cols=(64,64)


# In[4]:



def cnn_filter(model,unt):
    lyr=Dense(unt,activation='relu',name='adddense'+str(unt))(model.layers[-2].output)
    op_layer=Dense(1,activation='sigmoid')(lyr)
    return Model(inputs=model.input,outputs=[op_layer])


# In[5]:


model.summary()


# In[6]:


with open('des_acc.txt','r+') as f:
    data=f.read()
    acc=int(data)
    print(acc)


# In[7]:


#data genrator
train_datagen=ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_data_dir = '../cell_images/cell_images/'
validation_data_dir = '../cell_images/valid/'
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

#checkpoints and data training
checkpoint = ModelCheckpoint("maleria_detection_tweak.h5 ",
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


# In[9]:



count=1
while acc<93 and count<5:
    no_dense=randint(30 ,100)
    new_model=cnn_filter(model, no_dense)
    print(new_model.summary())
    new_model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    history=new_model.fit_generator(train_generator,
                    epochs=1,
                    validation_data=validation_generator,
                    callbacks=callbacks
)
    acc=int(history.history['val_accuracy'][-1]*100)
    count+=1
    model=new_model
model.save('maleria_detection_tweak.h5')
print(acc,new_model.summary())   


# In[ ]:




