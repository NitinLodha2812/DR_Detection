import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, concatenate


### Define the standard CNN ###

'''Function to define a standard CNN model'''
def build_model(input_size,nb_classes):
  inputs = Input(input_size)

  conv1 = Conv2D(8, 3, input_shape=input_size, activation='relu', padding='same',
                         kernel_initializer='he_normal')(inputs)
  conv2 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
  merge1 = concatenate([conv1, conv2], axis=3)

  mp1 = MaxPooling2D((2,2))(merge1)
  drop1 = Dropout(0.2)(mp1)

  conv3 = Conv2D(8, 3, activation='relu', padding='same',
                        kernel_initializer='he_normal')(drop1)
  conv4 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
  merge2 = concatenate([conv3, conv4], axis=3)

  mp2 = MaxPooling2D((2, 2))(merge2)
  drop2 = Dropout(0.35)(mp2)
  conv5 = Conv2D(16, 3, activation='relu', padding='same',
                        kernel_initializer='he_normal')(drop2)
  conv6 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
  merge3 = concatenate([conv5, conv6], axis=3)

  mp3 = MaxPooling2D((2, 2))(merge3)

  conv7 = Conv2D(16, 3, activation='relu', padding='same',
                        kernel_initializer='he_normal')(mp3)
  conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
  merge3 = concatenate([conv7, conv8], axis=3)

  conv9 = Conv2D(32, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge3)
  conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
  merge4 = concatenate([conv9, conv10], axis=3)

  mp4 = MaxPooling2D((2, 2))(merge4)

  conv11 = Conv2D(32, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(mp4)
  conv12 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
  merge5 = concatenate([conv11, conv12], axis=3)

  conv13 = Conv2D(64, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge5)
  conv14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
  merge6 = concatenate([conv13, conv14], axis=3)

  conv15 = Conv2D(64, 3, activation='relu', padding='same',
                  kernel_initializer='he_normal')(merge6)
  conv16 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)
  merge7 = concatenate([conv15, conv16], axis=3)

  mp5 = MaxPooling2D((2, 2))(merge7)

  drop4 = Dropout(0.45)(mp5)
  flat = Flatten()(drop4)

  dense1 = Dense(nb_classes, activation='softmax')(flat)

  return Model(inputs, dense1)
