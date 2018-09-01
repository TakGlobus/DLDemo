# _*_ coding : utf-8 _*_

from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

# user settings
ndim1   = 94
nkernel1= 3
ndim2   = 44
nkernel2= 4
ndim3   = 20
nkernel3= 3

# model 
model=Sequential()
# 1
model.add(Conv2D(ndim1,nkernel1,input_shape(ndim1, ndim1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_zise=2)))
# 2
model.add(Conv2D(ndim2,nkernel2,input_shape(ndim2, ndim2)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_zise=2)))
# 3
model.add(Conv2D(ndim3,nkernel3,input_shape(ndim3, ndim3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_zise=2)))
# 4
model.add(Flatten())
model.add(Dense(100, activation='softmax'))
# compile
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=["accuracy"]  )

# learning 
history = model.fit(train_data, 
                    train_label,
                    batch_size=bsize,
                    epochs=epochs ,
                    verbose=1,
                    validation_data=(test_data, test_label)
                   )


# check score
score_model.evaluate(test_data, test_label, verbose1)
print()
print('test loss:', score[0]
print('test accuracy:', score[1]
