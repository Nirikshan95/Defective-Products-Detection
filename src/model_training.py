import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from preprocessing import data

dt=data()
dt.get_data()
model=Sequential([
    
    Conv2D(32,(3,3),input_shape=(128,128,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(64,(3,3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    
    Dense(60,activation='relu'),
    Dropout(0.4),
    
    Dense(20,activation='relu'),
    Dropout(0.4),
    
    Dense(len(dt.classes),activation='softmax')
    
]
)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(dt.scaled_train,validation_data=dt.scaled_test,epochs=10,batch_size=60)