import tensorflow as tf
import json
import os
import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Activation
from preprocessing import data

dt=data()
dt.get_data()
model=Sequential([
    
    Conv2D(32,(3,3),input_shape=config.INPUT_SHAPE),
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
    
    Dense(len(dt.classes),activation=config.ACTIVATION)
]
)

model.compile(optimizer=config.OPTIMIZER ,loss=config.LOSS,metrics=config.METRICS)
history=model.fit(dt.scaled_train,validation_data=dt.scaled_test,epochs=config.EPOCHS,batch_size=config.BATCH_SIZE)

# model JSON
model_archi=model.to_json()

with open(config.MODEL_ARCHITECTURE,"w") as file:
    json.dump(model_archi,file)

#save trained model
os.makedirs(os.path.dirname(config.MODEL_PATH),exist_ok=True)
model.save(config.MODEL_PATH)

#model history
with open(config.MODEL_HISTORY_PATH,"w") as hist:
    json.dump(history.history,hist)