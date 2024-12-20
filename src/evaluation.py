import config
from preprocessing import data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report

model=load_model(config.MODEL_PATH)

dt=data()
dt.get_data()

model.evaluate(dt.scaled_test)

# got 93% accuracy on test data
y_true=[]
y_pred=[]
for image,label in dt.scaled_test:
    pred=model.predict(image)
    y_pred.extend(np.argmax(pred,axis=1))
    y_true.extend(label.numpy())
    
#confution matrix
sns.heatmap(confusion_matrix(y_true,y_pred),cmap='Greens',square=False)
os.makedirs(os.path.dirname(config.PLOTS_PATH),exist_ok=True)
plt.savefig(config.PLOTS_PATH+'/confusion_matrix.png')
plt.show()

#classification report 
print('classification_report :\n',classification_report(y_true,y_pred))