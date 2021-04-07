import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score
import numpy as np

class Metrics(Callback):
    def __init__(self, validation):   
        super(Metrics, self).__init__()
        self.validation = validation    
            
        print('validation shape', len(self.validation[0]))
        
    def on_train_begin(self, logs={}):        
        self.val_f1s = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()        
    
        val_f1 = f1_score(val_targ, val_predict, average='macro')
        
        self.val_f1s.append(round(val_f1, 6))
 
        print(f' — val_f1: {val_f1}')