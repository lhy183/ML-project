import tensorflow as tf
from tensorflow import keras

WINDOW_LENGTH = 599
class seq2SeqNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(WINDOW_LENGTH,1))
            conv1 = keras.layers.Conv1D(30,10,1,activation='relu')(myinput)
            conv2 = keras.layers.Conv1D(30,8,1,activation='relu')(conv1)
            conv3 = keras.layers.Conv1D(40,6,1,activation='relu')(conv2)
            conv4 = keras.layers.Conv1D(50,5,1,activation='relu')(conv3)
            conv5 = keras.layers.Conv1D(50,5,1,activation='relu')(conv4)
            flat = keras.layers.Flatten()(conv5)
            den1 = keras.layers.Dense(1024,activation='relu')(flat)
            output = keras.layers.Dense(WINDOW_LENGTH,activation='linear')(den1)
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mean_squared_error",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='myNet.h5'):
        self.mymodel.save(filename)

class daeNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(WINDOW_LENGTH,1))
            conv1 = keras.layers.Conv1D(8,4,1,activation='linear')(myinput)
            flat1 = keras.layers.Flatten()(conv1)
            den1 = keras.layers.Dense((WINDOW_LENGTH-3)*8,activation='relu')(flat1)
            den2 = keras.layers.Dense(128,activation='relu')(den1)
            den3 = keras.layers.Dense((WINDOW_LENGTH-3)*8,activation='relu')(den2)
            reshape = keras.layers.Reshape((2368,1))(den3)
            conv2 = keras.layers.Conv1D(1,4,1,activation='linear')(reshape)
            flat2 = keras.layers.Flatten()(conv2)
            output = keras.layers.Dense(WINDOW_LENGTH,activation='linear')(flat2)
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mean_squared_error",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='daenet.h5'):
        self.mymodel.save(filename)

class seq2PointNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(599,1))
            conv1 = keras.layers.Conv1D(30,10,1,activation='relu',padding='same')(myinput)
            conv2 = keras.layers.Conv1D(30,8,1,activation='relu',padding='same')(conv1)
            conv3 = keras.layers.Conv1D(40,6,1,activation='relu',padding='same')(conv2)
            conv4 = keras.layers.Conv1D(50,5,1,activation='relu',padding='same')(conv3)
            conv5 = keras.layers.Conv1D(50,5,1,activation='relu',padding='same')(conv4)
            flat = keras.layers.Flatten()(conv5)
            den1 = keras.layers.Dense(1024,activation='relu')(flat)
            output = keras.layers.Dense(1,activation='linear')(den1)
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mean_squared_error",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='seq2pointNet.h5'):
        self.mymodel.save(filename)


class seq2PointAttNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(599,1))
            conv1 = keras.layers.Conv1D(30,10,1,activation='relu')(myinput)
            conv2 = keras.layers.Conv1D(30,8,1,activation='relu')(conv1)
            conv3 = keras.layers.Conv1D(40,6,1,activation='relu')(conv2)
            conv4 = keras.layers.Conv1D(50,5,1,activation='relu')(conv3)
            conv5 = keras.layers.Conv1D(50,5,1,activation='relu')(conv4)
            flat = keras.layers.Flatten()(conv5)
            dropout = keras.layers.Dropout(0.3)(flat)
            den1 = keras.layers.Dense(1024,activation='relu',name='feature')(dropout)
            att = keras.layers.Dense(1024,activation='sigmoid')(den1)
            # att = keras.layers.Dense(1024,activation='softmax',name='alpha')(att)
            att = keras.layers.Multiply()([den1,att])
            den2 = keras.layers.Dense(1024,activation='relu',kernel_regularizer = keras.regularizers.l2(0.01))(att)
            output = keras.layers.Dense(1,activation='linear')(den2)
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mse",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='seq2pointAttNet.h5'):
        self.mymodel.save(filename)

class seq2PointDNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(599,1))
            conv1 = keras.layers.Conv1D(50,10,1,activation='relu')(myinput)
            conv2 = keras.layers.Conv1D(60,10,1,activation='relu')(conv1)
            conv3 = keras.layers.Conv1D(70,8,1,activation='relu')(conv2)
            conv4 = keras.layers.Conv1D(70,8,1,activation='relu')(conv3)
            conv5 = keras.layers.Conv1D(70,8,1,activation='relu')(conv4)
            conv6 = keras.layers.Conv1D(80,6,1,activation='relu')(conv5)
            conv7 = keras.layers.Conv1D(80,6,1,activation='relu')(conv6)
            conv8 = keras.layers.Conv1D(100,5,1,activation='relu')(conv7)
            flat = keras.layers.Flatten()(conv8)
            dropout = keras.layers.Dropout(0.3)(flat)
            den1 = keras.layers.Dense(1024,activation='relu',name='feature')(dropout)
            den1 = keras.layers.Dense(1024,activation='relu')(den1)
            den1 = keras.layers.Dense(1024,activation='relu')(den1)
            output = keras.layers.Dense(1,activation='linear')(den1)
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mse",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='seq2pointDNet.h5'):
        self.mymodel.save(filename)

class seq2PointAttGRUNet():
    def __init__(self,filename = None):
        if filename == None:
            myinput = keras.layers.Input(shape=(599,1))
            conv1 = keras.layers.Conv1D(30,10,1,activation='relu')(myinput)
            conv2 = keras.layers.Conv1D(30,8,1,activation='relu')(conv1)
            conv3 = keras.layers.Conv1D(40,6,1,activation='relu')(conv2)
            conv4 = keras.layers.Conv1D(50,5,1,activation='relu')(conv3)
            conv5 = keras.layers.Conv1D(50,5,1,activation='relu')(conv4)
            flat = keras.layers.Flatten()(conv5)
            dropout = keras.layers.Dropout(0.3)(flat)
            feature1 = keras.layers.Dense(1024,activation='relu',name='feature1')(dropout)
            feature2 = keras.layers.GRU(1)(myinput)
            att = keras.layers.Dense(1024,activation='sigmoid')(feature1)
            # att = keras.layers.Dense(1024,activation='softmax',name='alpha')(att)
            attfeature1 = keras.layers.Multiply()([feature1,att])
            den2 = keras.layers.Dense(1024,activation='relu',kernel_regularizer = keras.regularizers.l2(0.01))(attfeature1)
            output1 = keras.layers.Dense(1,activation='linear')(den2)
            output = keras.layers.Dense(1,activation='linear')(keras.layers.Concatenate()([feature2,output1]))
            self.mymodel = keras.models.Model(inputs=myinput, outputs=output) 
            self.mymodel.compile(loss = "mse",metrics=[tf.keras.metrics.mae],optimizer="adam")
        else:
            self.mymodel = keras.models.load_model(filename)
        for i in self.mymodel.layers:
            print(i.output_shape)

    def fit(self,*args,**kwargs):
        self.mymodel.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        self.mymodel.predict(*args,**kwargs)

    def evaluate(self,*args,**kwargs):
        self.mymodel.evaluate(*args,**kwargs)
    
    def save(self,filename='seq2pointAttGRUNet.h5'):
        self.mymodel.save(filename)