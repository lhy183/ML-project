from net import seq2SeqNet
from net import seq2PointNet
from net import seq2PointAttNet
from net import daeNet
from dataPreprocess import myData
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

config = {
    "fridge":{
            "Mean" : 200,
            "StandardDeviation" : 400
        },
    "kettle":{
            "Mean" : 700,
            "StandardDeviation" : 1000
        },
    "microwave":{
            "Mean" : 500,
            "StandardDeviation" : 800
        },
    "dishwasher":{
            "Mean" : 700,
            "StandardDeviation" : 1000
        },
    "washingmachine":{
            "Mean" : 400,
            "StandardDeviation" : 700
        },
    }
Mean = config["microwave"]["Mean"]
StandardDeviation = config["microwave"]["StandardDeviation"]
if __name__ ==  "__main__":

    alldatax = pickle.load(open('./data/microwave/seq2pointTrainDatax.pk','rb'))
    alldatay = pickle.load(open('./data/microwave/seq2pointTrainDatay.pk','rb'))
    testdatax = pickle.load(open('./data/microwave/seq2pointTestDatax.pk','rb'))
    testdatay = pickle.load(open('./data/microwave/seq2pointTestDatay.pk','rb'))
    syntheticdatax = pickle.load(open('./data/microwave/syntheticTrainDatax.pk','rb'))
    syntheticdatay = pickle.load(open('./data/microwave/syntheticTrainDatay.pk','rb'))

    alldatax = np.concatenate([alldatax,syntheticdatax])
    alldatay = np.concatenate([alldatay,syntheticdatay])
    
    rng = np.random.RandomState(0)
    state = rng.get_state()
    rng.shuffle(alldatax)
    rng.set_state(state)
    rng.shuffle(alldatay)

    # dataset = myData(trainBuildings = [1,3,4,5],testBuildings = [2],applications = ['fridge','kettle','dish washer','microwave','washing machine'],targetapplication = 'kettle')
    # alldatax,alldatay = dataset.getTrainDataForSeq2Point(1000)
    # testdatax,testdatay = dataset.getTestDataForSeq2Point(200)

    print(alldatax.shape)
    for i in range(len(alldatax)):          
        alldatax[i] = (alldatax[i] - Mean) / StandardDeviation
        alldatay[i] = (alldatay[i] - Mean) / StandardDeviation
    for i in range(len(testdatax)):
        testdatax[i] = (testdatax[i] - Mean) / StandardDeviation
        testdatay[i] = (testdatay[i] - Mean) / StandardDeviation

    # x1,x2,y1,y2 = train_test_split(alldatax,alldatay,test_size=0.3)
    # pickle.dump(x2,open('./data/testDatax.pk', 'wb+'))
    # pickle.dump(y2,open('./data/testDatay.pk', 'wb+'))

    # y1 = np.array(y1)
    # y2 = np.array(y2)
    # x1 = np.array(x1)
    # print(x1.shape)
    # x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
    # x2 = np.array(x2)
    # x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
    alldatax = alldatax.reshape(alldatax.shape[0],alldatax.shape[1],1)

    testdatax = testdatax.reshape(testdatax.shape[0],testdatax.shape[1],1)


    net = seq2PointNet()
    net.fit(alldatax,alldatay,epochs=20,validation_split=0.2,batch_size=512)
    net.evaluate(testdatax, testdatay)
    net.save("./net/microwave/seq2pointMicrowave_s.h5")