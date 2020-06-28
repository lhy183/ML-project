
#from dataPreprocess import myData
import numpy as np
import pickle
import matplotlib.pyplot as plt 


filename = './data'
def get_data_train(filename):
    alldatax = pickle.load(open(filename +'/SGN_TrainDatax.pk','rb'))
    alldatay = pickle.load(open(filename +'/SGN_TrainDatay.pk','rb'))

    alldatax = alldatax.reshape(-1)
    alldatay = alldatay.reshape(-1)
    npx = []
    npy = []
    w = 200
    widouw_size = 32

    print(int(len(alldatax)/(2*w + widouw_size)))
    for i in range(int(len(alldatax)/(2*w + widouw_size))):
        npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
        npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])

    npx = np.array(npx)
    npy = np.array(npy)

    return npx,npy

def get_data_test(filename):
    alldatax = pickle.load(open(filename +'/SGN_TestDatax.pk','rb'))
    alldatay = pickle.load(open(filename +'/SGN_TestDatay.pk','rb'))

    alldatax = alldatax.reshape(-1)
    alldatay = alldatay.reshape(-1)
    npx = []
    npy = []
    w = 200
    widouw_size = 32

    #print(int(len(alldatax)/(2*w + widouw_size)))
    for i in range(int(len(alldatax)/(2*w + widouw_size))):
        npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
        npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])

    npx = np.array(npx)
    npy = np.array(npy)

    return npx,npy








""" dataset = myData(trainBuildings = [1,3,4,5],testBuildings = [2],applications = ['fridge','kettle','dish washer','microwave','washing machine'],targetapplication = 'kettle')

#dataset.getTrainDataForSGN()
dataset.getTestDataForSGN() """

""" alldatax = pickle.load(open('./data/alltrainDatax.pk','rb'))
alldatay = pickle.load(open('./data/alltrainDatay.pk','rb')) """

""" alldatax = pickle.load(open('./data/alltestDatax.pk','rb'))
alldatay = pickle.load(open('./data/alltestDatay.pk','rb')) """

""" alldatax = pickle.load(open('./data/SGN_TrainDatax.pk','rb'))
alldatay = pickle.load(open('./data/SGN_TrainDatay.pk','rb')) """

alldatax = pickle.load(open('./data/SGN_TestDatax.pk','rb'))
alldatay = pickle.load(open('./data/SGN_TestDatay.pk','rb'))

npx,npy = get_data_test(filename)
npy = npy.reshape(-1)
base_folder = './expresult/sgn'
plt.figure(figsize=(10,5), constrained_layout=True)
plt.plot(np.arange(len(npy)),npy,color ='r') 



plt.xlabel(u"time (s)",fontsize=20)
plt.ylabel(u"power",fontsize=20)
plt.title('keetet ',fontsize=20)
plt.savefig(base_folder  + 'test_y curve'+'.png',format='png')

""" print(np.argmax(np.max(npy, axis = 1)))
print(np.max(npy, axis = 1)[1514]) """

""" npy_o = np.zeros(npy.shape)
npy_o[npy>0] = 1 

print(np.max(npy_o, axis = 1)[1514])
print(npy_o.shape) """
""" print(alldatax.shape)
print(alldatay.shape)
 """




