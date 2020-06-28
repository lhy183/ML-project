from net import seq2SeqNet
from net import seq2PointNet
from net import daeNet
import pickle
import matplotlib.pyplot as plt
import numpy as np

WINDOW_LENGTH = 599
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
app = "microwave"
netname = "seq2point"
Mean = config[app]["Mean"]
StandardDeviation = config[app]["StandardDeviation"]
if __name__ ==  "__main__":
    # alldatax = pickle.load(open('./data/alltestDatax.pk','rb'))
    # alldatay = pickle.load(open('./data/alltestDatay.pk','rb'))
    # seq2seqNet = seq2SeqNet('./net/seq2seqKettle.h5')
    # #daeNet = daeNet("daekettle.h5")
    # rng = np.random.RandomState(202)
    
    # l = len(alldatax)
    # for _ in range(100):
    #     index =  rng.randint(0,l)
    #     x = alldatax[index]
    #     y = alldatay[index]
    #     x = (x-Mean)/StandardDeviation
    #     y = (y-Mean)/StandardDeviation
    #     tx = np.array([x.reshape(x.shape[0],1)])

    #     y_pre_seq = seq2seqNet.mymodel.predict(tx)[0]
    #     # y_pre_dae = daeNet.mymodel.predict(tx)[0]

    #     x = x*StandardDeviation+Mean
    #     y = y*StandardDeviation+Mean
    #     y_pre_seq = y_pre_seq*StandardDeviation+Mean
    #     #y_pre_dae = y_pre_dae*1000+700
    #     plt_x = np.linspace(1,WINDOW_LENGTH,WINDOW_LENGTH) * 6
    #     fig = plt.figure(figsize=(12.8,9.6))
    #     #plt.subplot(2,1,1)
    #     plt.plot(plt_x,x,label='total power')
    #     plt.plot(plt_x,y,label='real fridge power')
    #     plt.plot(plt_x,y_pre_seq,label='predicted fridge power')
    #     plt.xlabel('Time (second)')
    #     plt.ylabel('Power (Watt)')
    #     plt.title("Seq2Seq MAE = "+str((abs(y_pre_seq-y)).sum()/WINDOW_LENGTH))
    #     plt.legend()

    #     # plt.subplot(2,1,2)
    #     # plt.plot(plt_x,x,label='total power')
    #     # plt.plot(plt_x,y,label='real kettle power')
    #     # plt.plot(plt_x,y_pre_dae,label='predicted kettle power')
    #     # plt.xlabel('Time (second)')
    #     # plt.ylabel('Power (Watt)')
    #     # plt.title("DAE MAE = "+str((abs(y_pre_dae-y)).sum()/WINDOW_LENGTH))
    #     # plt.legend()

    #     plt.savefig("./Result/"+str(_)+".png")
    #     plt.close(_)

    seq2pointNet = seq2SeqNet('./net/'+app+'/seq2pointMicrowave.h5')
    alldatax = pickle.load(open('./data/'+app+'/seq2pointTestDatax.pk','rb'))
    alldatay = pickle.load(open('./data/'+app+'/seq2pointTestDatay.pk','rb'))

    for _ in range(100):
        x = alldatax[_*WINDOW_LENGTH:_*WINDOW_LENGTH+WINDOW_LENGTH]
        y = alldatay[_*WINDOW_LENGTH:_*WINDOW_LENGTH+WINDOW_LENGTH]
        tx = []
        total = []
        for i in range(WINDOW_LENGTH):
            tx.append((x[i] - Mean)/StandardDeviation)
            total.append(x[i][int(WINDOW_LENGTH/2)])
        tx = np.array(tx)
        tx = tx.reshape(x.shape[0],x.shape[1],1)
        plt_x = np.linspace(1,WINDOW_LENGTH,WINDOW_LENGTH) * 6
        y_pre = seq2pointNet.mymodel.predict(tx)
        y_pre = y_pre * StandardDeviation + Mean
        fig = plt.figure(figsize=(12.8,9.6))
        plt.plot(plt_x,total,label='total power')
        plt.plot(plt_x,y,label='real '+app+' power')
        plt.plot(plt_x,y_pre,label='predicted '+app+ ' power')
        plt.xlabel('Time (second)')
        plt.ylabel('Power (Watt)')
        Sum = 0
        for i in range(len(y)):
            Sum = Sum + abs(y[i]-y_pre[i])[0]
        Sum = Sum / WINDOW_LENGTH
        plt.title(netname+" MAE = "+format(Sum, '.2f'))
        plt.legend()
        plt.savefig("./Result/"+app+'/'+netname+'/'+str(_)+".png")
        plt.close(_)


        