import torch
import torch.nn as nn
import torch.optim as optim 
from torch.nn import functional as F 
import numpy as np
import pickle

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


print(torch.__version__)
""" torch.manual_seed(1)
torch.cuda.manual_seed(1) """

class SGN(nn.Module):
    def __init__(self,n_input = 432,n_output = 32):
        super(SGN,self).__init__()
        self.cnov1 = nn.Conv1d(1,30,10)
        self.cnov2 = nn.Conv1d(30,30,8)
        self.cnov3 = nn.Conv1d(30,40,6)
        self.cnov4 = nn.Conv1d(40,50,5)
        self.cnov5 = nn.Conv1d(50,50,5)

        self.gate_cnov1 = nn.Conv1d(1,30,10)
        self.gate_cnov2 = nn.Conv1d(30,30,8)
        self.gate_cnov3 = nn.Conv1d(30,40,6)
        self.gate_cnov4 = nn.Conv1d(40,50,5)
        self.gate_cnov5 = nn.Conv1d(50,50,5)

        self.n_fcl_input = n_input - 10 -8 -6 -5 -5 +5 #calculate the input unit of full connect layer

        self.Fcl_R1 = nn.Linear(self.n_fcl_input*50,1024)
        self.Fcl_R2 = nn.Linear(1024,n_output)
        self.Fcl_O1 = nn.Linear(self.n_fcl_input*50,1024)
        self.Fcl_O2 = nn.Linear(1024,n_output)

        self.n_output = n_output

    def forward(self,x):

        p = torch.relu(self.cnov1(x))
        p = torch.relu(self.cnov2(p))
        p = torch.relu(self.cnov3(p))
        p = torch.relu(self.cnov4(p))
        p = torch.relu(self.cnov5(p))
        
        p = p.reshape(len(p),self.n_fcl_input*50)
        p = torch.relu(self.Fcl_R1(p))
        p = self.Fcl_R2(p)

        o = torch.relu(self.gate_cnov1(x))
        o = torch.relu(self.gate_cnov2(o))
        o = torch.relu(self.gate_cnov3(o))
        o = torch.relu(self.gate_cnov4(o))
        o = torch.relu(self.gate_cnov5(o))

        o = o.reshape(len(o),self.n_fcl_input*50)
        o = torch.relu(self.Fcl_O1(o))      
        o = torch.sigmoid(self.Fcl_O2(o))

        #o = torch.softmax(self.Fcl_O2(o),dim = 1)

        #print(o.shape)
        p = p.reshape(len(p),self.n_output)
        o = o.reshape(len(o),self.n_output)
        p = p*o
        return p,o

    def save(self,filename):
        torch.save(self.state_dict(),filename)

    def load(self,filename):
        self.load_state_dict(torch.load(filename))

class SGN_sp(nn.Module):
    def __init__(self,n_input = 432,n_output = 32):
        super(SGN_sp,self).__init__()
        self.cnov1 = nn.Conv1d(1,5,10)
        self.cnov2 = nn.Conv1d(5,8,8)
        self.cnov3 = nn.Conv1d(8,10,6)
        self.cnov4 = nn.Conv1d(10,10,5)

        self.n_fcl_input = n_input - 10 -8 -6 -5 +4 #calculate the input unit of full connect layer

        self.Fcl_R1 = nn.Linear(self.n_fcl_input*10,128)
        self.Fcl_R2 = nn.Linear(128,n_output)
        self.Fcl_O1 = nn.Linear(self.n_fcl_input*10,128)
        self.Fcl_O2 = nn.Linear(128,n_output)

        self.n_output = n_output

    def forward(self,x):

        p = torch.relu(self.cnov1(x))
        p = torch.relu(self.cnov2(p))
        p = torch.relu(self.cnov3(p))
        p = torch.relu(self.cnov4(p))
        
        p = p.reshape(len(p),self.n_fcl_input*10)
        o = p

        p = torch.relu(self.Fcl_R1(p))
        p = torch.relu(self.Fcl_R2(p))


        o = torch.relu(self.Fcl_O1(o))
        """ o = torch.tanh(self.Fcl_O2(o))
        o = torch.relu(o) """
        #o = self.Fcl_O2(o)
        #o = F.softmax(self.Fcl_O2(o))
        o = torch.sigmoid(self.Fcl_O2(o))

        #print(o.shape)
        p = p.reshape(len(p),self.n_output)
        o = o.reshape(len(o),self.n_output)
        p = p*o
        return p,o

    def save(self,filename):
        torch.save(self.state_dict(),filename)

    def load(self,filename):
        self.load_state_dict(torch.load(filename))

class SGN_2point(nn.Module):
    def __init__(self,n_input = 432,n_output = 1):
        super(SGN_2point,self).__init__()

        self.cnov1 = nn.Conv1d(1,30,10)
        self.cnov2 = nn.Conv1d(30,40,8)
        self.cnov3 = nn.Conv1d(40,50,6)
        self.cnov4 = nn.Conv1d(50,50,5)
        self.cnov5 = nn.Conv1d(50,1,5)

        self.n_fcl_input = n_input - 10 -8 -6 -5 -5+5 #calculate the input unit of full connect layer

        self.Fcl_R1 = nn.Linear(self.n_fcl_input,512)
        self.Fcl_R2 = nn.Linear(512,n_output)
        self.Fcl_O1 = nn.Linear(self.n_fcl_input,512)
        self.Fcl_O2 = nn.Linear(512,n_output)

        self.n_output = n_output

    def forward(self,x):

        p = torch.relu(self.cnov1(x))
        p = torch.relu(self.cnov2(p))
        p = torch.relu(self.cnov3(p))
        p = torch.relu(self.cnov4(p))
        p = torch.relu(self.cnov5(p))
        
        o = p

        p = torch.relu(self.Fcl_R1(p))
        p = self.Fcl_R2(p)


        o = torch.relu(self.Fcl_O1(o))
        o = torch.tanh(self.Fcl_O2(o))
        o = torch.relu(o)
        #o = torch.softmax(self.Fcl_O2(o),dim = 1)

        #print(o.shape)
        p = p.reshape(len(p),self.n_output)
        o = o.reshape(len(o),self.n_output)
        return p,o

    def save(self,filename):
        torch.save(self.state_dict(),filename)

    def load(self,filename):
        self.load_state_dict(torch.load(filename))

def train_sgn(model,train_x,train_y,onstate):
    epoch_num = 500
    batch = 2000

    criterion_P = nn.MSELoss()
    #criterion_O = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    #optimizer = optim.SGD(model.parameters(),lr = 0.001)

    loss = 0
    train_x = train_x.reshape(len(train_x),1,len(train_x[1]))
    train_y = train_y.reshape(len(train_y),1,len(train_y[1]))


    for epoch in range(epoch_num):
        for i in range(int(len(train_x)/batch)):
            x = train_x[i*batch : (i+1)*batch]
            y = train_y[i*batch : (i+1)*batch]
            on_real = onstate[i*batch : (i+1)*batch]
            #print('model on_real shape: ',on_real.shape)
            power,on = model(x)

            #print('model output shape: ',on.shape)

            loss_on = 0
                        
            temp =  on_real*torch.log(on + 1e-10) + (1-on_real)*torch.log(1-on + 1e-10)               
            
            loss_on = -torch.sum(torch.sum(temp))/(batch*len(on[1]))
            """ for i in range(len(on[1])):
                temp =  nn.BCELoss(nn.Sigmoid(on[:,i]),on_real[:,i])               
                loss_on += temp
            loss_on = loss_on/32 """
            #loss_on = criterion_P(on, on_real)
            loss_output = criterion_P(power,y)  
            
            loss = loss_output + loss_on
            
            #on_real = on_real.squeeze()
            #print('model on_real shape: ',on_real.shape)
            #loss_on = criterion_P(on, on_real)
            #loss_output = criterion_P(power*on,y)  
            
            #loss = criterion_P(power*on,y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%100 == 0:            
                model.eval()
                #pre= model(train_data[domain])
                #print(pre)                
                #print('SGN epoch:{}  -- LOSS :  {}'.format(epoch+1, loss))
                """ print(on[1])
                print(on_real[1])
                print(temp[1]) """
                print('SGN epoch:{}  -- LOSS :  {} -- loss on :{} -- loss power: {}'.format(epoch+1, loss,loss_on,loss_output))

def train_sgn_sp(model,train_x,train_y,onstate):
    epoch_num = 2000
    batch = 2000

    criterion_P = nn.MSELoss()
    criterion_O = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    #optimizer = optim.SGD(model.parameters(),lr = 0.001)

    loss = 0
    train_x = train_x.reshape(len(train_x),1,len(train_x[1]))
    train_y = train_y.reshape(len(train_y),1,len(train_y[1]))
    #onstate = torch.sum(onstate,dim = 1)

    for epoch in range(epoch_num):  
        for i in range(int(len(train_x)/batch)):
            x = train_x[i*batch : (i+1)*batch]
            y = train_y[i*batch : (i+1)*batch]
            on_real = onstate[i*batch : (i+1)*batch]
            #print('model on_real shape: ',on_real.shape)
            power,on = model(x)
            #print('model output shape: ',on.shape)
            #on_p = torch.sum(on,dim = 1)
            #on_real = on_real.squeeze()
            #print('model on_real shape: ',on_real.shape)
            loss_on = 0                      
            temp =  on_real*torch.log(on + 1e-10) + (1-on_real)*torch.log(1-on + 1e-10)                     
            loss_on = -torch.sum(torch.sum(temp))/(batch*len(on[1]))

            loss_output = criterion_P(power,y)  
            
            loss = loss_output + loss_on

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%100 == 0:            
            model.eval()
            #pre= model(train_data[domain])
            #print(pre)                
            print('SGN-SP epoch:{}  -- LOSS :  {} -- loss on :{} -- loss power: {}'.format(epoch+1, loss, loss_on, loss_output))
            #print('SGN-sp epoch:{}  -- loss power: {}'.format(epoch+1, loss_output))

def train_sgn_2point(model,train_x,train_y,onstate):
    epoch_num = 1000
    batch = 2000

    criterion_P = nn.MSELoss()
    #criterion_O = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)
    #optimizer = optim.SGD(model.parameters(),lr = 0.001)

    loss = 0
    train_x = train_x.reshape(len(train_x),1,len(train_x[1]))
    train_y = train_y.reshape(len(train_y),1,len(train_y[1]))


    for epoch in range(epoch_num):
        for i in range(int(len(train_x)/batch)):
            x = train_x[i*batch : (i+1)*batch]
            y = train_y[i*batch : (i+1)*batch]
            on_real = onstate[i*batch : (i+1)*batch]
            #print('model on_real shape: ',on_real.shape)
            power,on = model(x)

            #print('model output shape: ',on.shape)

            loss_on = 0
                        
            temp =  on_real*torch.log(on + 1e-10) + (1-on_real)*torch.log(1-on + 1e-10)               
            
            loss_on = -torch.sum(torch.sum(temp))/(batch*len(on[1]))

            loss_output = criterion_P(power*on,y)  
            
            loss = loss_output + loss_on

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)%100 == 0:            
                model.eval()
                #pre= model(train_data[domain])
                #print(pre)                
                #print('SGN epoch:{}  -- LOSS :  {}'.format(epoch+1, loss))
                """ print(on[1])
                print(on_real[1])
                print(temp[1]) """
                print('SGN 2point epoch:{}  -- LOSS :  {} -- loss on :{} -- loss power: {}'.format(epoch+1, loss,loss_on,loss_output))

def plot_sgn(model,test_x,test_y):
    import matplotlib.pyplot as plt 
    import random
    base_folder = './expresult/sgn'


    test_x = test_x.reshape(len(test_x),1,len(test_x[1]))
    

    i = random.randint(1,len(test_y))
    """ while np.sum(test_y[i]) < 2:
        i = random.randint(1,len(test_y)) """

    power, on = model(test_x)
    y_pre = power
    y_pre = y_pre.to('cpu')
    y_pre = y_pre.detach().numpy().reshape(-1)
    y = test_y.reshape(-1)

    on_pre = on.to('cpu').detach().numpy().reshape(-1)
    on_pre = on_pre
    power = y_pre*1000 +700
    y_pre = y_pre*1000 +700
    y = y*1000 +700
    plt.figure(figsize=(10,5), constrained_layout=True)
    plt.plot(np.arange(len(y_pre)),y_pre,label='predict power') 
    plt.plot(np.arange(len(y_pre)),y,label='real power') 
    plt.plot(np.arange(len(y_pre)),on_pre*3000,label='gate ') 
    #plt.plot(np.arange(len(y_pre)),power,label='power out ') 
    

    plt.xlabel(u"epoch (s)",fontsize=20)
    plt.ylabel(u"loss",fontsize=20)
    plt.title('test curve {}'.format(i),fontsize=20)
    plt.legend()
    plt.savefig(base_folder  + 'test curve_{}'.format(i) +'.png',format='png')

def get_data_train(filename,data_size):
    alldatax = pickle.load(open(filename +'/SGN_TrainDatax.pk','rb'))
    alldatay = pickle.load(open(filename +'/SGN_TrainDatay.pk','rb'))

    alldatax = alldatax.reshape(-1)
    alldatay = alldatay.reshape(-1)
    npx = []
    npy = []
    w = 200
    widouw_size = 32
    all_data_number = int((len(alldatax)-2*w)/( widouw_size))
    #print(int(len(alldatax)/(2*w + widouw_size)))

    if data_size > all_data_number : data_size = all_data_number
    count_off = data_size/3
    for i in range(all_data_number):
        if np.sum(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)]) > 100 :  
            npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
            npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])
        elif count_off>0:
            npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
            npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])
            count_off -=1


    npx = np.array(npx[:data_size])
    npy = np.array(npy[:data_size])
    
    onstate = np.zeros(npy.shape)
    onstate[npy > 700 ] = 1

    return npx,npy, onstate

def get_data_test(filename,data_size):
    alldatax = pickle.load(open(filename +'/SGN_TestDatax.pk','rb'))
    alldatay = pickle.load(open(filename +'/SGN_TestDatay.pk','rb'))

    alldatax = alldatax.reshape(-1)
    alldatay = alldatay.reshape(-1)
    npx = []
    npy = []
    w = 200
    widouw_size = 32
    all_data_number = int((len(alldatax)-2*w)/( widouw_size))
    #print(int(len(alldatax)/(2*w + widouw_size)))

    if data_size > all_data_number : data_size = all_data_number
    count_off = data_size/5
    for i in range(all_data_number):
        if np.sum(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)]) > 100 :  
            npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
            npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])           
        elif count_off>0:
            npx.append(alldatax[widouw_size*i : 2*w + widouw_size*(i+1)])
            npy.append(alldatay[w+ widouw_size*i : w+ widouw_size*(i+1)])
            count_off -=1


    npx = np.array(npx[:data_size])
    npy = np.array(npy[:data_size])

    return npx,npy
        
    """ plt.figure(figsize=(10,5), constrained_layout=True)
    pl.plot(np.arange(len(losses))*10,losses) 
    pl.xlabel(u"epoch (s)",fontsize=20)
    pl.ylabel(u"loss",fontsize=20)
    plt.title('sub_{}'.format(domain),fontsize=20)
    plt.savefig(base_folder  + 'train losses of sub_{}'.format(domain) +'.jpg',format='jpg') """

def data_nomalize(train_x,train_y):
    npx = (train_x - 700)/1000
    npy = (train_y - 700)/1000

    return npx,npy

def evalate(model,test_x, test_y):
    test_x = test_x.reshape(len(test_x),1,len(test_x[1]))
    power,on = model(test_x)
    y_pre = power
    y_pre = y_pre.to('cpu').detach().numpy()
    mae = 0
    for i in range(len(test_y)):    
        mae += abs(y_pre[i] - test_y[i])[0]*1000 + 700
    print('MAE=  ',mae/len(test_y))



""" --------------------------------------------------------------------------------- """

filename = './data'
model_file = './net/sgn_1.pth'

model_file_sp = './net/sgn_sp.pth'

train_x ,train_y ,onstate = get_data_train(filename,2000)
test_x, test_y = get_data_test(filename,300)

#print(train_x.shape)

rain_x ,train_y = data_nomalize(train_x,train_y)
test_x, test_y = data_nomalize(test_x, test_y)

train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
#onstate = torch.tensor(onstate, dtype=torch.long).to(device)
onstate = torch.tensor(onstate, dtype=torch.float32).to(device)
test_x = torch.tensor(test_x, dtype=torch.float32).to(device)

print('train start')
""" model_sgn = SGN(432,32).to(device)
model_sgn.load(model_file)
#train_sgn(model_sgn,train_x,train_y,onstate)
#model_sgn.save(model_file) """

model_sgn_sp = SGN_sp(432,32).to(device)
model_sgn_sp.load(model_file_sp)
#train_sgn_sp(model_sgn_sp,train_x,train_y,onstate)
model_sgn_sp.save(model_file_sp)

######plot
#evalate(model_sgn,test_x,test_y)
#plot_sgn(model_sgn,test_x[100:120],test_y[100:120])
#plot_sgn(model_sgn,test_x[55:75],test_y[55:75])
plot_sgn_new(model_sgn_sp,test_x[55:75],test_y[55:75])
""" evalate(model_sgn_sp,test_x,test_y)
#plot_sgn(model_sgn,test_x[100:120],test_y[100:120])
plot_sgn(model_sgn_sp,test_x[55:75],test_y[55:75]) """
