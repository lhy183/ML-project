import nilmtk
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.timeframe import TimeFrame
import os
import pandas as pd
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
os.environ.setdefault("NUMEXPR_MAX_THREADS","48")
class myData():
    def __init__(self, **config):
        if 'filename' not in config.keys():
            self.dataSet = nilmtk.DataSet("ukdale.h5")
        else:
            self.dataSet = nilmtk.DataSet(config['fileName'])

        if 'startTime' not in config.keys() or 'endTime' not in config.keys():
            self.dataSet.set_window("2012-11-01","2015-01-31")
        else:
            self.dataSet.set_window(config['startTime'],config['endTime'])

        if 'trainBuildings' not in config.keys():
            self.trainBuildings = [1,3,4,5]
        else:
            self.trainBuildings = config['trainBuildings']
        if 'testBuildings' not in config.keys():
            self.testBuildings = [2]
        else:
            self.testBuildings = config['testBuildings']

        if 'applications' not in config.keys():
            raise KeyError("please input applications")
        self.applications = config['applications']

        if 'targetapplication' not in config.keys():
            raise KeyError("please input targetapplication")
        self.targetApplication = config['targetapplication']

        if 'randSeed' not in config.keys():
            randSeed = 0
        else:
            randSeed = config['randSeed']
        
        self.otherApplications = [i for i in self.applications if i not in [self.targetApplication]]
        self.allBuildings = set(self.trainBuildings+self.testBuildings)
        self.window = 599
        self.inputSeqs = []
        self.targetSeqs = []
        self.rng = np.random.RandomState(randSeed)
        activationConfig = {
            'fridge':{
                'min_off_duration':     18,# 12 in paper here
                'min_on_duration':      60,
                'on_power_threshold':   50,
                'sample_period':        6,
            },
            'kettle':{
                'min_off_duration':     18, # 0 in paper here
                'min_on_duration':      12,
                'on_power_threshold':   2000,
                'sample_period':        6,
            },
            'washing machine':{
                'min_off_duration':     160,
                'min_on_duration':      1800,
                'on_power_threshold':   20,
                'sample_period':        6,
            },
            'microwave':{
                'min_off_duration':     30,
                'min_on_duration':      12,
                'on_power_threshold':   200,
                'sample_period':        6,
            },
            'dish washer':{
                'min_off_duration':     1800,
                'min_on_duration':      1800,
                'on_power_threshold':   10,
                'sample_period':        6,
            }
         }
        
        self.elecMains = {}
        self.goodSections = {}
        for building in self.allBuildings:
            self.goodSections[building] = self.dataSet.buildings[building].elec.mains().good_sections()
            self.elecMains[building] = self.dataSet.buildings[building].elec.mains().power_series_all_data(sample_period=6,sections=self.goodSections[building]).dropna()            

        self.numApp = {}
        self.elecApp = {}
        self.activationsApp = {}
        self.activationAppSections = {}
        for app in self.applications:
            self.elecApp[app] = {}
            self.activationsApp[app] = {}
            self.numApp[app] = 0
            self.activationAppSections[app] = {}
            for building in self.allBuildings:
                try:
                    self.elecApp[app][building] = self.dataSet.buildings[building].elec[app].power_series_all_data(sample_period=6).dropna()

                    self.activationsApp[app][building] = self.dataSet.buildings[building].elec[app].get_activations(**activationConfig[app])
                    self.activationsApp[app][building] = [activation.astype(np.float32) for activation in self.activationsApp[app][building]]
                    self.numApp[app] += len(self.activationsApp[app][building])
                    self.activationAppSections[app][building] = TimeFrameGroup()
                    for activation in self.activationsApp[app][building]:
                        self.activationAppSections[app][building].append(TimeFrame(activation.index[0],activation.index[-1]))
                except KeyError as exception:
                    logger.info(str(building) + " has no " + app + ". Full exception: {}".format(exception))
                    continue
        logger.info("Done loading NILMTK data.")

        for building in self.allBuildings:
            activationsToRemove = []
            try:
                activations = self.activationsApp[self.targetApplication][building]
                mains = self.elecMains[building]
                for i, activation in enumerate(activations):
                    activationDuration = (activation.index[-1] - activation.index[0])
                    start = (activation.index[0] - activationDuration)
                    end = (activation.index[-1] + activationDuration)
                    if start < mains.index[0] or end > mains.index[-1]:
                        activationsToRemove.append(i)
                    else:
                        mainsForAct = mains[start:end]
                        if not self._hasSufficientSamples(start,end,mainsForAct):
                            activationsToRemove.append(i)
                activationsToRemove.reverse()
                for i in activationsToRemove:
                    activations.pop(i)
                self.activationsApp[self.targetApplication][building] = activations
            except KeyError as exception:
                continue
        
        self.sectionsWithNoTarget = {}
        for building in self.allBuildings:
            try:
                activationsTarget = self.activationsApp[self.targetApplication][building]
                mainGoodSections = self.goodSections[building]
                mains = self.elecMains[building]
                gapsBetweenActivations = TimeFrameGroup() 
                prev = mains.index[0]
                for activation in activationsTarget:
                    try:
                        p2=prev
                        gapsBetweenActivations.append(TimeFrame(prev,activation.index[0]))
                        prev = activation.index[-1]
                        p1=activation.index[0]
                    except ValueError:
                        logger.debug("----------------------")
                        logger.debug(p1)
                        logger.debug(p2)
                        logger.debug(activation.index[0])
                        logger.debug(activation.index[-1])

                gapsBetweenActivations.append(TimeFrame(prev,mains.index[-1]))
                
                intersection = gapsBetweenActivations.intersection(mainGoodSections)
                intersection = intersection.remove_shorter_than(6*self.window)
                self.sectionsWithNoTarget[building]= intersection
            except KeyError:
                continue

    def getTrainDataForSeq2Point(self, num = None):
        trainInputSeqs = []
        trainTargetSeqs = []
        trainMainSeqs,trainAppSeqs = self._getRealData("Train",beforeStart = int(self.window/2),afterEnd = int(self.window/2))

        state = self.rng.get_state()
        self.rng.shuffle(trainMainSeqs)
        self.rng.set_state(state)
        self.rng.shuffle(trainAppSeqs)

        total = len(trainAppSeqs)
        if num != None and num < total:
            total = num
        trainMainSeqs = trainMainSeqs[:total]
        trainAppSeqs = trainAppSeqs[:total]

        for _ in range(total):
            for i in range(self.window):
                trainTargetSeqs.append(trainAppSeqs[_][i])
                trainInputSeqs.append(np.array(trainMainSeqs[_][i:i+self.window]))

        state = self.rng.get_state()
        self.rng.shuffle(trainTargetSeqs)
        self.rng.set_state(state)
        self.rng.shuffle(trainInputSeqs)

        npx,npy = np.array(trainInputSeqs),np.array(trainTargetSeqs)

        with open('./data/seq2pointTrainDatax.pk', 'wb+') as f:
            pickle.dump(npx,f)
            f.close()
        with open('./data/seq2pointTrainDatay.pk', 'wb+') as f:
            pickle.dump(npy,f)
            f.close()
        return npx,npy

    def getTestDataForSeq2Point(self, num = None):
        testInputSeqs = []
        testTargetSeqs = []
        testMainSeqs,testAppSeqs = self._getRealData("Test",beforeStart = int(self.window/2),afterEnd = int(self.window/2))
        state = self.rng.get_state()
        self.rng.shuffle(testMainSeqs)
        self.rng.set_state(state)
        self.rng.shuffle(testAppSeqs)

        total = len(testAppSeqs)
        if num != None and num < total:
            total = num
        testMainSeqs = testMainSeqs[:total]
        testAppSeqs = testAppSeqs[:total]

        for _ in range(total):
            for i in range(self.window):
                testTargetSeqs.append(testAppSeqs[_][i])
                testInputSeqs.append(np.array(testMainSeqs[_][i:i+self.window]))
        npx,npy = np.array(testInputSeqs),np.array(testTargetSeqs)
        with open('./data/seq2pointTestDatax.pk', 'wb+') as f:
            pickle.dump(npx,f)
            f.close()
        with open('./data/seq2pointTestDatay.pk', 'wb+') as f:
            pickle.dump(npy,f)
            f.close()
        return npx,npy
        
            
        
    def getTestData(self,num = None):
        testInputSeqs,testTargetSeqs = self._getRealData("Test")
        total = len(testInputSeqs)
        if num != None and num < total:
            total = num
        logger.debug("Number of test real data is "+str(total))
        npx,npy = np.array(testInputSeqs[:total]),np.array(testTargetSeqs[:total])
        with open('./data/alltestDatax.pk', 'wb+') as f:
            pickle.dump(npx,f)
            f.close()
        with open('./data/alltestDatay.pk', 'wb+') as f:
            pickle.dump(npy,f)
            f.close()
        return npx,npy

    def getTrainData(self,enableSynthetic = False,syntheticRatio = 0.5,distractorRatio = 0.25,targetRatio = 0.5):
        realInputSeqs,realTargetSeqs = self._getRealData("Train")
        realNum = len(realInputSeqs)
        self.inputSeqs.extend(realInputSeqs)
        self.targetSeqs.extend(realTargetSeqs)
        if enableSynthetic:
            syntheticInputSeqs,syntheticTargetSeqs = self._getSyntheticData(int(realNum*syntheticRatio/(1-syntheticRatio)),distractorRatio,targetRatio)
            self.inputSeqs.extend(syntheticInputSeqs)
            self.targetSeqs.extend(syntheticTargetSeqs)
        state = self.rng.get_state()
        self.rng.shuffle(self.inputSeqs)
        self.rng.set_state(state)
        self.rng.shuffle(self.targetSeqs)
        logger.debug("Number of train real data is "+str(realNum))
        npx = np.array(self.inputSeqs)
        npy = np.array(self.targetSeqs)
        with open('./data/alltrainDatax.pk', 'wb+') as f:
            pickle.dump(npx,f)
            f.close()
        with open('./data/alltrainDatay.pk', 'wb+') as f:
            pickle.dump(npy,f)
            f.close()
        return npx,npy


    def _getAtivation(self,application):
        buildings = [ i for i in set(self.activationsApp[application].keys()) if i in self.trainBuildings]
        num = len(buildings)
        building = buildings[self.rng.randint(0,num)]
        num = len(self.activationsApp[application][building])
        index = self.rng.randint(0,num)
        activation = self.activationsApp[application][building][index]
        return activation,building,index
            

    def _getSyntheticData(self,syntheticNum,distractorRatio = 0.5,targetRatio = 0.5):
        applicationNum = len(self.applications)
        inputSeqs = []
        targetSeqs = []
        for i in range(0,syntheticNum):
            inputSeq = np.zeros(self.window, dtype=np.float32)
            targetSeq = np.zeros(self.window, dtype=np.float32)
            if self.rng.binomial(n=1,p=targetRatio):
                targetActivation,building,index = self._getAtivation(self.targetApplication)
                activationLength = len(targetActivation)
                positioned_activation = self._positionActivation(targetActivation,self.targetApplication,building,self.window,index).values
                inputSeq = inputSeq + positioned_activation
                targetSeq = targetSeq + positioned_activation
            for application in self.otherApplications:
                if self.rng.binomial(n=1,p=distractorRatio):
                    try:
                        activation,building,index = self._getAtivation(application)
                        positioned_activation = self._positionActivation(activation,application,building,self.window,index,isReal = False).values
                        inputSeq = inputSeq + positioned_activation
                    except TypeError:
                        logger.info(application)
            inputSeqs.append(inputSeq)
            targetSeqs.append(targetSeq)
        return inputSeqs,targetSeqs

    def getSyntheticDataForSeq2Point(self,syntheticNum,distractorRatio = 0.5,targetRatio = 0.5):
        applicationNum = len(self.applications)
        inputSeqs = []
        targetSeqs = []
        for i in range(0,syntheticNum):
            inputSeq = np.zeros(self.window, dtype=np.float32)
            targetSeq = np.zeros(self.window, dtype=np.float32)
            if self.rng.binomial(n=1,p=targetRatio):
                targetActivation,building,index = self._getAtivation(self.targetApplication)
                activationLength = len(targetActivation)
                positioned_activation = self._positionActivation(targetActivation,self.targetApplication,building,self.window,index).values
                inputSeq = inputSeq + positioned_activation
                targetSeq = targetSeq + positioned_activation
            for application in self.otherApplications:
                if self.rng.binomial(n=1,p=distractorRatio):
                    try:
                        activation,building,index = self._getAtivation(application)
                        positioned_activation = self._positionActivation(activation,application,building,self.window,index,isReal = False).values
                        inputSeq = inputSeq + positioned_activation
                    except TypeError:
                        logger.info(application)
            inputSeqs.append(inputSeq)
            targetSeqs.append(targetSeq[int(len(targetSeq)/2)])
            if i%10000 == 0:
                logger.info(str(i)+" is ok")
        npx = np.array(inputSeqs)
        npy = np.array(targetSeqs)
        with open('./data/syntheticTrainDatax.pk', 'wb+') as f:
            pickle.dump(npx,f)
            f.close()
        with open('./data/syntheticTrainDatay.pk', 'wb+') as f:
            pickle.dump(npy,f)
            f.close()
        return npx,npy
     
    def _hasSufficientSamples(self,start,end,data,ratio = 0.99):
        if len(data)<2:
            return False
        numExpected = (end-start).total_seconds()/6
        hitRate = len(data)/numExpected
        return hitRate>=ratio
    
    def _selectActivation(self,activations):
        total = len(activations)
        index = self.rng.randint(0,total)
        return index
 
    def _positionActivation(self,activation,application,building,windowLen,activationIndex,isReal = True):
        startTime = activation.index[0]
        endTime = activation.index[-1]
        if(len(activation)<windowLen):
            addnum = windowLen - len(activation)
            an = self.rng.randint(0,addnum)
            bn = addnum -an
            positioned_activation = np.pad(activation.values, pad_width=(an, 0), mode='constant')
            positioned_activation = np.pad(positioned_activation, pad_width=(0, bn),mode='constant')
            seq_start_time = activation.index[0] - timedelta(seconds=an * 6)
            index = pd.date_range(seq_start_time, periods=windowLen, freq="{:d}S".format(6))
            
            if isReal:
                intersections = []
                activationsnum = len(self.activationsApp[application][building])
                if an > 0 and activationIndex >= 1:
                    beforeStart = TimeFrame(startTime - timedelta(seconds = an * 6),startTime)
                    ai = activationIndex - 1
                    beforeActivation = self.activationsApp[application][building][ai]
                    beforeSection = TimeFrame(beforeActivation.index[0],beforeActivation.index[-1])
                    intersection = beforeSection.intersection(beforeStart)
                    while intersection.start != None and intersection.end != None:
                        intersections.append(intersection)
                        ai = ai - 1
                        if ai < 0:
                            break
                        beforeActivation = self.activationsApp[application][building][ai]
                        beforeSection = TimeFrame(beforeActivation.index[0],beforeActivation.index[-1])
                        intersection = beforeSection.intersection(beforeStart)
                if bn > 0 and activationIndex < activationsnum-1:
                    afterEnd = TimeFrame(endTime,endTime + timedelta(seconds = bn * 6))
                    bi = activationIndex + 1
                    afterActivation = self.activationsApp[application][building][bi]
                    afterSection = TimeFrame(afterActivation.index[0],afterActivation.index[-1])
                    intersection = afterSection.intersection(afterEnd)
                    while intersection.start != None and intersection.end != None:
                        intersections.append(intersection)
                        bi = bi + 1
                        if bi >= activationsnum:
                            break
                        afterActivation = self.activationsApp[application][building][bi]
                        afterSection = TimeFrame(afterActivation.index[0],afterActivation.index[-1])
                        intersection = afterSection.intersection(afterEnd)
                
                for intersection in intersections:
                    intersectionStart = intersection.start
                    intersectionEnd = intersection.end
                    length = int((intersectionEnd - intersectionStart).total_seconds() / 6) + 1
                    offset = int((intersectionStart - seq_start_time).total_seconds() / 6)
                    positioned_activation[offset:offset+length] = positioned_activation[offset:offset+length] + self.elecApp[application][building][intersectionStart:intersectionEnd].values

            positioned_activation_series = pd.Series(positioned_activation, index=index)
        else:
            positioned_activation_series = activation[:windowLen]
        if len(positioned_activation_series) != windowLen:
            logger.error("error")
        return positioned_activation_series

    def _getRealData(self,TrainOrTest = "Train",beforeStart = 0,afterEnd = 0):
        inputSeqs = []
        targetSeqs = []
        mainLen = self.window + beforeStart + afterEnd
        if TrainOrTest == "Train":
            buildings = self.trainBuildings
        else:
            buildings = self.testBuildings
        for building in buildings:
            try:
                activations = self.activationsApp[self.targetApplication][building]
                realNum = len(activations)
                for i,activation in enumerate(activations):
                    positionedActivation = self._positionActivation(activation,self.targetApplication,building,self.window,i)
                    startTime = positionedActivation.index[0] - timedelta(seconds = 6 * beforeStart)
                    endTime = positionedActivation.index[-1] + timedelta(seconds = 6 * afterEnd) + timedelta(seconds=6*10)
                    mains = self.elecMains[building][startTime:endTime]
                    if len(mains) >= mainLen:
                        inputSeqs.append(np.array(mains.values[:mainLen]))
                        targetSeqs.append(np.array(positionedActivation.values))
                for _ in range(realNum):
                    l = len(self.sectionsWithNoTarget[building])
                    randindex = self.rng.randint(0,l)
                    gap = self.sectionsWithNoTarget[building][randindex]
                    endTime = gap.end
                    startTime = gap.start
                    lastStartTime = endTime - timedelta(seconds=mainLen*6)
                    totalSeconds = (lastStartTime - startTime).total_seconds()
                    if totalSeconds < 1:
                        continue
                    offset = self.rng.randint(0,totalSeconds)
                    offset = offset - offset % 6
                    startTime = startTime + timedelta(seconds=offset)
                    mains = self.elecMains[building][startTime:endTime]
                    if len(mains) >= mainLen:
                        inputSeqs.append(np.array(mains.values[:mainLen]))
                        targetSeqs.append(np.zeros(self.window, dtype=np.float32))
            except KeyError:
                continue

        return inputSeqs,targetSeqs

        
if __name__ ==  "__main__":
    dataset = myData(trainBuildings = [1,3,4,5],testBuildings = [2],applications = ['fridge','kettle','dish washer','microwave','washing machine'],targetapplication = 'washing machine')
    dataset.getSyntheticDataForSeq2Point(300000)
    dataset.getTrainDataForSeq2Point(1000)
    dataset.getTestDataForSeq2Point(1000)
    
    
    