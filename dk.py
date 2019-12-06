import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### The data file needs to be in the same folder as this file####

#####  SET UP THE SWING DATA STRUCTURE   ######
#### In order to select a column from the data structure, just do ds[colname] where colname is 1 of timestamp,ax,ay,ax, wx,wy,wz ####
data = pd.read_csv("./latestSwing.csv",header=None)

ds = {}
ds['timestamp'] = data[0].tolist()
ds['ax'] = data[1].tolist()
ds['ay'] = data[2].tolist()
ds['az'] = data[3].tolist()

ds['wx'] = data[4].tolist()
ds['wy'] = data[5].tolist()
ds['wz'] = data[6].tolist()

#We can use this function to plot each axis individually. This would tell us some about the motion in each direction
def plotData(ds):
    _, ax = plt.subplots(4,2,figsize=(16,8))
    count = 0
    for name in ds:
        data = ds[name]
        inds = [x for x in range(len(data))]
        col = int(count/4)
        row = int(count%4)
        #print(row,col)
        ax[row][col].scatter(inds,data,2)
        ax[row,col].set_title(name,pad=-50)
        count += 1
    plt.show()

#We can use this function to plot a 3d path, name is either 'a' or 'w'
def plot3Ddata(ds,name):
    if not (name == 'a' or name == 'w'): raise NameError("BadNameError")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if(name == "a"):
        x = ds['ax']
        y = ds['ay']
        z = ds['az']
        title = "accelerometer"
    else:
        x = ds['wx']
        y = ds['wy']
        z = ds['wz']
        title = "gyroscope"
    ax.plot(x, y, z, label=title)
    plt.show()

#plotData(ds)
#plot3Ddata(ds,"w")

def loopWithLambda(thresholdCondition,start,end,winLength,forward=True):
    found = False
    inc = 1 if forward else -1

    while(not found):
        end = start + winLength if forward else start -winLength 
        for y in range(start, end,inc):
            if(forward and y > end): return None
            if(not forward and y < end): return None
            if(not thresholdCondition(y)):
                start = y+1
                found = False
                break
            else:
                found = True
    return start


## Assumption is data is a column, meaning it can be identified by the name timestamp,ax,ay,ax, wx,wy,wz
## We will use a sliding window so we don't have to keep rechecking indices that have already been checked
def searchContinuityAboveValue(data, indexBegin, indexEnd, threshold, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    start = indexBegin
    found = False
    while(not found):
        for y in range(start, start+winLength):
            if(y > indexEnd): return None
            if not (data[y] >= threshold):
                start = y+1
                found = False
                break
            else:
                found = True
    return start

def searchContinuityAboveValueLambda(data, indexBegin, indexEnd, threshold, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    return loopWithLambda(lambda y: data[y] >= threshold, indexBegin,indexEnd,winLength)

## Assumption is data is a column, meaning it can be identified by the name timestamp,ax,ay,ax, wx,wy,wz
## We will use a sliding window so we don't have to keep rechecking indices that have already been checked
def backSearchContinuityWithinRange(data, indexBegin, indexEnd, thresholdLo, thresholdHi, winLength):
    if(indexBegin < indexEnd): raise ValueError("indexBegin is smaller than the indexEnd")
    if(-indexEnd + indexBegin < winLength): raise ValueError("winLength is too long for the range")
    if(thresholdHi < thresholdLo): raise ValueError("thresholdLo is larger than thresholdHi")
    start = indexBegin
    found = False
    while(not found):
        for y in range(start, start-winLength,-1):
            if(y < indexEnd): return None
            if not (data[y] <= thresholdHi and data[y] >= thresholdLo):
                start = y-1
                found = False
                break
            else:
                found = True
    return start

def backSearchContinuityWithinRangeLambda(data, indexBegin, indexEnd, thresholdLo, thresholdHi, winLength):
    if(indexBegin < indexEnd): raise ValueError("indexBegin is smaller than the indexEnd")
    if(-indexEnd + indexBegin < winLength): raise ValueError("winLength is too long for the range")
    if(thresholdHi < thresholdLo): raise ValueError("thresholdLo is larger than thresholdHi")
    return loopWithLambda(lambda y: data[y] >= thresholdLo and data[y] <= thresholdHi, indexBegin, indexEnd,winLength, False)
    
## Assumption is data is a column, meaning it can be identified by the name timestamp,ax,ay,ax, wx,wy,wz
## We will use a sliding window so we don't have to keep rechecking indices that have already been checked
def searchContinuityAboveValueTwoSignals(data1, data2, indexBegin, indexEnd, threshold1, threshold2, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    start = indexBegin
    found = False
    while(not found):
        for y in range(start, start+winLength):
            if(y > indexEnd): return None
            if not(data1[y] > threshold1 and data2[y] > threshold2):
                start = y+1
                found = False
                break
            else:
                found = True
    return start

def searchContinuityAboveValueTwoSignalsLambda(data1, data2, indexBegin, indexEnd, threshold1, threshold2, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    return loopWithLambda(lambda y: data1[y] > threshold1 and data1[y] > threshold2, indexBegin,indexEnd,winLength)



## Assumption is data is a column, meaning it can be identified by the name timestamp,ax,ay,ax, wx,wy,wz
## We will use a sliding window so we don't have to keep rechecking indices that have already been checked
def searchMultiContinuityWithinRange(data, indexBegin, indexEnd, thresholdLo, thresholdHi, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    if(thresholdHi < thresholdLo): raise ValueError("thresholdLo is larger than thresholdHi")
    start = indexBegin
    found = False
    vals = []
    while(not found):
        for y in range(start, start+winLength):
            if(y > indexEnd or y > len(data)): return vals
            if  not (data[y] <= thresholdHi and data[y] >= thresholdLo):
                start = y+1
                found = False
                break
            else:
                found = True
        if(found):
            vals.append((start,start+winLength))
            start = start + 1
            found = False
    
    return vals

#given a list of indices, merge them into tuples
def merge_vals(vals,winLength):
    indices = []
    start = 0
    end = 0
    for i in range(len(vals)-1):
        if(vals[i+1]-vals[i] > 1):
            end = i+1
            if(end - start >= winLength):
                indices.append((start,end))
            start = end
    return indices


def searchMultiContinuityWithinRangeLambda(data, indexBegin, indexEnd, thresholdLo, thresholdHi, winLength):
    if(indexBegin > indexEnd): raise ValueError("indexBegin is greater than the indexEnd")
    if(-indexBegin + indexEnd < winLength): raise ValueError("winLength is too long for the range")
    if(thresholdHi < thresholdLo): raise ValueError("thresholdLo is larger than thresholdHi")
    vals = []
    for i in range(indexBegin,indexEnd):
        thresholded = loopWithLambda(lambda y: data[y] >= thresholdLo and data[y] <= thresholdHi, i,i+1,1)
        vals.append(thresholded)
        
    return merge_vals(vals,winLength)


#### WE use this function to see at what point we ha    ve the swing, we can see from the plots that there is a huge jump in each
# component when this takes place, so we just want to find the frame at which this jump happens
# USING THIS WE SEE THE SWING HAPPENS AROUND FRAME 875, this is consistent with the plots
#   ####

def getEventFrame(dscol):
    maxSoFar = 0
    maxInd = -1
    for i in range(1,len(dscol)):
        diff = abs(dscol[i]-dscol[i-1])
        if(diff > maxSoFar):
            maxSoFar = diff
            maxInd = i
    return maxInd


for name in ds:
    if(name == "timestamp"): continue
    print(getEventFrame(ds[name]))

plotData(ds)
print(backSearchContinuityWithinRangeLambda([1,2,3,4,5,5,5,5,5,3,3,2],10,0,2,5,3))

