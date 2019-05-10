import numpy as np

data = np.array(np.load('training_data_1.npy'))

def getNearestPt(target):

    ## scale to screen
    target[0] = target[0]+430
    target[1] = target[1]+105
    
    m = 10**5
    diff = 0
    index = -1

    for d in range(len(data)):
        diff = abs( data[d][0][0] - target[0] )
        if m > diff:
            m = diff
            index = d

    x = data[index][0][0]

    m = 10**5
    diff = 0
    index = -1
    for d in range(len(data)):
        diff = abs( data[d][0][1] - target[1] )
        if m > diff:
            m = diff
            index = d


    y = data[index][0][1]

    m = 10**5
    diff = 0
    index = -1
    for d in range(len(data)):
        diff = abs( data[d][0][2] - target[2] )
        if m > diff:
            m = diff
            index = d

    r = data[index][0][2]

    return np.array([x,y,r])


