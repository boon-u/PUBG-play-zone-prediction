import numpy as np
from PIL import ImageGrab
import cv2
import time
from alexnet import alexnet
from getNearest import getNearestPt
import random
from math import sqrt

lt = [1,0,0,0]
lb = [0,1,0,0]
rt = [0,0,1,0]
rb = [0,0,0,1]



diameter = [330, 208, 106, 52, 25, 15] # from my analysis (units in pixels)
radius = []

for d in diameter:
    radius.append(d//2)
    

WIDTH = 3
HEIGHT = 1
LR = 1e-3
EPOCHS = 5

MODEL_NAME = 'PUBG_PLAYZONE_PREDICTION.model'

model = alexnet(WIDTH,HEIGHT,LR)
model.load(MODEL_NAME)

def top_left(target):

    possibilities = [[570,286], [585,267], [624,253]]
    return possibilities[random.randint(0,2)]

    

def top_right(target):
    possibilities = [[660, 263], [685, 290], [690,325]]
    return possibilities[random.randint(0,2)]


def bottom_left(target):
    possibilities = [[565,330], [600,360], [640,365]]
    return possibilities[random.randint(0,2)]
    

def  bottom_right(target):
    possibilities = [[690,315], [677, 331], [668,344], [653,357], [632,368]]
    return possibilities[random.randint(0,4)]


def process_img(image):
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(image, threshold1 = 200, threshold2 = 300)
    processed_img = cv2.GaussianBlur(processed_img,(7,7), 0) ## 3,3 is original
    return processed_img


while True:
    screen = np.array(ImageGrab.grab(bbox = (430,105,985,610)))
    img = process_img(screen)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,200,
                                param1=1,param2=105,minRadius=10,maxRadius=200)


    
    target = getNearestPt([circles[0][0][0], circles[0][0][1], circles[0][0][2]])

##    print([target.reshape(-1, WIDTH, HEIGHT, 1)][0])    
    prediction = model.predict([target.reshape(-1, WIDTH, HEIGHT, 1)][0])[0]

    choice = np.argmax(prediction)

    r = radius.index(target[2])

    r += 1
        
            
    draw = [0,0]
    if choice == 0:
        draw = top_left(target)        
    elif choice == 1:
        draw = bottom_left(target)    
    elif choice == 2:
        draw = top_right(target)    
    elif choice == 3:
        draw = bottom_right(target)    

    ##    circles = np.uint16(np.around(circles))

   
    print(draw[0] - 430, draw[1] - 105)
    cv2.circle(img, (circles[0][0][0], circles[0][0][1]), circles[0][0][2], (255,0,0), 2)
    cv2.circle(img,(draw[0]-430, draw[1]-105), radius[r], (255, 0, 0), 2)
    cv2.imshow('Prediction', img)
    
        
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
