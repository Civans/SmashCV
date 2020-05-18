import cv2
import pytesseract
import re
from difflib import SequenceMatcher
import numpy as np

charFile = open("char list","r")
charList = []
ocrCheckList =[]
true = True
false = False
for line in charFile:
    charList.append(re.sub(r'[^a-zA-z-. ]','',line).strip())
    ocrCheckList.append(re.sub(r'[^a-zA-z]','',line).lower())
def comparison(ocrList):
    similarity = []
    for name in ocrCheckList:
        print(name)
        similarities = []
        avgSim = 0
        for ocrName in ocrList:
            ratio = SequenceMatcher(None,name,ocrName).ratio()
            similarities.append(ratio)
            print(name+":"+ocrName+"::"+str(ratio))
        for sim in similarities:
            avgSim+=sim
        avgSim/=len(similarities)
        print(name+"::"+str(avgSim))
        similarity.append(avgSim)
        index = 0
    for i in range(0,len(similarity)):
        print(str(similarity[index])+"<"+ str( similarity[i]))
        if similarity[index] < similarity[i]:
            print(str(similarity[i]) + "is more similar than "+str(similarity[index]))
            index = i
    print("Highest:"+str(index))
    return charList[index]
    
def determineChars(video):
    config = ('-l eng --oem 1 --psm 11') 
    p1Rect = [(338,670),(472,686)]
    p2Rect = [(834,670),(968,686)]
    cap = cv2.VideoCapture(video)
    if(cap.isOpened() == False):
        print("No shit")

    p1TextSamples = []
    p2TextSamples = []
    for i in range(0,9):
        if cap.isOpened():
            ret,frame = cap.read()
            if ret == True:
                p1ROI = frame[p1Rect[0][1]:p1Rect[1][1],p1Rect[0][0]:p1Rect[1][0]]
                p2ROI = frame[p2Rect[0][1]:p2Rect[1][1],p2Rect[0][0]:p2Rect[1][0]]
                gray = cv2.cvtColor(p1ROI, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                p1ROI = cv2.resize(thresh,None,fx=2,fy=2)
                gray = cv2.cvtColor(p2ROI, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                p2ROI = cv2.resize(thresh,None,fx=2,fy=2)
                p1Char = pytesseract.image_to_string(p1ROI,config=config)
                p2Char = pytesseract.image_to_string(p2ROI,config=config)
                p1Char = re.sub(r'[^a-zA-Z]','',p1Char)
                p1Char = p1Char.lower()
                p2Char = re.sub(r'[^a-zA-Z]','',p2Char)
                p2Char = p2Char.lower()
                p1TextSamples.append(p1Char)
                p2TextSamples.append(p2Char)
    cap.release()
    cv2.destroyAllWindows()

    p1Name = (comparison(p1TextSamples))
    p2Name = (comparison(p2TextSamples))
    print(p1Name)
    print(p2Name)
    return [p1Name,p2Name]

def determineCharsDebug(video):
    config = ('-l eng --oem 1 --psm 11') 
    p1Rect = [(338,670),(472,686)]
    p2Rect = [(834,670),(968,686)]
    cap = cv2.VideoCapture(video)
    if(cap.isOpened() == False):
        print("No shit")

    p1TextSamples = []
    p2TextSamples = []
    fitler = (0,0,0,255,255,255)
    
    for i in range(0,9):
        if cap.isOpened():
            ret,frame = cap.read()
            if ret == True:
                p1ROI = frame[p1Rect[0][1]:p1Rect[1][1],p1Rect[0][0]:p1Rect[1][0]]
                p2ROI = frame[p2Rect[0][1]:p2Rect[1][1],p2Rect[0][0]:p2Rect[1][0]]

      
                gray = cv2.cvtColor(p1ROI, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # Morph open to remove noise and invert image
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                invert = 255-opening
                invert=cv2.resize(thresh,None,fx=2,fy=2)
                key = cv2.waitKey(5)
                cv2.imshow("",invert)
                p1Char = pytesseract.image_to_string(invert,config=config)
                print(p1Char)
                        

                p1Char = pytesseract.image_to_string(p1ROI,config=config)
                p2Char = pytesseract.image_to_string(p2ROI,config=config)
                p1Char = re.sub(r'[^a-zA-Z]','',p1Char)
                p1Char = p1Char.lower()
                p2Char = re.sub(r'[^a-zA-Z]','',p2Char)
                p2Char = p2Char.lower()
                p1TextSamples.append(p1Char)
                p2TextSamples.append(p2Char)
    cap.release()
    cv2.destroyAllWindows()

    p1Name = (comparison(p1TextSamples))
    p2Name = (comparison(p2TextSamples))
    print(p1Name)
    print(p2Name)
    return [p1Name,p2Name]

def determineWinner(video):
    winnerRect = [(70,95),(146,152)]
    cap = cv2.VideoCapture(video)
    if(cap.isOpened() == False):
        print("No shit")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    print(frames)
    desiredSeek = frames - int(fps * 9)
    print(desiredSeek)
    seconds = desiredSeek/fps
    print(seconds)
    minutes = seconds/60
    print(minutes)
    partial = minutes - int(minutes)
    print(partial)
    seconds = partial * 60
    print(seconds)
    print(str(int(minutes)) +":"+ str(seconds))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,(desiredSeek))

    ret,img = cap.read()
    winTxt = []
    p1Count = 0
    p2Count = 0
    
    while ret:
            ret,img = cap.read()
            if ret:
                low_blue, low_green, low_red, upper_blue, upper_green, upper_red = (115, 0, 0, 255, 178, 255)
                lower_color = np.array((low_blue, low_green, low_red))
                upper_color = np.array((upper_blue, upper_green, upper_red))

 
                winROI = img[winnerRect[0][1]:winnerRect[1][1],winnerRect[0][0]:winnerRect[1][0]]
                # extract binary image with active blue regions
                binary_image = cv2.inRange(winROI, lower_color, upper_color)

                #erode for the little white contour to dissapear
                binary_image = cv2.erode(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                binary_image = cv2.dilate(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                winROI = binary_image
                winROI=cv2.resize(winROI,None,fx=2,fy=2)
                wConfig='-l eng --oem 1 --psm 10 -c tessedit_char_whitelist=P12'
                Txt = pytesseract.image_to_string(winROI,config=wConfig)
                print(Txt)
                winTxt.append(Txt)
                p1Count += SequenceMatcher(None,Txt,"P1").ratio()
                p2Count += SequenceMatcher(None,Txt,"P2").ratio()
                desiredSeek+=1
                seconds = desiredSeek/fps
                minutes = seconds/60
                partial = minutes - int(minutes)
                seconds = partial * 60
                print(str(int(minutes)) +":"+ str(seconds))
            else:
                break
    cap.release()
    cv2.destroyAllWindows()

    p1Count/=len(winTxt)
    p2Count/=len(winTxt)
    if p1Count > p2Count:
        return 0
    else :
        return 1
    cap.release()
    cv2.destroyAllWindows()
    
def determineWinnerDebug(video):
    winnerRect = [(70,95),(150,152)]
    cap = cv2.VideoCapture(video)
    if(cap.isOpened() == False):
        print("No shit")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(fps)
    print(frames)
    desiredSeek = frames - int(fps * 9)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,(desiredSeek))

    ret,img = cap.read()
    winTxt = []
    p1Count = 0
    p2Count = 0
    
    cv2.namedWindow("",cv2.WINDOW_NORMAL)
    p2Filter = (202, 0, 0, 354, 235, 174)
    p1Filter = (95, 121, 200, 255, 255, 261)

    ret,img = cap.read()
    auto = False
    skipFilter = False
    while ret:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if not auto:
            auto = key == ord('w')
        if key == ord('p'):
            skipFilter = not skipFilter
        if key == ord('e') or auto:
            ret,img = cap.read()
            if ret:
                redSum = 0
                blueSum = 0



 
                winROI = img[winnerRect[0][1]:winnerRect[1][1],winnerRect[0][0]:winnerRect[1][0]]

                row,col,channels = winROI.shape
                for i in range(row):
                    for j in range(col):
                        redSum+=winROI[i,j][2]
                        blueSum+=winROI[i,j][0]
                redSum/=(i*j)
                blueSum/=(i*j)
                print(redSum)
                print(blueSum)
                lower_color = None
                upper_Color = None
                if redSum>blueSum:
                    lower_color = np.array([p1Filter[0],p1Filter[1],p1Filter[2]])
                    upper_color = np.array([p1Filter[3],p1Filter[4],p1Filter[5]])
                else:
                    lower_color = np.array([p2Filter[0],p2Filter[1],p2Filter[2]])
                    upper_color = np.array([p2Filter[3],p2Filter[4],p2Filter[5]])
                if not skipFilter:
                    while 1:
                        # extract binary image with active blue regions
                        binary_image = cv2.inRange(winROI, lower_color, upper_color)

                        cv2.imshow('Original image', binary_image)

                        #erode for the little white contour to dissapear
                        binary_image = cv2.erode(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                        binary_image = cv2.dilate(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))

                        cv2.imshow('Binary image  ', binary_image)

                        k = cv2.waitKey(5) & 0xFF
                        if k == ord('p'):
                            skipFilter = not skipFilter
                        if k == 27:
                            break
                        if k == ord('q'):
                            low_blue += 1
                        if k == ord('w'):
                            low_blue -= 1
                        if k == ord('a'):
                            low_green += 1
                        if k == ord('s'):
                            low_green -= 1
                        if k == ord('z'):
                            low_red += 1
                        if k == ord('x'):
                            low_red -= 1
                        if k == ord('e'):
                            upper_blue += 1
                        if k == ord('r'):
                            upper_blue -= 1
                        if k == ord('d'):
                            upper_green += 1
                        if k == ord('f'):
                            upper_green -= 1
                        if k == ord('c'):
                            upper_red += 1
                        if k == ord('v'):
                            upper_red -= 1
                        print("low_blue, low_green, low_red, upper_blue, upper_green, upper_red =",(low_blue, low_green,low_red,upper_blue,upper_green,upper_red))

                # extract binary image with active blue regions
                binary_image = cv2.inRange(winROI, lower_color, upper_color)

                cv2.imshow('Original image', binary_image)

                #erode for the little white contour to dissapear
                binary_image = cv2.erode(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                binary_image = cv2.dilate(binary_image, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
                winROI = binary_image
                winROI=cv2.resize(winROI,None,fx=2,fy=2)
                wConfig='-l eng --oem 1 --psm 10 -c tessedit_char_whitelist=P12'
                cv2.rectangle(img,winnerRect[0],winnerRect[1],(0,0,0),2)
                wConfig='-l eng --oem 1 --psm 10 -c tessedit_char_whitelist=P12'
                Txt = pytesseract.image_to_string(winROI,config=wConfig)
                print(Txt)
                winTxt.append(Txt)
                p1Count += SequenceMatcher(None,Txt,"P1").ratio()
                p2Count += SequenceMatcher(None,Txt,"P2").ratio()
                cv2.rectangle(img,winnerRect[0],winnerRect[1],(255,255,255),2)
                cv2.imshow("winroi",winROI)
                cv2.imshow("",img)
                cv2.resizeWindow("",800,600)
                print(winTxt)
                desiredSeek+=1
                seconds = desiredSeek/fps
                minutes = seconds/60
                partial = minutes - int(minutes)
                seconds = partial * 60
                print(str(int(minutes)) +":"+ str(seconds))
            else:
                break
    p1Count/=len(winTxt)
    p2Count/=len(winTxt)
    
    if p1Count > p2Count:
        return 0
    else :
        return 1
    cap.release()
    cv2.destroyAllWindows()
winp1 = "C:/Users/Captain Falcon/Desktop/Extra/0E7DF678130F4F0FA2C88AE72B47AFDF/2020/05/16/2020051601004000-C6D726972790F87F6521C61FBA400A1DX.mp4"
winp2 = "C:/Users/Captain Falcon/Desktop/Extra/0E7DF678130F4F0FA2C88AE72B47AFDF/2020/05/16/2020051601084700-C6D726972790F87F6521C61FBA400A1DX.mp4"

char1 = "C:/Users/Captain Falcon/Desktop/Extra/0E7DF678130F4F0FA2C88AE72B47AFDF/2020/04/28/2020042810530300-C6D726972790F87F6521C61FBA400A1DX.mp4"
#print(determineCharsDebug(char1))
