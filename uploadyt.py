from os import listdir
from os.path import isfile, join
import os
import time
import upload_video
import http
import httplib2
from apiclient.errors import HttpError
import ocr
import sys
import io
fileList = []
print(listdir("C:\\Users\\Captain Falcon\\Desktop\\Extra\\0E7DF678130F4F0FA2C88AE72B47AFDF\\2020"))



def getFiles(fList):
    dirList = listdir(fList)
    for item in dirList:
        if item.endswith(".mp4"):
            fileList.append(fList+"\\"+item)
        else:
            getFiles(fList+"\\"+item)

getFiles("C:\\Users\\Captain Falcon\\Desktop\\Extra\\0E7DF678130F4F0FA2C88AE72B47AFDF\\2020")
for item in fileList:
    epochtime = os.path.getmtime(item)
    epochtime = (time.ctime(epochtime))
    ogStd = sys.stdout
    sys.stdout = io.StringIO()
    chars = ocr.determineChars(item)
    winner = ocr.determineWinner(item)
    sys.stdout=ogStd

    fileStr = "--file=\""+item+"\" "
    titleStr = "--title=\""+chars[0]+" vs "+ chars[1]+"\" "
    descStr = "--description=\""+chars[0]+" vs "+ chars[1] + "\n"+"Winner:"+chars[winner]+"\n"+epochtime+"\" "
    catStr = "--category=\"20\" "
    keyStr = "--keyword=\"SSBU Smash SmashUltimate Ultimate SuperSmashBro SuperSmashBroUltimate "
    keyStr+= chars[0] + " " + chars[1]+"\" "
    privStr = "--privacyStatus=\"public\" "
    upStr = "upload_video.py "+fileStr+titleStr+descStr+catStr+privStr
    print(upStr)
    os.system(upStr)
