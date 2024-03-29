# -*- coding: utf-8 -*-
"""Lab11_Exindividual1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AwgxUlnEiUGjCsjggByXsG1I9BmQlM1h
"""

import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dropout
from keras.optimizers import Adam, SGD # Douamodelede optimizare
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import warnings # Afisamsituatiiin care unelemetodesunt depreciate
import numpy as np # In relatiecu prelucrareaimaginilorcu Open CV
import cv2
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, EarlyStopping

#START
warnings.simplefilter(action='ignore', category=FutureWarning)
background = None
accumulated_weight= 0.5 # diferentaintreobiectsifundal
#Cream dimensiunilepentrufereastra[ROI] in care sapreluamdatele-conturmana om
ROI_top= 100
ROI_bottom= 300
ROI_right= 150
ROI_left= 350


def cal_accum_avg(frame, accumulated_weight):
  global background # variabilapentrufundal
  if background is None:
    background = frame.copy().astype("float")
    return None
  # metodain cadrulOpen CV
  cv2.accumulateWeighted(frame, background, accumulated_weight)
  

def segment_hand(frame, threshold=50):
  global background
  diff = cv2.absdiff(background.astype("uint8"), frame)
  _ , thresholded= cv2.threshold(diff, threshold,255,cv2.THRESH_BINARY)# Facemcapturaconturuluiextern
  image, contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  if len(contours) == 0:
    return None
  else:
    hand_segment_max_cont= max(contours, key=cv2.contourArea)
    return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)
num_frames= 0
element = 10 # aicivompunepe rand de la 0 la 10 numarulframe-uluicand captamimaginicu semne
num_imgs_taken= 0
while True:
  ret, frame = cam.read()
  frame = cv2.flip(frame, 1) # cu flip preveniminversiuneimagine
  frame_copy= frame.copy()
  roi= frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
  gray_frame= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  gray_frame= cv2.GaussianBlur(gray_frame, (9, 9), 0)
  if num_frames< 60:
    cal_accum_avg(gray_frame, accumulated_weight)
    if num_frames<= 59:
      cv2.putText(frame_copy, "SETARE FUNDAL...ASTEAPTA",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    
  #Lasam 300 de frame-uritimpca sapotrivimmana in frame
  elif num_frames<= 300:
     hand = segment_hand(gray_frame)
     cv2.putText(frame_copy, "Ajusteazamana pt. " +str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)


# Verificamdacamana estedetectatacontorizandnumrulde contururidetectate
  if hand is not None:
    thresholded, hand_segment= hand
    # Desenamconturulin jurulmainiidetectate
    cv2.drawContours(frame_copy, [hand_segment+ (ROI_right,ROI_top)], -1, (255, 0, 0),1)
    cv2.putText(frame_copy, str(num_frames)+" pentru" + str(element),(70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # Afisamsiimagineageneratacu threshold
    cv2.imshow("Palierimagine mana", thresholded)
  else:
    # Segmentamregiuneamainii
    hand = segment_hand(gray_frame)
    # Verificamdacaputemdetectamana
    if hand is not None:
      # despachetam
      thresholded, hand_segment = hand
      # desenamconturin jurulmainii
      cv2.drawContours(frame_copy, [hand_segment+ (ROI_right,ROI_top)], -1, (255, 0, 0),1)
      cv2.putText(frame_copy, str(num_frames), (70, 45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
      cv2.putText(frame_copy, str(num_imgs_taken) + ' imagini' +"pentru"+ str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)

      # Afisamimagineaalb-negru
      hh=0 # poatefi 0 sau300 (dacavremsasalvam601 imaginiin "train")
      cv2.imshow("Palierimagine mana", thresholded)
      # aicitrebuie sainlocuimtrain sitest in functiede ceimaginicapturam
      # la fel, 40 se inlocuiestecu 300 in cazul"train"
      # ideal arfi safacemca fiecareimagine safie unica pentrua cresteacurateteape setulde testare
      if num_imgs_taken<= 40:
        cv2.imwrite(r"S:\III\Sem II\AI\AI_Labs_Python\Lab11\Ex\Maini"+str(element)+"/" +str(num_imgs_taken+hh) + '.jpg', thresholded)
      else:
        break
        num_imgs_taken+=1
    else:
      cv2.putText(frame_copy, 'Nu estedetectatamana!', (200, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
  cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right,ROI_bottom), (255,128,0), 3)
  cv2.putText(frame_copy, "Recunoasteresemn: ", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)

  # incrementamframe-urile
  num_frames+= 1
  # afisamframe-ul cu mana segmentata
  cv2.imshow("Detectaresemn", frame_copy)
  # Inchidemtoateferestrelecu Esc sauoricealtatasta
  k = cv2.waitKey(1) & 0xFF
  if k == 27:
    break# Eliberamcamera sidistrugemtoateferestrele
cv2.destroyAllWindows()
cam.release()





