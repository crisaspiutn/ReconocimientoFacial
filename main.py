import cv2
import mediapipe as mp
import numpy as np

my_selfie=mp.solutions.selfie_segmentation
cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
# video_name="movie.mp4"
# cap2=cv2.VideoCapture(video_name)
BG_COLOR=(219,203,255)
with my_selfie.SelfieSegmentation(0) as selfie_segmentation:
  while True:
    ret, frame=cap.read()
    if ret ==False:
      break
    
    # ret2, bg_image=cap2.read()
    # if ret2 ==False:
    #   cap2=cv2.VideoCapture(video_name)
    #   ret2, bg_image= cap2.read()
    
    frame_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results=selfie_segmentation.process(frame_rgb)
    
    _, th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
    th=th.astype(np.uint8) # al parecer contiene la parte visible / en imagen / creo que sirve para usarlo con frames de video / DIJO: ES UNA IMAGEN BINARIA
    th=cv2.medianBlur(th,13) # esto suavisa la visibilidad
    
    th_inv=cv2.bitwise_not(th) # al parecer contiene o selecciona la parte visible   / en negro / creo que sirve para usarlo con fondo artifical hecho con pixeles
    # crear background imagen rosa
    bg_image=np.ones(frame.shape, dtype=np.uint8)
    bg_image[:]=BG_COLOR
    # crear background imagen rosa
    bg=cv2.bitwise_and(bg_image,bg_image,mask=th_inv);# al decir mask se refiere a la parte visible, entonces parte visible anexado al fondo rosa
    # ahora mostrar parte visible de video
    fg=cv2.bitwise_and(frame,frame,mask=th)
    # sumar frames de video con frames artificiales de pixel
    suma_de_frames=cv2.add(bg,fg)# suma bien porque bg tiene negro la parte visible y fg tiene la parte invisible negro / negro significa 0 entonces al sumarse solo queda el color
    
    # cv2.imshow("results.segmentation_mask",results.segmentation_mask)
    # cv2.imshow("Th", th)
    # cv2.imshow("fg", fg) # muestra el fondo en color negro / y su parte mostrante tiene color
    cv2.imshow("suma_de_frames", suma_de_frames)
    cv2.imshow('Th_inv',th_inv)
    # cv2.imshow('titulo bg',bg) # muestra la parte visible en color negro / y su fondo tiene color
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()