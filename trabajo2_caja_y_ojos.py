import cv2
import mediapipe as mp
from ultralytics import YOLO
model_yolo= YOLO('yolov8n.pt')
mp_face_mesh = mp.solutions.face_mesh
model= mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
ojo_izquierdo= [33,133]
ojo_derecho= [362,263]
iris_izquierdo = [469, 470, 471, 472]
iris_derecho = [474, 475, 476, 477]


def centro_iris(landmarks, indices, shape):
    h, w = shape[:2]
    x = int(sum(landmarks[i].x for i in indices) / len(indices) * w)
    y = int(sum(landmarks[i].y for i in indices) / len(indices) * h)
    return (x, y)

cap= cv2.VideoCapture('corriendo.mp4')
while True:
    ret, frame = cap.read()#LECTURAS DE FRAMES
    if not ret:
        break
    frame= cv2.resize(frame, (640,480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#CANBIAMOS A RGB MEDIAPIPE

    #---PRIMERO: CAJA EN LA PERSONA DETECTADA ---
    resultados= model_yolo.track(frame, persist=True, classes=[0],verbose=False)[0]
    if resultados.boxes.id is not None:
        for caja in resultados.boxes.xyxy:
            x1,y1,x2,y2= map(int, caja)#limites de la caja
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    #---DESPUES: PUNTOS EN LOS OJOS---OJALA FUERAN SUS OJOS ---
    results = model.process(rgb_frame) 
    if results.multi_face_landmarks:
        h, w, c = frame.shape
        for face_landmarks in results.multi_face_landmarks:
            #OJOS IZQUIERDOS
            ubicacion_externa_ojo_izquierdo= (int(face_landmarks.landmark[ojo_izquierdo[0]].x * w),
                         int(face_landmarks.landmark[ojo_izquierdo[0]].y * h))
            cv2.circle(frame, ubicacion_externa_ojo_izquierdo, 3, (0, 255, 0), -1)

            ubicacion_externa_ojo_interno= (int(face_landmarks.landmark[ojo_izquierdo[1]].x * w),
                         int(face_landmarks.landmark[ojo_izquierdo[1]].y * h))
            cv2.circle(frame, ubicacion_externa_ojo_interno, 3, (0, 255, 0), -1)

            #OJOS DERCHOS
            ubicacion_interna_ojo_derecho= (int(face_landmarks.landmark[ojo_derecho[0]].x * w),
                         int(face_landmarks.landmark[ojo_derecho[0]].y * h))
            cv2.circle(frame, ubicacion_interna_ojo_derecho, 3, (0, 255, 0), -1)

            ubicacion_externa_ojo_derecho= (int(face_landmarks.landmark[ojo_derecho[1]].x * w),
                         int(face_landmarks.landmark[ojo_derecho[1]].y * h))
            cv2.circle(frame, ubicacion_externa_ojo_derecho, 3, (0, 255, 0), -1)

            # IRIS IZQUIERDO
            centro_iris_izquierdo = centro_iris(face_landmarks.landmark, iris_izquierdo, frame.shape)
            cv2.circle(frame, centro_iris_izquierdo, 4, (0, 0, 255), -1)

            # IRIS DERECHO
            centro_iris_derecho = centro_iris(face_landmarks.landmark, iris_derecho, frame.shape)
            cv2.circle(frame, centro_iris_derecho, 4, (0, 0, 255), -1)

    cv2.imshow('Caja y ojos', frame)
    if cv2.waitKey(50)== ord('q'):break
cap.release()
cv2.destroyAllWindows()
