from ultralytics import YOLO
import cv2

# Cargamos el modelo YOLO
model = YOLO("spiderman.pt")

# Cargamos el video de entrada
#video_path = "./Inputs/000_video.mp4"
cap = cv2.VideoCapture(0)

# Opcional: Ajustar resolución para mayor fluidez
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    # Leemos el frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Realizamos la inferencia de YOLO sobre el frame
    results = model(frame)

    # Extraemos los resultados
    annotated_frame = results[0].plot()

    # Visualizamos los resultados
    cv2.imshow("YOLO Inference", annotated_frame)

    # El ciclo se rompe al presionar "Esc"
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
