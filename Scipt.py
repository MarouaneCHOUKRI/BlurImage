import cv2

# Ouvre le fichier vidéo
cap = cv2.VideoCapture('video.mp4')

# Charge le classifieur de visage
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Crée un objet vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

# Boucle sur chaque frame de la vidéo
while True:
    # Lit le frame actuel
    ret, frame = cap.read()

    # Si le frame est None, c'est la fin de la vidéo
    if frame is None:
        break

    # Convertit l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecte les visages dans l'image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Boucle sur chaque visage détecté
    for (x, y, w, h) in faces:
        # Floute le visage
        face_img = cv2.GaussianBlur(frame[y:y+h, x:x+w], (23, 23), 30)

        # Copie le visage flouté de nouveau dans l'image originale
        frame[y:y+h, x:x+w] = face_img

    # Écrit le frame modifié dans le fichier vidéo de sortie
    out.write(frame)

# Libère les objets vidéo
cap.release()
out.release()

# Ferme toutes les fenêtres OpenCV
cv2.destroyAllWindows()
