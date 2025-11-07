import cv2
import os
import pyttsx3

MODEL = "model_lbph.yml"
LABELS = "labels.txt"


def load_labels():
    labels = {}
    if not os.path.exists(LABELS):
        print("⚠️ labels.txt not found. Run train_lbph.py first.")
        return labels
    with open(LABELS, "r", encoding="utf-8") as f:
        for line in f:
            id_str, name = line.strip().split(",", 1)
            labels[int(id_str)] = name
    return labels


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def main():
    if not os.path.exists(MODEL):
        print("⚠️ model_lbph.yml not found. Run train_lbph.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL)
    labels = load_labels()

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    spoken_names = set()  # To avoid repeating voice endlessly

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)

            if conf < 70:  # smaller = better
                name = labels.get(id_, "Unknown")
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Speak only once per session
                if name not in spoken_names:
                    speak(f"Hello {name}")
                    spoken_names.add(name)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
