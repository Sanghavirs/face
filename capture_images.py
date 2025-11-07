import cv2
import os

# Haarcascade for face detection
FACE_CASCADE: object = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    person_id = input("Enter numeric ID (e.g., 1): ").strip()
    person_name = input("Enter name (e.g., Alice): ").strip()

    save_dir = f"dataset/{person_id}_{person_name}"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print("📸 Press 'c' to capture image, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c") and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            file_path = f"{save_dir}/{person_name}_{count}.jpg"
            cv2.imwrite(file_path, face_img)
            print(f"✅ Saved {file_path}")
            count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
