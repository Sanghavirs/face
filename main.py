# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
import cv2
import numpy as np
import os

# TTS: try plyer (Android). fallback to pyttsx3 on desktop for testing.
try:
    from plyer import tts
    def speak_text(text):
        try:
            tts.speak(text)
        except Exception:
            # sometimes plyer can fail on desktop; ignore quietly
            pass
except Exception:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        def speak_text(text):
            engine.say(text)
            engine.runAndWait()
    except Exception:
        def speak_text(text):
            print("TTS:", text)


class FaceApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Kivy Camera widget (uses Android camera provider when on device)
        # resolution: you can tune resolution argument for performance
        self.kivy_camera = Camera(play=True, resolution=(640, 480), index=0)
        self.img_widget = Image()
        self.face_label = Label(text="Faces detected: 0", size_hint=(1, 0.12))

        self.layout.add_widget(self.img_widget)
        self.layout.add_widget(self.face_label)

        # face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # load dataset images (grayscale)
        self.known_faces = {}
        dataset_path = "dataset"
        if os.path.exists(dataset_path):
            for person_name in os.listdir(dataset_path):
                person_folder = os.path.join(dataset_path, person_name)
                if os.path.isdir(person_folder):
                    images = []
                    for file in os.listdir(person_folder):
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            img = cv2.imread(os.path.join(person_folder, file), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                images.append(img)
                    if images:
                        self.known_faces[person_name] = images

        self.greeted = set()
        Clock.schedule_interval(self.update, 1.0 / 10.0)  # 10 FPS
        return self.layout

    def texture_to_frame(self, texture):
        """Convert Kivy Texture to OpenCV BGR frame"""
        size = texture.size
        # get texture pixels as bytes
        pixels = texture.pixels  # RGBA on many providers
        if not pixels:
            return None
        arr = np.frombuffer(pixels, dtype=np.uint8)
        # if texture is RGBA (4 channels)
        if len(arr) == size[0] * size[1] * 4:
            arr = arr.reshape((size[1], size[0], 4))
            # drop alpha, convert RGBA -> BGR
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            # try RGB
            arr = arr.reshape((size[1], size[0], 3))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    def match_face(self, face_crop):
        """Compare with known faces using template matching (keeps your original approach)."""
        if face_crop is None or face_crop.size == 0:
            return "Unknown"
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        for name, imgs in self.known_faces.items():
            for known_img in imgs:
                try:
                    known_resized = cv2.resize(known_img, (gray_face.shape[1], gray_face.shape[0]))
                    res = cv2.matchTemplate(gray_face, known_resized, cv2.TM_CCOEFF_NORMED)
                    if np.max(res) > 0.6:
                        return name
                except Exception:
                    pass
        return "Unknown"

    def update(self, dt):
        # get latest texture from Kivy camera widget
        try:
            camera_texture = self.kivy_camera.texture
        except Exception:
            camera_texture = None

        if camera_texture:
            frame = self.texture_to_frame(camera_texture)
            if frame is None:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            names = []

            for (x, y, w, h) in faces:
                face_crop_color = frame[y:y+h, x:x+w]
                name_found = self.match_face(face_crop_color)

                if name_found != "Unknown" and name_found not in self.greeted:
                    speak_text(f"Hello {name_found}")
                    self.greeted.add(name_found)

                names.append(name_found)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name_found, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            self.face_label.text = "Faces: " + ", ".join(names) if names else "No faces detected"

            # convert BGR to texture for display
            display_frame = cv2.flip(frame, 0)
            buf = display_frame.tobytes()
            texture = Texture.create(size=(display_frame.shape[1], display_frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = texture

            if len(faces) == 0:
                self.greeted.clear()

    def on_stop(self):
        # nothing special since we didn't open cv2.VideoCapture
        pass


if __name__ == "__main__":
    FaceApp().run()
