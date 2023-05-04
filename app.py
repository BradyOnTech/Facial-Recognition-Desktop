from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy import utils
import cv2
import tensorflow as tf
import os
import numpy as np


class FaceID(App):

    def getWebcam(self):
        for i in range(0, 5):
            cap = cv2.VideoCapture(i)
            if cap is None or not cap.isOpened():
                print('Warning: No video source found at: ', source)
                print('Please connect a webcam')
            else:
                return cap

    def build(self):
        self.capture = self.getWebcam()
        self.web_cam = Image()
        self.button = Button(text="Verify", on_press=self.verifyFace, size_hint=(
            1, .1), background_normal='', background_color=utils.get_color_from_hex('#24a0ed'))
        self.setupButton = Button(text="Setup User", on_press=(self.setup), size_hint=(
            1, .1), background_normal='', background_color=utils.get_color_from_hex('#24a0ed'))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(
            1, .1), color=(207/255, 19/255, 19/255, 1), bold=True)
        self.validation_imgs = Label(
            text="Stored Validation images: 0", size_hint=(1, .1))
        self.vImgCount = len(os.listdir(
            os.path.join('app_data', 'validation')))
        self.validation_imgs.text = "Stored Validation images: {i}".format(
            i=str(self.vImgCount))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        layout.add_widget(self.setupButton)
        layout.add_widget(self.validation_imgs)

        # load trained siamese neural net model
        self.model = tf.keras.models.load_model('../best_SNN_model/')

        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        # opencv
        ret, frame = self.capture.read()

        # Get face recognition from haarcascade model
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (12, 235, 12), 2)

        buffer = cv2.flip(frame, 0)
        buffer = cv2.flip(buffer, 1)
        buffer = buffer.tobytes()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def getFace(self, file_path):
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        face_img = np.expand_dims(cv2.imread(file_path, 0), -1)
        img = face_cascade.detectMultiScale(face_img, 1.3, 5)
        # returns array of faces - Example: [[ 22,  55, 166, 166]] or ()
        return img

    def resizeImg(self, faces, file_path):
        x, y, w, h = faces[0]
        return cv2.resize(np.expand_dims(cv2.imread(file_path, 0), -1)[y:y+h, x:x+w, :], (100, 100), interpolation=cv2.INTER_AREA)

    def takeImg(self, path):
        ret, frame = self.capture.read()
        cv2.imwrite(path, cv2.flip(frame, 1))

    def verifyFace(self, *args):
        ANCHOR_PATH = os.path.join('app_data', 'anchor', 'anchor_image.jpg')
        self.takeImg(ANCHOR_PATH)
        anchor_face = self.getFace(ANCHOR_PATH)
        if len(anchor_face) != 1:
            popup = Popup(title='Face Detection Error', content=Label(text='One face necessary for verification'),
                          size_hint=(.5, .5), size=(250, 250))
            popup.open()
        else:
            self.validate(self.resizeImg(anchor_face, ANCHOR_PATH))

    def validate(self, anchor_img):
        detection_threshold = 0.5
        verification_threshold = 0.5
        anchors = []
        valids = []
        results = []

        if self.vImgCount >= 10:
            for i in os.listdir(os.path.join('app_data', 'validation')):
                file_path = os.path.join('app_data', 'validation', i)
                v_face = self.getFace(file_path)
                validation_img = self.resizeImg(v_face, file_path)

                # append images to array to feed to model prediction
                anchors.append(anchor_img)
                valids.append(validation_img)

            # make prediction on group and write to result array
            results = self.model.predict([np.array(anchors), np.array(valids)])

            detection = np.sum(np.array(results) > detection_threshold)

            # Verification Threshold: Proportion of positive predictions / total positive samples
            verification = detection / \
                len(os.listdir(os.path.join('app_data', 'validation')))
            verified = verification > verification_threshold

            # Set verification text
            if verified == True:
                self.verification_label.text = 'Verified'
                self.verification_label.color = utils.get_color_from_hex(
                    '#0ceb0c')
                self.verification_label.font_size = '20dp'
            else:
                self.verification_label.text = 'Unverified'
                self.verification_label.font_size = '20dp'
                self.verification_label.color = (207/255, 19/255, 19/255, 1)
        else:
            popup = Popup(title='User Setup Error', content=Label(text='Minimum of 10 validation images needed.\nPlease setup user by following User Guide.'),
                          size_hint=(.5, .5), size=(250, 250))
            popup.open()

        return None

    def setup(self, *args):
        if self.setupButton.text == "Setup User":
            import shutil
            VALIDATION_PATH = os.path.join('app_data', 'validation')
            validation_images = os.listdir(VALIDATION_PATH)

            if len(validation_images) > 0:
                try:
                    shutil.rmtree(VALIDATION_PATH)
                except OSError as e:
                    print("Error: %s : %s" % (VALIDATION_PATH, e.strerror))

            # Check if validation image path exists
            pathExists = os.path.exists(VALIDATION_PATH)
            if not pathExists:
                os.makedirs(VALIDATION_PATH)

            self.validation_imgs.text = "Stored Validation images: 0"

            self.vImgCount = len(os.listdir(
                os.path.join('app_data', 'validation')))
            if self.vImgCount == 0:
                # Blit info on taking at least 10 imgs
                m = Popup(title='New User Setup', content=Label(text='Setting up a new user.\nPlease setup user by following User Guide.\nNeed to take at least 10 photos\nfor best results.'),
                          size_hint=(.5, .5), size=(250, 250))
                m.open()
                self.setupButton.text = "Take Photo"

        elif self.setupButton.text == "Take Photo":
            validation_img = os.path.join(
                'app_data', 'validation', f"{self.vImgCount}_verif_img.jpg")
            self.takeImg(validation_img)
            faces = self.getFace(validation_img)
            if len(faces) == 1:
                self.vImgCount += 1
                self.validation_imgs.text = "Stored Validation images: {i}".format(
                    i=str(self.vImgCount))
            else:
                os.remove(validation_img)
                f = Popup(title='Face Not Found', content=Label(text='One face necessary for validation images\nPlease take photo when blue box\nhighlights face'),
                          size_hint=(.5, .5), size=(250, 250))
                f.open()
        return None


if __name__ == '__main__':
    FaceID().run()
