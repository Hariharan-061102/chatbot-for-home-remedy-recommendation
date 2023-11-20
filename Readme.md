# Emotion detection 


This project is a Streamlit application that recommends songs based on users' captured emotions. It uses computer vision techniques to detect emotions from video frames and provides song recommendations on YouTube or Spotify.

## Features

- Captures users' emotions through the webcam.
- Performs real-time emotion classification using a pre-trained model.
- Overlays the predicted emotions on the video frames.
- Recommends songs on YouTube or Spotify based on the captured emotions, language, and singer preferences.
- Opens a web browser window with relevant search results for song recommendations.

## Requirements

- Python 3.x
- Required Python packages: streamlit, streamlit_webrtc, av, opencv-python, numpy, mediapipe, keras, webbrowser

## Architecture Diagram/Flow

![Diagram1](diagram1.png)
![Diagram2](diagram2.png)

## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/emotion-based-music-recommender.git

2. Install the required packages:

   ```shell
   pip install -r requirements.txt

3. Download the pre-trained emotion classification model and label mappings.
   (Place the model.h5 file and labels.npy file in the project directory.)

## Usage

1. Run the Streamlit application
   ```shell
   streamlit run app.py
   ```

2. Access the application in your web browser at http://localhost:8501.

3. Enter the desired language, singer, and music player preferences.

4. Allow access to your webcam.

5. The application will start capturing your emotions in real-time.

6. Click the "Recommend me songs" button to get song recommendations based on your captured emotions.

7. A web browser window will open with search results on YouTube or Spotify, depending on your music player preference.

8. Repeat the process by providing new inputs and capturing emotions again.

## Program:

```python
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Emotion Based Music Recommender")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not (emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"


class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1,
                                                                         circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        ##############################

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


lang = st.text_input("Language")
singer = st.text_input("singer")
music_player = st.selectbox("Music Player", ["YouTube", "Spotify"])


if lang and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key", desired_playing_state=True,
                    video_processor_factory=EmotionProcessor)

btn = st.button("Recommend me songs")
if btn:
    if not emotion:
        st.warning("Please let me capture your emotion first")
        st.session_state["run"] = "true"
    else:
        if music_player == "YouTube":
            webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
        elif music_player == "Spotify":
            webbrowser.open(f"https://open.spotify.com/search/{lang} {emotion} song {singer}")
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = "false"
```
## Output:

![output1](output1.png)
![output2](output2.png)

## Result:

The Emotional sensing music therapy project is a real-time application that captures users' emotions through their webcam and provides personalized song recommendations. By leveraging computer vision techniques and a pre-trained emotion classification model, the application accurately detects users' emotions and overlays them on the live video stream.

With the Emotional sensing music therapy project, users can explore a personalized music playlist tailored to their emotions, language, and preferred artist. Whether they want to discover new songs or find comfort in familiar melodies, this project enhances the music listening experience by leveraging the power of computer vision and machine learning.

The project is a valuable tool for music enthusiasts, researchers, and developers interested in emotion recognition, recommendation systems, and human-computer interaction.


