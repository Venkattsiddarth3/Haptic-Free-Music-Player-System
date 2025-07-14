# Haptic-Free-Music-Player-System

This project is a **contactless, haptic-free music player control system which uses Computer Vision** to navigate through **hand gestures** detected through your webcam to control media playback and volume on your computer. It leverages **OpenCV**, **MediaPipe**, and **PyAutoGUI** to enable you to interact with your music player without any physical contact.

---

## ✨ Features

- 🚀 **Volume Control using Hand Gestures**
  - Open left palm to enable/disable volume control mode.
  - Pinch (move thumb & index finger) to increase/decrease system volume.

- 🎧 **Playback Control using Face & Hand**
  - Tap left ear to **Play/Pause**.
  - Double tap right ear to go to **Previous Track**.
  - Double tap left ear to go to **Next Track**.

- 🔇 **Mute Control**
  - Touch your mouth with hand to toggle mute.

- 📊 **On-screen Indicators**
  - Shows volume bar, current volume level, and action messages.
  - Displays whether **Volume Access** is enabled.

 ## Dependencies
**Opencv-python :** For video capture & drawing.

 **Mediapipe :** For hand fingers landmark detection.

 **Numpy :** Array math.

 **Pyautogui :** Simulate key presses to control system volume & media.
 
**Install them via:**

```bash
pip install opencv-python mediapipe numpy pyautogui
```

## ✋ Controls & Gestures

| Gesture                          | Action                    |
|----------------------------------|---------------------------|
| Open left palm                   | Toggle volume control mode|
| Thumb + Index pinch movement     | Increase / Decrease volume|
| Touch mouth                      | Toggle mute               |
| Tap left ear                     | Play / Pause              |
| Double tap right ear quickly     | Previous track            |
| Double tap left ear quickly      | Next track                |


## Demo Video :
https://www.linkedin.com/posts/m-v-siddardha-314546315_ai-gesturecontrol-machinelearning-activity-7311813831238856704-yldZ?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE_9-kAB10T5ApOag3wtf0gT0kGqEERgAW8 





