# VisioPointer-AI-Driven-Gaze-Navigation-System
VisioPointer is an assistive technology project that enables hands-free computer interaction using eye gaze tracking, blink detection, and voice commands.
It combines computer vision (MediaPipe), speech recognition (Vosk), and automation (PyAutoGUI) to create an accessible system for people with motor impairments or those who want a futuristic way of navigating their computer.

✨ Key Features

👀 Eye Tracking – Move the mouse cursor based on gaze direction.

🖱️ Blink-to-Click – Perform mouse clicks with intentional blinks.

🕒 Dwell Click – Automatically click if the user looks at one spot for 2 seconds.

⬆️⬇️ Smart Scrolling – Scroll up or down by eye position.

🎤 Voice Commands – Control the system with simple commands like:

"click" → Mouse click

"scroll up" / "scroll down" → Page navigation

"exit" → Close the program

🧠 Fuzzy Command Matching – Handles mispronunciations with fuzzy matching.

🛠️ Tech Stack

OpenCV + MediaPipe → Face mesh and eye landmark tracking

PyAutoGUI → Mouse & keyboard automation

Vosk + PyAudio → Offline speech recognition

Threading → Parallel voice and gaze control

FuzzyWuzzy → Robust voice command matching

🎯 Use Cases

Accessibility tool for users with motor disabilities

Hands-free navigation in sterile or restricted environments

Experimental project in HCI (Human-Computer Interaction)
