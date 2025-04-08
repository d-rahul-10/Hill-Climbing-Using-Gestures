import cv2
import mediapipe as mp
from pynput.keyboard import Controller, Key
from time import sleep

# கீபோர்டு கட்டுப்படுத்தி (Key presses simulate செய்ய)
keyboard = Controller()

# MediaPipe கைகள் தீர்வு (Hands solution)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)  # அதிகரிக்கப்பட்ட நம்பிக்கை
mp_drawing = mp.solutions.drawing_utils

# வெப்கேம் தொடங்குவது
cap = cv2.VideoCapture(0)

# கீ-பொருள்களின் நிலையைச் செலுத்த
is_gas_pressed = False
is_brake_pressed = False

# மூடிய கை கண்டறிதல் (Fist gesture)
def is_fist(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    little_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # அனைத்து விரல்கள் உள்ளே சுருண்டு இருப்பதைக் கண்டறிதல் (fist)
    if (index_tip.y > landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and 
        middle_tip.y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and 
        ring_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and 
        little_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
        return True
    return False

# திறந்த கை கண்டறிதல் (Open hand gesture)
def is_open_hand(landmarks):
    # விரல்கள் மற்றும் கை குறியீடுகளைப் பெறுதல்
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    little_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # விரல்கள் இடையே தூரம் கணக்கிடல்
    thumb_index_distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    index_middle_distance = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)
    middle_ring_distance = abs(middle_tip.x - ring_tip.x) + abs(middle_tip.y - ring_tip.y)
    ring_little_distance = abs(ring_tip.x - little_tip.x) + abs(ring_tip.y - little_tip.y)

    # திறந்த கை கண்டறிதல் (விரல்கள் போதுமான தூரத்தில் பிரிந்துள்ளன)
    if thumb_index_distance > 0.05 and index_middle_distance > 0.05 and middle_ring_distance > 0.05 and ring_little_distance > 0.05:
        return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # படம் திருப்புதல் (mirror effect)
    frame = cv2.flip(frame, 1)

    # படம் RGB ஆக மாற்றுதல்
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # கைகளை கண்டறிதல்
    results = hands.process(rgb_frame)

    # கைகள் கண்டறியப்பட்டால்
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # கைகளின் நிலைகளை காட்டுதல்
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # **Brake** கீயை (Left Arrow) Fist gesture மூலம் அழுத்துதல்
            if is_fist(landmarks):
                if not is_brake_pressed:  # மீண்டும் மீண்டும் பிரஸ் செய்யாதிருக்க
                    print("Fist gesture detected: Brake")
                    keyboard.press(Key.left)  # Brake action (left arrow)
                    is_brake_pressed = True
                    is_gas_pressed = False  # Gas ஐ அழுத்தாமல் இருக்க
                sleep(0.1)  # சிறிய தாமதம்

            # **Gas** கீயை (Right Arrow) Open hand gesture மூலம் அழுத்துதல்
            elif is_open_hand(landmarks):
                if not is_gas_pressed:  # மீண்டும் மீண்டும் பிரஸ் செய்யாதிருக்க
                    print("Open hand gesture detected: Gas")
                    keyboard.press(Key.right)  # Gas action (right arrow)
                    is_gas_pressed = True
                    is_brake_pressed = False  # Brake ஐ அழுத்தாமல் இருக்க
                sleep(0.1)  # சிறிய தாமதம்

            # கை fist அல்லது open hand இல்லையெனில், கீஸ்களை விடுதல்
            else:
                if is_gas_pressed:
                    keyboard.release(Key.right)  # Gas key விடுதல்
                    is_gas_pressed = False
                if is_brake_pressed:
                    keyboard.release(Key.left)  # Brake key விடுதல்
                    is_brake_pressed = False

    # கைகளை காட்டும் படம்
    cv2.imshow("Hand Gesture Control", frame)

    # 'q' அழுத்தி வெளியேற
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# வெப்கேம் மற்றும் அனைத்து OpenCV விண்டோவை மூடுதல்
cap.release()
cv2.destroyAllWindows()
