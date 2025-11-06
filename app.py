from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import pyttsx3

# -------------------------------
# Suppress Mediapipe / protobuf warnings
# -------------------------------
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead."
)

# -------------------------------
# Flask + SocketIO setup
# -------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# -------------------------------
# Load the trained model
# -------------------------------
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -------------------------------
# Initialize text-to-speech engine
# -------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 1.0)

last_spoken = ""  # to prevent repeating same word continuously

# -------------------------------
# Homepage route
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected')


# -------------------------------
# Generate frames for live video + gesture prediction
# -------------------------------
def generate_frames():
    global last_spoken

    # Select working camera index (1 works for you)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

    if not cap.isOpened():
        print("❌ Could not open camera at index 0.")
        return
    else:
        print("✅ Camera opened successfully at index 0.")

    # Mediapipe setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Gesture labels
    labels_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
        26: 'Hello', 27: 'Done', 28: 'Thank You', 29: 'I Love you',
        30: 'Sorry', 31: 'Please', 32: 'You are welcome.'
    }

    print("🎥 Video streaming started...")

    while True:
        data_aux, x_, y_ = [], [], []

        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    prediction_proba = model.predict_proba([np.asarray(data_aux)])
                    confidence = max(prediction_proba[0])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Emit prediction to frontend
                    socketio.emit('prediction', {'text': predicted_character, 'confidence': confidence})

                    # Draw box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_character} ({confidence*100:.2f}%)",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 0), 3, cv2.LINE_AA)

                    # 🗣️ Automatically speak when new and confident
                    if confidence > 0.80 and predicted_character != last_spoken:
                        last_spoken = predicted_character
                        print(f"🗣️ Speaking: {predicted_character}")
                        try:
                            engine.say(predicted_character)
                            engine.runAndWait()
                        except Exception as e:
                            print("⚠️ TTS error:", e)

                except Exception as e:
                    print("Prediction error:", e)

        # Stream the processed frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Flask video feed route
# -------------------------------
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == '__main__':
    print("🚀 Starting Sign2Text Flask App...")
    socketio.run(app, debug=True)
