import streamlit as st
import hashlib, json, os, cv2
import numpy as np
import pyttsx3, threading
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
USER_DB_PATH = "users.json"
def load_users():
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "w") as f: json.dump({}, f)
    with open(USER_DB_PATH, "r") as f: return json.load(f)
def save_users(users):
    with open(USER_DB_PATH, "w") as f: json.dump(users, f)
def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()
def register_user(username, password):
    users = load_users()
    if username in users: return False, "Username already exists."
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful! Please login."
def verify_user(username, password):
    users = load_users()
    if username not in users: return False, "Username does not exist."
    if users[username] != hash_password(password): return False, "Incorrect password."
    return True, "Login successful."
engine = pyttsx3.init()
engine_lock = threading.Lock()
last_spoken = {"elbow": None, "knee": None, "posture": None}
def set_language(language):
    voices = engine.getProperty('voices')
    if language == "Hindi":
        for voice in voices:
            if "hi" in voice.languages[0].decode('utf-8'):
                engine.setProperty('voice', voice.id); break
    else:
        engine.setProperty('voice', voices[0].id)
def speak_in_thread(message):
    def speak():
        with engine_lock:
            engine.say(message)
            engine.runAndWait()
    threading.Thread(target=speak, daemon=True).start()
def speak_once(category, message):
    if last_spoken.get(category) != message:
        speak_in_thread(message)
        last_spoken[category] = message
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(np.degrees(radians) % 360)
smoothed_points = {}
def smooth_point(label, new_point, alpha=0.6):
    if label not in smoothed_points or smoothed_points[label] is None:
        smoothed_points[label] = new_point
    else:
        old_x, old_y = smoothed_points[label]
        new_x, new_y = new_point
        smoothed_points[label] = (
            int(alpha * new_x + (1 - alpha) * old_x),
            int(alpha * new_y + (1 - alpha) * old_y),
        )
    return smoothed_points[label]
def process_frame(frame, net):
    inWidth, inHeight = 368, 368
    frameHeight, frameWidth = frame.shape[:2]
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    inp_blob = cv2.dnn.blobFromImage(frame_blur, 1.0, (inWidth, inHeight),
                                     (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inp_blob)
    out = net.forward()
    POSE_PAIRS = [
        (1, 2), (2, 3), (3, 4),     # Right Arm
        (1, 5), (5, 6), (6, 7),     # Left Arm
        (1, 8), (8, 9), (9, 10),    # Right Leg
        (1, 11), (11, 12), (12, 13) # Left Leg
    ]
    points = []
    for i in range(18):
        prob_map = out[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(prob_map)
        x = int((point[0] * frameWidth) / out.shape[3])
        y = int((point[1] * frameHeight) / out.shape[2])
        if prob > 0.2:
            if i in [2, 5, 9, 12]:  # shoulders & knees
                smoothed = smooth_point(str(i), (x, y))
                points.append(smoothed)
            else:
                points.append((x, y))
            cv2.circle(frame, points[-1], 5, (0, 255, 0), -1)
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partA, partB = pair
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255, 255, 0), 2)

    # Calculate angles
    elbow_angle = (
        calculate_angle(points[2], points[3], points[4])
        if points[2] and points[3] and points[4] else 0
    )
    knee_angle = (
        calculate_angle(points[8], points[9], points[10])
        if points[8] and points[9] and points[10] else 0
    )
    score = max(0, 100 - abs(135 - elbow_angle) - abs(90 - knee_angle))
    return frame, elbow_angle, knee_angle, score
def plot_to_image(elbow_vals, knee_vals):
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(elbow_vals, label="Elbow Angle", color='blue')
    ax.plot(knee_vals, label="Knee Angle", color='green')
    ax.set_ylim([0, 180])
    ax.legend(); ax.set_title("Pose Angle Progress")
    ax.set_xlabel("Frames"); ax.set_ylabel("Angle (degrees)")
    fig.tight_layout()
    canvas = FigureCanvas(fig); canvas.draw()
    buf = canvas.buffer_rgba(); image = np.asarray(buf)
    plt.close(fig)
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
def give_feedback(elbow, knee, elbow_thresh, knee_thresh, language):
    feedback = ""
    if elbow > elbow_thresh:
        feedback += "Elbow too straight!\n"
        speak_once("elbow", "Elbow too straight!" if language == "English" else "कोहनी बहुत सीधी है")
    if knee < knee_thresh:
        feedback += "Knee too bent!\n"
        speak_once("knee", "Knee too bent!" if language == "English" else "घुटना बहुत मुड़ा हुआ है")
    if abs(elbow - 135) < 10 and abs(knee - 90) < 10:
        feedback += "Perfect posture!\n"
        speak_once("posture", "Perfect posture!" if language == "English" else "सही मुद्रा!")
    return feedback
st.set_page_config(layout="wide")
st.title("🏀 Basketball Pose Coach")
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = ""
if "page" not in st.session_state: st.session_state.page = "login"
def login_page():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        valid, msg = verify_user(username, password)
        if valid:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
        else:
            st.error(msg)
    if st.button("Go to Register"):
        st.session_state.page = "register"
def register_page():
    st.subheader("Register")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Register"):
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            success, msg = register_user(username, password)
            if success:
                st.success(msg); st.session_state.page = "login"
            else:
                st.error(msg)
    if st.button("Go to Login"):
        st.session_state.page = "login"
if not st.session_state.logged_in:
    login_page() if st.session_state.page == "login" else register_page()
else:
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()
    net = cv2.dnn.readNetFromTensorflow("C:/Users/navee/Downloads/graph_opt.pb")
    mode = st.sidebar.radio("Select Mode", ["Image", "Video", "Webcam"])
    language = st.sidebar.selectbox("Voice Language", ["English", "Hindi"])
    set_language(language)
    elbow_thresh = st.sidebar.slider("Elbow Max Angle", 100, 180, 160)
    knee_thresh = st.sidebar.slider("Knee Min Angle", 60, 120, 90)
    if "webcam_running" not in st.session_state: st.session_state.webcam_running = False
    if "video_running" not in st.session_state: st.session_state.video_running = False
    if mode == "Webcam":
        if st.sidebar.button("Start Webcam"): st.session_state.webcam_running = True
        if st.sidebar.button("Stop Webcam"): st.session_state.webcam_running = False
    if mode == "Video":
        if st.sidebar.button("Start Video"): st.session_state.video_running = True
        if st.sidebar.button("Stop Video"): st.session_state.video_running = False
    elbow_vals = deque(maxlen=100)
    knee_vals = deque(maxlen=100)
    frame_display = st.empty()
    if mode == "Image":
     file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
     if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        frame, elbow, knee, score = process_frame(image, net)
        feedback = give_feedback(elbow, knee, elbow_thresh, knee_thresh, language)
        elbow_vals.append(elbow)
        knee_vals.append(knee)
        plot_img = plot_to_image(elbow_vals, knee_vals)
        plot_img_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
        cols = st.columns(2)
        with cols[0]:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
            st.write(f"Elbow Angle: {int(elbow)}")
            st.write(f"Knee Angle: {int(knee)}")
            st.write(f"Score: {int(score)}")
            st.text(feedback)
        with cols[1]:
            st.image(plot_img_rgb, caption="Pose Angle Progress", use_column_width=True)
    elif mode == "Video":
        file = st.sidebar.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        if file:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f: f.write(file.read())
            cap = cv2.VideoCapture(temp_path)
            while cap.isOpened() and st.session_state.video_running:
                ret, frame = cap.read()
                if not ret: break
                frame, elbow, knee, score = process_frame(frame, net)
                feedback = give_feedback(elbow, knee, elbow_thresh, knee_thresh, language)
                st.markdown(f"**Feedback:** {feedback.replace(chr(10), '<br>')}", unsafe_allow_html=True)
                elbow_vals.append(elbow)
                knee_vals.append(knee)
                plot_img = plot_to_image(elbow_vals, knee_vals)
                plot_img = cv2.resize(plot_img, (frame.shape[1], frame.shape[0]))
                combined = np.hstack((frame, plot_img))
                frame_display.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB")
            cap.release()
            os.remove(temp_path)
    elif mode == "Webcam":
        cap = cv2.VideoCapture(0)
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            if not ret: break
            frame, elbow, knee, score = process_frame(frame, net)
            feedback = give_feedback(elbow, knee, elbow_thresh, knee_thresh, language)
            st.markdown(f"**Feedback:** {feedback.replace(chr(10), '<br>')}", unsafe_allow_html=True)
            elbow_vals.append(elbow)
            knee_vals.append(knee)
            plot_img = plot_to_image(elbow_vals, knee_vals)
            plot_img = cv2.resize(plot_img, (frame.shape[1], frame.shape[0]))
            combined = np.hstack((frame, plot_img))
            frame_display.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()
