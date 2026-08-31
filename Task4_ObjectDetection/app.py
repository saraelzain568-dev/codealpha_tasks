import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

st.set_page_config(
    page_title="AI Object Detection & Tracking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background:
            radial-gradient(circle at 10% 10%, rgba(196, 181, 253, 0.55), transparent 32%),
            radial-gradient(circle at 90% 90%, rgba(221, 214, 254, 0.65), transparent 35%),
            linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 48%, #DDD6FE 100%) !important;
        color: #35265C !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 1150px !important;
        padding-top: 40px !important;
        padding-bottom: 100px !important;
    }

    .title {
        text-align: center;
        color: #5B3E96 !important;
        font-size: 46px !important;
        font-weight: 800 !important;
        margin-bottom: 8px;
        text-shadow: 0 5px 18px rgba(91, 62, 150, 0.15);
    }

    .subtitle {
        text-align: center;
        color: #8069B5 !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 22px;
    }

    .status {
        width: fit-content;
        margin: 0 auto 35px auto;
        padding: 9px 22px;
        border-radius: 30px;
        background: #FFFFFF !important;
        border: 1px solid #C4B5FD;
        color: #6748A3 !important;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 0 8px 25px rgba(91, 62, 150, 0.12);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.82) !important;
        border: 2px dashed #A78BFA !important;
        border-radius: 24px !important;
        padding: 18px !important;
        box-shadow: 0 12px 35px rgba(91, 62, 150, 0.12);
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    [data-testid="stFileUploader"] label {
        color: #5B3E96 !important;
        font-weight: 700 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #7C5CBF !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #6949AA !important;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #C4B5FD;
        border-radius: 24px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 12px 35px rgba(91, 62, 150, 0.10);
        text-align: center;
    }

    .info-title {
        color: #5B3E96;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .info-text {
        color: #705D99;
        font-size: 15px;
        line-height: 1.8;
    }

    .result-title {
        color: #5B3E96 !important;
        font-size: 25px !important;
        font-weight: 800 !important;
        margin-top: 25px !important;
        margin-bottom: 15px !important;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid #C4B5FD;
        border-radius: 18px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(91, 62, 150, 0.10);
    }

    .metric-number {
        color: #65479D;
        font-size: 25px;
        font-weight: 800;
    }

    .metric-label {
        color: #8069B5;
        font-size: 13px;
        margin-top: 4px;
    }

    .stButton > button {
        background: linear-gradient(135deg, #8B6BC7, #7051AE) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #7959B8, #62459E) !important;
    }

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #EDE9FE;
    }

    ::-webkit-scrollbar-thumb {
        background: #A78BFA;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #8B6BC7;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 16px !important;
            padding-right: 16px !important;
            padding-top: 25px !important;
        }

        .title {
            font-size: 34px !important;
        }

        .subtitle {
            font-size: 15px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">🔍 AI Object Detection & Tracking</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">YOLO • Computer Vision • Real-Time Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status">● AI Detection System Ready</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <div class="info-title">🤖 Welcome to AI Object Detection</div>
        <div class="info-text">
            Upload an image or video and let the AI detect objects automatically.
            <br>
            Images are processed for object detection, while videos can be used
            for object tracking.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("⚙️ Model Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
line_width = st.sidebar.slider("Line Width", 1, 5, 2)
font_size = st.sidebar.slider("Font Size", 0.4, 2.0, 1.0, 0.1)

uploaded_file = st.file_uploader(
    "📁 Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file:

    model = YOLO("yolo11n.pt")

    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension in [".jpg", ".jpeg", ".png"]:

        file_bytes = uploaded_file.read()

        image_array = cv2.imdecode(
            np.frombuffer(file_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image_array is not None:

            results = model(image_array, conf=conf_threshold)

            annotated_image = results[0].plot(
                line_width=line_width,
                font_size=font_size,
                pil=False
            )

            st.markdown(
                '<div class="result-title">🎯 Detection Result</div>',
                unsafe_allow_html=True
            )

            st.image(
                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            detected_objects = results[0].boxes.cls

            object_count = len(detected_objects) if detected_objects is not None else 0

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-number">{object_count}</div>
                        <div class="metric-label">Detected Objects</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-number">YOLO</div>
                        <div class="metric-label">Detection Model</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        )

        temp_file.write(uploaded_file.read())
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)

        st.markdown(
            '<div class="result-title">🎯 Tracking Result</div>',
            unsafe_allow_html=True
        )

        frame_placeholder = st.empty()

        frame_count = 0

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            results = model.track(
                frame,
                conf=conf_threshold,
                persist=True,
                verbose=False
            )

            annotated_frame = results[0].plot(
                line_width=line_width,
                font_size=font_size,
                pil=False
            )

            frame_placeholder.image(
                cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                ),
                channels="RGB",
                use_container_width=True
            )

            frame_count += 1

        cap.release()

        os.unlink(temp_file.name)

        st.success(
            f"Video processing completed successfully. {frame_count} frames processed."
        )