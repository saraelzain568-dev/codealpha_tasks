import streamlit as st
from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="AI Language Translator",
    page_icon="🌷",
    layout="centered"
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(
                135deg,
                #f7d6e9 0%,
                #e4d4f4 50%,
                #d8d9f2 100%
            );
        }

        .main-title {
            text-align: center;
            font-size: 43px;
            font-weight: 800;
            color: #70406f;
            margin-top: 20px;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #765878;
            font-size: 17px;
            margin-bottom: 30px;
        }

        .block-container {
            max-width: 850px;
            padding-top: 2rem;
        }

        textarea {
            border-radius: 14px !important;
            background-color: #f9eaf4 !important;
            border: 1px solid #c99bc3 !important;
            color: #583d5c !important;
        }

        textarea:focus {
            border: 2px solid #a96aa5 !important;
            box-shadow: 0 0 10px rgba(169, 106, 165, 0.25);
        }

        div[data-baseweb="select"] > div {
            border-radius: 13px !important;
            background-color: #f2dced !important;
            border: 1px solid #c48cbd !important;
            color: #643f63 !important;
        }

        div[data-baseweb="select"] > div:hover {
            border-color: #9d6199 !important;
        }

        div[data-baseweb="popover"] {
            background-color: #f5e3f0 !important;
            border-radius: 13px !important;
        }

        div[role="option"] {
            color: #643f63 !important;
            background-color: #f5e3f0 !important;
        }

        div[role="option"]:hover {
            background-color: #dfc2df !important;
        }

        .stButton > button {
            border-radius: 13px !important;
            height: 48px;
            font-size: 16px;
            font-weight: 700;
            background-color: #c98abb !important;
            color: #ffffff !important;
            border: 1px solid #ae70a4 !important;
            box-shadow: 0 3px 8px rgba(112, 64, 111, 0.15);
        }

        .stButton > button:hover {
            background-color: #b976ad !important;
            border-color: #9e5f96 !important;
            color: #ffffff !important;
        }

        .stButton > button:active {
            background-color: #a8649d !important;
        }

        div[data-testid="stAlert"] {
            background-color: #e6c9e3 !important;
            color: #613f60 !important;
            border: 1px solid #bd91b8 !important;
            border-radius: 13px !important;
        }

        div[data-testid="stAlert"] svg {
            color: #8c5587 !important;
        }

        .result-title {
            color: #70406f;
            font-size: 22px;
            font-weight: 750;
            margin-top: 10px;
        }

        .stCodeBlock {
            border-radius: 12px !important;
            border: 1px solid #c99bc3 !important;
        }

        hr {
            border: none !important;
            height: 1px !important;
            background: #c99bc3 !important;
            margin-top: 30px !important;
        }

        .footer {
            text-align: center;
            color: #765878;
            font-size: 13px;
            margin-top: 25px;
            padding-bottom: 15px;
        }

        label {
            color: #70406f !important;
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🌷 AI Language Translator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Translate text quickly and easily using AI'
    '</div>',
    unsafe_allow_html=True
)

languages = {
    "English": "en",
    "Arabic": "ar",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Turkish": "tr",
    "Hindi": "hi",
    "Portuguese": "pt",
    "Russian": "ru"
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "translation" not in st.session_state:
    st.session_state.translation = ""

if "source_language" not in st.session_state:
    st.session_state.source_language = "English"

if "target_language" not in st.session_state:
    st.session_state.target_language = "Arabic"

if "translation_version" not in st.session_state:
    st.session_state.translation_version = 0

def clear_fields():
    st.session_state.input_text = ""
    st.session_state.translation = ""
    st.session_state.translation_version += 1

col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox(
        "🔤 Source Language",
        list(languages.keys()),
        key="source_language"
    )

with col2:
    target_language = st.selectbox(
        "🌐 Target Language",
        list(languages.keys()),
        key="target_language"
    )

text = st.text_area(
    "✍️ Enter your text",
    height=150,
    placeholder="Type the text you want to translate...",
    key="input_text"
)

col1, col2 = st.columns(2)

with col1:
    translate_button = st.button(
        "🔄 Translate",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
        on_click=clear_fields
    )

if translate_button:
    if not text.strip():
        st.warning("⚠️ Please enter some text first.")

    elif source_language == target_language:
        st.info("ℹ️ Source and target languages are the same.")

    else:
        try:
            with st.spinner("Translating..."):
                translation = GoogleTranslator(
                    source=languages[source_language],
                    target=languages[target_language]
                ).translate(text)

            st.session_state.translation = translation
            st.session_state.translation_version += 1

            st.success(
                "🌷 Translation completed successfully!"
            )

        except Exception:
            st.error(
                "❌ Something went wrong while translating."
            )

            st.info(
                "Please check your internet connection and try again."
            )

if st.session_state.translation:
    st.markdown(
        '<div class="result-title">📌 Translated Text</div>',
        unsafe_allow_html=True
    )

    st.text_area(
        "Result",
        value=st.session_state.translation,
        height=150,
        key=f"translation_result_{st.session_state.translation_version}"
    )

    st.write("📋 Copy the translation:")

    st.code(
        st.session_state.translation,
        language=None
    )

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'CodeAlpha Internship — AI Language Translation Tool'
    '</div>',
    unsafe_allow_html=True
)