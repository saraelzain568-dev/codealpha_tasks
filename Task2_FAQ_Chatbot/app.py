import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from google import genai

nltk.download("stopwords", quiet=True)

st.set_page_config(
    page_title="AI FAQ Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
    [data-testid="stMain"], [data-testid="stMainBlockContainer"] {
        background:
            radial-gradient(circle at 10% 10%, rgba(143,49,92,.55), transparent 32%),
            radial-gradient(circle at 90% 90%, rgba(169,58,105,.45), transparent 32%),
            linear-gradient(135deg,#350D1F 0%,#50132E 48%,#691B3E 100%) !important;
        color:#FFFFFF !important;
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    [data-testid="stBottomBlockContainer"] {
        background:transparent !important;
    }

    .block-container {
        max-width:1100px !important;
        padding-top:40px !important;
        padding-bottom:120px !important;
    }

    .title-text {
        color:#FFFFFF !important;
        text-align:center !important;
        font-size:48px !important;
        font-weight:800 !important;
        margin-bottom:8px !important;
        text-shadow:0 5px 20px rgba(0,0,0,.25);
    }

    .subtitle-text {
        color:#F8C9DA !important;
        text-align:center !important;
        font-size:18px !important;
        font-weight:500 !important;
        margin-bottom:18px !important;
    }

    .status-box {
        width:fit-content;
        margin:0 auto 38px auto;
        padding:10px 22px;
        border-radius:50px;
        background:linear-gradient(135deg,rgba(166,57,103,.65),rgba(110,29,62,.75)) !important;
        border:1px solid #D88BA8;
        color:#FFFFFF !important;
        font-size:14px;
        font-weight:700;
        box-shadow:0 8px 25px rgba(20,0,10,.30);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:linear-gradient(145deg,rgba(108,28,60,.96),rgba(62,13,32,.98)) !important;
        border:1px solid rgba(220,120,157,.55) !important;
        border-radius:26px !important;
        box-shadow:0 15px 40px rgba(18,0,9,.30) !important;
        padding:8px !important;
    }

    .welcome-title {
        color:#FFFFFF !important;
        font-size:27px !important;
        font-weight:800 !important;
        text-align:center !important;
        margin:5px 0 15px 0 !important;
    }

    .welcome-text {
        color:#FFEAF2 !important;
        font-size:16px !important;
        line-height:1.8 !important;
        text-align:center !important;
    }

    .feature-card {
        min-height:125px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        text-align:center;
        border-radius:22px;
        background:linear-gradient(145deg,rgba(139,43,80,.92),rgba(78,18,42,.97));
        border:1px solid rgba(225,133,165,.55);
        box-shadow:0 10px 28px rgba(20,0,10,.28);
        margin-top:18px;
        margin-bottom:25px;
        padding:18px 10px;
    }

    .feature-icon {
        font-size:30px;
        margin-bottom:7px;
    }

    .feature-title {
        color:#FFFFFF !important;
        font-size:15px !important;
        font-weight:800 !important;
    }

    .feature-description {
        color:#F4C6D6 !important;
        font-size:12px !important;
        margin-top:5px !important;
    }

    [data-testid="stChatMessage"] {
        background:linear-gradient(145deg,rgba(108,28,60,.98),rgba(64,13,33,.99)) !important;
        border:1px solid rgba(218,117,154,.58) !important;
        border-radius:22px !important;
        padding:18px !important;
        margin:15px 0 !important;
        box-shadow:0 9px 28px rgba(20,0,10,.28) !important;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong {
        color:#FFFFFF !important;
    }

    [data-testid="stChatMessage"] p {
        font-size:17px !important;
        line-height:1.85 !important;
    }

    [data-testid="stChatMessage"] li {
        font-size:16px !important;
        line-height:1.75 !important;
    }

    .chat-badges {
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin-top:13px;
    }

    .chat-badge {
        display:inline-flex;
        align-items:center;
        width:fit-content;
        padding:7px 14px;
        border-radius:20px;
        background:linear-gradient(135deg,rgba(190,82,124,.45),rgba(124,38,73,.55));
        border:1px solid rgba(235,145,177,.65);
        color:#FFFFFF !important;
        font-size:13px;
        font-weight:700;
    }

    [data-testid="stChatInput"] {
        background:linear-gradient(135deg,#5B1734,#701F45) !important;
        border:2px solid #A94E73 !important;
        border-radius:28px !important;
        box-shadow:0 10px 32px rgba(25,0,13,.45) !important;
        padding:5px !important;
    }

    [data-testid="stChatInput"] textarea {
        background:#76264A !important;
        color:#FFFFFF !important;
        border-radius:22px !important;
        font-size:15px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color:#F2B7CD !important;
    }

    [data-testid="stChatInput"] button {
        background:linear-gradient(135deg,#C55A83,#A9436B) !important;
        border:1px solid #E19AB5 !important;
        border-radius:50% !important;
        color:#FFFFFF !important;
    }

    [data-testid="stChatInput"] button:hover {
        background:linear-gradient(135deg,#D36B92,#B94D76) !important;
    }

    [data-testid="stAlert"],
    [data-testid="stNotification"],
    .stSuccess {
        background:#7A2147 !important;
        color:#FFFFFF !important;
        border:2px solid #FFFFFF !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stNotification"] *,
    .stSuccess * {
        color:#FFFFFF !important;
    }

    ::-webkit-scrollbar {
        width:8px;
    }

    ::-webkit-scrollbar-track {
        background:#3A1023;
    }

    ::-webkit-scrollbar-thumb {
        background:#8E3157;
        border-radius:10px;
    }

    @media (max-width:700px) {
        .block-container {
            padding-left:16px !important;
            padding-right:16px !important;
            padding-top:28px !important;
        }

        .title-text {
            font-size:35px !important;
        }

        .subtitle-text {
            font-size:15px !important;
        }

        .welcome-title {
            font-size:22px !important;
        }

        .welcome-text {
            font-size:14px !important;
        }

        [data-testid="stChatMessage"] p {
            font-size:15px !important;
        }

        [data-testid="stChatMessage"] li {
            font-size:14px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title-text">🤖 AI FAQ Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">Smart Answers • Any Language • FAQ + AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="status-box">● AI Assistant Online</div>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown(
            '<div class="welcome-title">🤖 Welcome to your AI Assistant</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="welcome-text">
            Ask your question in Arabic, English, French, Spanish,
            or another language.<br>
            I search the FAQ knowledge base first, then use AI for
            broader questions, science, astronomy, education,
            and programming.
            </div>
            """,
            unsafe_allow_html=True
        )

    cols = st.columns(4, gap="medium")

    features = [
        ("🌍","Any Language","Ask in any language"),
        ("📚","FAQ Search","Find FAQ answers"),
        ("🧠","AI Answers","Smart answers"),
        ("💻","Programming","Coding assistance")
    ]

    for col, feature in zip(cols, features):
        with col:
            icon, title, description = feature

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

faqs = [
    {
        "question":"What is this chatbot?",
        "answer":"This is an AI FAQ chatbot that uses Natural Language Processing to understand questions and find the most relevant answer."
    },
    {
        "question":"How does the chatbot work?",
        "answer":"The chatbot preprocesses the user's question, converts the text into TF-IDF vectors, and uses cosine similarity to find the most relevant FAQ."
    },
    {
        "question":"What technology is used?",
        "answer":"The project uses Python, Streamlit, NLTK, Scikit-learn, TF-IDF, cosine similarity, Google Gemini AI, and text translation tools."
    },
    {
        "question":"What is NLP?",
        "answer":"NLP stands for Natural Language Processing. It is a field of AI that helps computers understand, process, and analyze human language."
    },
    {
        "question":"What is TF-IDF?",
        "answer":"TF-IDF is a technique used to represent text numerically. It measures how important words are within a document compared with a collection of documents."
    },
    {
        "question":"What is cosine similarity?",
        "answer":"Cosine similarity measures how similar two text vectors are. A value close to 1 means the texts are highly similar."
    },
    {
        "question":"What is the second planet from the Sun?",
        "answer":"Venus is the second planet from the Sun."
    },
    {
        "question":"What is the largest planet in the Solar System?",
        "answer":"Jupiter is the largest planet in the Solar System."
    },
    {
        "question":"What is the closest planet to the Sun?",
        "answer":"Mercury is the closest planet to the Sun."
    },
    {
        "question":"How can I learn programming?",
        "answer":"Start with programming fundamentals, learn Python or another beginner-friendly language, practice regularly, build small projects, learn debugging, and gradually move to larger applications."
    }
]

quick_answers = {
    "what python":"Python is a high-level programming language known for its simple syntax. It is widely used in web development, automation, data analysis, artificial intelligence, and machine learning.",
    "what ai":"Artificial Intelligence, or AI, is the field of creating computer systems that can perform tasks that normally require human intelligence, such as understanding language, recognizing patterns, learning, and solving problems.",
    "what machine learning":"Machine Learning is a branch of AI where computers learn patterns from data and use those patterns to make predictions or decisions.",
    "what nlp":"NLP, or Natural Language Processing, is a field of AI that focuses on helping computers understand and process human language.",
    "what tf idf":"TF-IDF is a technique used to represent text numerically by measuring how important words are within documents.",
    "what cosine similarity":"Cosine similarity is a mathematical method used to measure how similar two text vectors are.",
    "what programming":"Programming is the process of writing instructions that tell a computer what to do. It is used to create software, applications, websites, and systems."
}

definition_keywords = {
    "programming": [
        "what programming",
        "what is programming",
        "define programming",
        "definition programming",
        "what does programming mean",
        "ما هي البرمجة",
        "ما هو البرمجة",
        "تعريف البرمجة"
    ],
    "python": [
        "what python",
        "what is python",
        "define python",
        "ما هو بايثون",
        "ما هي بايثون"
    ],
    "ai": [
        "what ai",
        "what is ai",
        "what artificial intelligence",
        "ما هو الذكاء الاصطناعي",
        "ما هي الذكاء الاصطناعي"
    ]
}

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def clean_ai_response(text):
    if not isinstance(text, str):
        return str(text)

    text = re.sub(r"<br\s*/?>","\n",text,flags=re.IGNORECASE)
    text = re.sub(
        r"</?(div|span|p|style|section|article|body|html|head|strong|b|i|em|h1|h2|h3|h4|h5|h6|ul|ol|li|table|tr|td|th)[^>]*>",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"<[^>]+>","",text)

    return text.strip()

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"

def translate_to_english(text, language):
    try:
        if language == "en":
            return text

        return GoogleTranslator(
            source=language,
            target="en"
        ).translate(text)

    except Exception:
        return text

def translate_from_english(text, language):
    try:
        if language == "en":
            return text

        return GoogleTranslator(
            source="en",
            target=language
        ).translate(text)

    except Exception:
        return text

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def generate_ai_response(question, language, messages):
    api_key = get_api_key()

    if not api_key:
        return "Gemini AI is not configured. Please add GEMINI_API_KEY to Streamlit secrets."

    try:
        client = genai.Client(api_key=api_key)

        conversation = ""

        for message in messages[-6:]:
            conversation += (
                "\n"
                + message["role"]
                + ": "
                + message["content"]
            )

        prompt = f"""
You are a helpful multilingual AI assistant.

The user asked:
{question}

The user's language is:
{language}

Always answer in exactly the same language as the user's question.

If the user asks in Arabic, answer in Arabic.
If the user asks in English, answer in English.
If the user asks in French, answer in French.
If the user asks in Spanish, answer in Spanish.

Keep simple answers short and direct.

Do not assume that every programming question means the user wants advice about learning programming.

If the user asks what programming is, explain programming itself.

If the user asks how to learn programming, explain how to learn it.

If the user asks about Python, explain Python.

If the user asks about C++, Java, loops, functions, variables, arrays, classes, debugging, or another programming topic, answer that specific question.

For simple questions, use 1 to 3 sentences.
For complex questions, use short clear sections or bullet points.

Never return HTML.
Never return CSS.
Never return webpage code.
Never use HTML tags.

Previous conversation:
{conversation}

Return only the final answer.
"""

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        if not response.text:
            return "I couldn't generate an answer."

        answer = clean_ai_response(
            response.text.strip()
        )

        if language != "en":
            try:
                if detect(answer) == "en":
                    answer = translate_from_english(
                        answer,
                        language
                    )
            except LangDetectException:
                pass

        return clean_ai_response(answer)

    except Exception:
        return "I couldn't connect to the AI service right now. Please try again."

faq_questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]

vectorizer = TfidfVectorizer(
    ngram_range=(1,2)
)

faq_vectors = vectorizer.fit_transform(
    faq_questions
)

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤"
        ):
            st.markdown(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.markdown(
                clean_ai_response(
                    message["content"]
                ),
                unsafe_allow_html=False
            )

            badges = []

            if "mode" in message:
                badges.append(
                    f'<span class="chat-badge">🤖 {message["mode"]}</span>'
                )

            if "score" in message:
                badges.append(
                    f'<span class="chat-badge">🎯 Similarity: {message["score"]:.2f}</span>'
                )

            if badges:
                st.markdown(
                    '<div class="chat-badges">'
                    + "".join(badges)
                    + "</div>",
                    unsafe_allow_html=True
                )

user_question = st.chat_input(
    "💬 Ask your question in any language..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":user_question
        }
    )

    language = detect_language(
        user_question
    )

    english_question = translate_to_english(
        user_question,
        language
    )

    normalized_question = preprocess_text(
        english_question
    )

    simple_key = normalized_question

    if simple_key in quick_answers:

        answer = translate_from_english(
            quick_answers[simple_key],
            language
        )

        st.session_state.messages.append(
            {
                "role":"assistant",
                "content":clean_ai_response(answer),
                "mode":"Quick Answer"
            }
        )

    else:

        user_words = set(
            normalized_question.split()
        )

        definition_match = False

        if (
            "programming" in user_words
            and (
                "what" in user_words
                or "define" in user_words
                or "meaning" in user_words
            )
        ):
            definition_match = True

        if (
            "python" in user_words
            and (
                "what" in user_words
                or "define" in user_words
            )
        ):
            definition_match = True

        if (
            "artificial" in user_words
            and "intelligence" in user_words
            and (
                "what" in user_words
                or "define" in user_words
            )
        ):
            definition_match = True

        if definition_match:

            answer = generate_ai_response(
                user_question,
                language,
                st.session_state.messages
            )

            st.session_state.messages.append(
                {
                    "role":"assistant",
                    "content":clean_ai_response(answer),
                    "mode":"AI Answer"
                }
            )

        else:

            user_vector = vectorizer.transform(
                [normalized_question]
            )

            similarity_scores = cosine_similarity(
                user_vector,
                faq_vectors
            )

            best_match_index = similarity_scores.argmax()

            best_score = similarity_scores[
                0
            ][best_match_index]

            best_question = faq_questions[
                best_match_index
            ]

            user_words = set(
                normalized_question.split()
            )

            faq_words = set(
                best_question.split()
            )

            common_words = user_words.intersection(
                faq_words
            )

            meaningful_words = {
                word for word in common_words
                if len(word) > 3
            }

            exact_phrase_match = (
                normalized_question == best_question
            )

            strong_match = (
                best_score >= 0.78
                and len(meaningful_words) >= 2
            )

            if exact_phrase_match or strong_match:

                answer = faqs[
                    best_match_index
                ]["answer"]

                answer = translate_from_english(
                    answer,
                    language
                )

                st.session_state.messages.append(
                    {
                        "role":"assistant",
                        "content":clean_ai_response(answer),
                        "score":best_score,
                        "mode":"FAQ Match"
                    }
                )

            else:

                with st.spinner(
                    "🤖 Thinking..."
                ):

                    answer = generate_ai_response(
                        user_question,
                        language,
                        st.session_state.messages
                    )

                st.session_state.messages.append(
                    {
                        "role":"assistant",
                        "content":clean_ai_response(answer),
                        "mode":"AI Answer"
                    }
                )

    st.rerun()