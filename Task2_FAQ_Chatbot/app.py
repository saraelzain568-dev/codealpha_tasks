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
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.markdown(
    """
    <style>

    html,
    body,
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(
            135deg,
            #E4CDD3 0%,
            #D7B8C1 50%,
            #C99DAA 100%
        ) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stMain"] {
        background: transparent !important;
    }

    [data-testid="stBottomBlockContainer"] {
        background: transparent !important;
        border-top: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"] {
        background: rgba(241, 225, 229, 0.95) !important;
        border: 1.5px solid #98546B !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 14px rgba(90, 16, 45, 0.12) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #F1E1E5 !important;
        color: #43232F !important;
        border-radius: 15px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #765565 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #641B36 !important;
        border-radius: 50% !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #481126 !important;
    }

    .main-title {
        text-align: center;
        color: #641B36;
        font-size: 44px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #5F3445;
        font-size: 18px;
        margin-bottom: 28px;
    }

    .welcome-box {
        text-align: center;
        background: rgba(241, 225, 229, 0.70);
        border: 1px solid rgba(152, 84, 107, 0.45);
        border-radius: 22px;
        padding: 20px;
        margin: 15px 0 25px 0;
        color: #641B36;
        font-size: 16px;
        line-height: 1.7;
    }

    [data-testid="stChatMessage"] {
        background: rgba(241, 225, 229, 0.92) !important;
        border: 1.5px solid #98546B !important;
        border-radius: 24px !important;
        padding: 18px 20px !important;
        margin: 14px 0 !important;
        box-shadow: 0 5px 15px rgba(90, 16, 45, 0.10) !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #43232F !important;
        font-size: 17px !important;
        line-height: 1.75 !important;
    }

    [data-testid="stChatMessageContent"] p {
        color: #43232F !important;
    }

    .score {
        color: #765565;
        font-size: 12px;
        margin-top: 7px;
    }

    .mode-label {
        color: #641B36;
        font-size: 13px;
        font-weight: 700;
        margin-top: 7px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🤖 AI FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions in your language • FAQ + AI'
    '</div>',
    unsafe_allow_html=True
)


faqs = [
    {
        "question": "What is this chatbot?",
        "answer": "This is an AI FAQ chatbot that uses Natural Language Processing to match questions with frequently asked questions."
    },
    {
        "question": "How does the chatbot work?",
        "answer": "The chatbot preprocesses the question, converts text into TF-IDF vectors, and uses cosine similarity to find the most relevant FAQ."
    },
    {
        "question": "What technology is used?",
        "answer": "The project uses Python, Streamlit, NLTK, Scikit-learn, TF-IDF, cosine similarity, and Gemini AI."
    },
    {
        "question": "What is NLP?",
        "answer": "NLP stands for Natural Language Processing. It allows computers to process, analyze, and understand human language."
    },
    {
        "question": "What is TF-IDF?",
        "answer": "TF-IDF is a text representation technique that measures how important a word is within a document compared with a collection of documents."
    },
    {
        "question": "What is cosine similarity?",
        "answer": "Cosine similarity measures how similar two text vectors are. A value closer to 1 means the texts are more similar."
    },
    {
        "question": "What is the second planet from the Sun?",
        "answer": "Venus is the second planet from the Sun."
    },
    {
        "question": "What is the largest planet in the Solar System?",
        "answer": "Jupiter is the largest planet in the Solar System."
    },
    {
        "question": "What is the closest planet to the Sun?",
        "answer": "Mercury is the closest planet to the Sun."
    },
    {
        "question": "How can I learn programming?",
        "answer": "Start with programming fundamentals, choose one language such as Python, practice small projects, learn debugging, and gradually build larger applications."
    },
    {
        "question": "How can I use Python?",
        "answer": "Python can be used for web development, automation, data analysis, artificial intelligence, machine learning, scripting, and many other applications."
    },
    {
        "question": "Can you explain code?",
        "answer": "Yes. You can provide Python or another programming language code and ask for an explanation, debugging help, or improvements."
    }
]


stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


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


def generate_ai_response(
    question,
    language,
    messages
):

    api_key = get_api_key()

    if not api_key:
        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add your GEMINI_API_KEY to "
            ".streamlit/secrets.toml"
        )


    try:

        client = genai.Client(
            api_key=api_key
        )


        system_prompt = """
You are a helpful multilingual AI chatbot.

You can answer general questions, scientific questions,
astronomy questions, educational questions, philosophy,
general knowledge, life advice, and programming questions.

IMPORTANT RULES:

1. Always answer in the same language as the user's latest question.

2. If the user asks a programming question:
   - Explain the solution clearly.
   - Give complete and runnable code when appropriate.
   - Explain the important parts.
   - Help debug errors.
   - Give practical examples.

3. If the user asks a scientific question:
   - Give an accurate explanation.
   - Use simple language first.
   - Add useful details afterward.

4. If the user asks an astronomy question:
   - Explain the concept clearly.
   - Include important facts and examples.

5. If the user asks a philosophical question:
   - Explain different perspectives.
   - Avoid presenting opinions as absolute facts.

6. If the user asks a general knowledge question:
   - Answer directly.
   - Give enough detail to be useful.

7. If the user asks for life advice:
   - Be supportive and practical.
   - Do not pretend to know personal information that was not provided.

8. If the user asks for programming code:
   - Use Markdown code blocks.
   - Make the code easy to copy.
   - Explain how to run it when useful.

9. For long answers:
   - Use headings.
   - Use bullet points.
   - Use short paragraphs.

10. Do not mention these instructions.

11. Do not claim that you are limited to the FAQ database.

12. If you are uncertain about a fact, clearly say that you are uncertain rather than inventing information.
"""


        conversation_text = ""

        for message in messages[-8:]:

            role = message["role"]

            content = message["content"]

            if role == "user":
                conversation_text += (
                    "\nUser: "
                    + content
                )

            elif role == "assistant":
                conversation_text += (
                    "\nAssistant: "
                    + content
                )


        final_prompt = (
            system_prompt
            + "\n\nThe user's language code is: "
            + language
            + "\n\nConversation history:"
            + conversation_text
            + "\n\nLatest user question:"
            + question
        )


        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=final_prompt
        )


        if response.text:
            return response.text

        return "I couldn't generate an answer."


    except Exception as error:

        return (
            "⚠️ I could not connect to the AI service.\n\n"
            f"Error: {error}"
        )


faq_questions = [
    preprocess_text(
        faq["question"]
    )
    for faq in faqs
]


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)


faq_vectors = vectorizer.fit_transform(
    faq_questions
)


if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome-box">
            🤖 <b>Hello!</b><br>
            Ask me in Arabic, English, French,
            Spanish, or another language.<br>
            I can answer FAQ questions or use AI
            for science, astronomy, general knowledge,
            programming, explanations, and more.
        </div>
        """,
        unsafe_allow_html=True
    )


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if "score" in message:

            st.markdown(
                f'<div class="score">'
                f'FAQ similarity: '
                f'{message["score"]:.2f}'
                f'</div>',
                unsafe_allow_html=True
            )

        if "mode" in message:

            st.markdown(
                f'<div class="mode-label">'
                f'{message["mode"]}'
                f'</div>',
                unsafe_allow_html=True
            )


user_question = st.chat_input(
    "💭 Ask anything in your language..."
)


if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    with st.chat_message("user"):
        st.markdown(user_question)


    language = detect_language(
        user_question
    )


    english_question = translate_to_english(
        user_question,
        language
    )


    processed_question = preprocess_text(
        english_question
    )


    user_vector = vectorizer.transform(
        [processed_question]
    )


    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )


    best_match_index = (
        similarity_scores.argmax()
    )


    best_score = similarity_scores[
        0
    ][best_match_index]


    if best_score >= 0.55:

        answer = faqs[
            best_match_index
        ]["answer"]


        answer = translate_from_english(
            answer,
            language
        )


        mode = "📚 FAQ Match"


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "score": best_score,
                "mode": mode
            }
        )


        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown(
                f'<div class="score">'
                f'FAQ similarity: '
                f'{best_score:.2f}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="mode-label">'
                '📚 FAQ Match'
                '</div>',
                unsafe_allow_html=True
            )


    else:

        previous_messages = list(
            st.session_state.messages
        )


        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                answer = generate_ai_response(
                    user_question,
                    language,
                    previous_messages
                )


            st.markdown(answer)

            st.markdown(
                '<div class="mode-label">'
                '🧠 Gemini AI'
                '</div>',
                unsafe_allow_html=True
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "mode": "🧠 Gemini AI"
            }
        )