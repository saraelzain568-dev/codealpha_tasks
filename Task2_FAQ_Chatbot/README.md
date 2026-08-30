# Task 2 - AI FAQ Assistant

## Project Overview

AI FAQ Assistant is an intelligent multilingual chatbot built with Python and Streamlit.

The application combines a predefined FAQ knowledge base with Natural Language Processing techniques and Google Gemini AI to provide relevant and intelligent answers to user questions.

## Features

* Interactive chatbot interface
* FAQ question matching
* Natural Language Processing
* TF-IDF text representation
* Cosine similarity
* Multilingual question support
* Automatic language detection
* Translation between languages
* Google Gemini AI integration
* Programming assistance
* AI-generated answers for questions outside the FAQ knowledge base
* Conversation history

## Technologies Used

* Python
* Streamlit
* NLTK
* Scikit-learn
* TF-IDF
* Cosine Similarity
* Google Gemini AI
* Deep Translator
* LangDetect

## How It Works

1. The user enters a question through the chat interface.
2. The application detects the language of the question.
3. Non-English questions are translated into English for processing.
4. The question is cleaned and preprocessed using NLP techniques.
5. The application checks predefined quick answers.
6. The FAQ knowledge base is searched using TF-IDF and cosine similarity.
7. If a strong FAQ match is found, the relevant FAQ answer is returned.
8. If there is no suitable FAQ match, Google Gemini AI generates an answer.
9. The response is returned in the user's language.
10. The conversation is stored and displayed in the chatbot interface.

## FAQ Matching

The chatbot uses TF-IDF to convert FAQ questions into numerical vectors.

Cosine similarity is then used to measure the similarity between the user's question and the available FAQ questions.

This allows the application to identify relevant FAQ answers while avoiding unrelated FAQ responses.

## AI Response System

When a user's question does not have a strong match in the FAQ database, the application uses Google Gemini AI to generate a suitable response.

The AI can answer questions related to:

* General knowledge
* Science
* Astronomy
* Education
* Programming
* Python
* C++
* Artificial Intelligence
* Machine Learning
* Problem solving

## Multilingual Support

The chatbot supports questions in multiple languages.

The application detects the user's language, translates the question when necessary, processes it, and returns the response in the same language whenever possible.

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## API Configuration

The application uses the Google Gemini API for AI-generated responses.

The API key should be stored securely in Streamlit secrets using the following key:

```text
GEMINI_API_KEY
```

The API key should not be uploaded to GitHub.

## Project Structure

```text
Task2_FAQ_Chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
    ├── task2_home.png
    ├── task2_faq.png
    └── task2_ai_multilingual.png
```

## Screenshots

The screenshots folder contains examples of the application interface, FAQ matching, and multilingual AI responses.

## Task Objective

The objective of this task is to develop an AI-powered FAQ chatbot capable of processing user questions, finding relevant FAQ answers using NLP techniques, and generating intelligent responses using AI when necessary.
