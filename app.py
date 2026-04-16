import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Setup aur Libraries
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

negative_words = ['not', 'no', 'nor', 'isn', "isn't", 'aren', "aren't", 'wasn', "wasn't", 
                  'weren', "weren't", 'hasn', "hasn't", 'haven', "haven't", 'hadn', "hadn't", 
                  'doesn', "doesn't", 'don', "don't", 'didn', "didn't", 'won', "won't", 
                  'wouldn', "wouldn't", 'shan', "shan't", 'shouldn', "shouldn't", 
                  'can', "can't", 'couldn', "couldn't", 'mustn', "mustn't", 'never']

for word in negative_words:
    if word in stop_words:
        stop_words.remove(word)

# Cleaning Function
def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\b(not|no|never|isnt|wasnt|arent|didnt|doesnt|dont|cant|couldnt|wouldnt|wont)\s+(\w+)', r'\1_\2', text)
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return ' '.join(cleaned_words)

# ML Models Load karna
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Hybrid Sentiment Predictor Function
def predict_hybrid_sentiment(text):
    text = str(text)
    cleaned_text = clean_tweet(text)
    if not cleaned_text.strip():
        return "Neutral"
        
    vectorized_text = vectorizer.transform([cleaned_text])
    ml_prediction = model.predict(vectorized_text)[0]
    vader_score = sia.polarity_scores(text)['compound']
    has_negation = any(neg in text.lower().split() for neg in negative_words)
    
    if has_negation and vader_score <= -0.05:
        return 'Negative'
    elif ml_prediction == 'Negative':
        return 'Negative'
    else:
        return 'Positive'

# ================= UI DESIGN START =================

st.set_page_config(page_title="Advanced Sentiment Analyzer", page_icon="📈", layout="wide")
st.title("📈 Advanced NLP Sentiment Analysis")
st.write("A Hybrid Model combining Machine Learning and Rule-Based AI")

tab1, tab2 = st.tabs(["✍️ Single Text Analysis", "📂 Bulk CSV Upload & Dashboard"])

# TAB 1: Single Text system
with tab1:
    st.subheader("Test a single review or tweet")
    user_input = st.text_area("Enter your text here:", height=150)

    if st.button("Analyze Single Sentiment", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze!")
        else:
            final_prediction = predict_hybrid_sentiment(user_input)
            if final_prediction == 'Positive':
                st.success("🟢 Result: Positive Sentiment")
            else:
                st.error("🔴 Result: Negative Sentiment")

# TAB 2: Bulk Upload & Visualizations (NAYA UPDATE)
with tab2:
    st.subheader("Analyze hundreds of texts and generate reports")
    st.info("Important: Ensure the text column in your CSV file is named 'Text' (with a capital T).")
    
    uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        if 'Text' not in df.columns:
            st.error("Error: No column named 'Text' found in your file. Please check.")
        else:
            if st.button("Start Bulk Analysis & Generate Dashboard", type="primary"):
                with st.spinner("AI is analyzing data and building your dashboard..."):
                    df['Predicted_Sentiment'] = df['Text'].apply(predict_hybrid_sentiment)
                    
                    st.success("Analysis Complete! Here is your Data Dashboard:")
                    
                    # === DASHBOARD & GRAPHS ===
                    col1, col2 = st.columns(2)
                    
                    sentiment_counts = df['Predicted_Sentiment'].value_counts()
                    
                    with col1:
                        st.write("### 📊 Sentiment Distribution")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        # Colors: Green for Positive, Red for Negative
                        colors = ['#4CAF50' if label == 'Positive' else '#F44336' for label in sentiment_counts.index]
                        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.pyplot(fig)
                        
                    with col2:
                        st.write("### 📈 Quick Stats")
                        st.metric(label="Total Analyzed Texts", value=len(df))
                        st.metric(label="🟢 Total Positive", value=sentiment_counts.get('Positive', 0))
                        st.metric(label="🔴 Total Negative", value=sentiment_counts.get('Negative', 0))
                    
                    st.markdown("---")
                    st.write("### 📋 Data Preview")
                    st.dataframe(df[['Text', 'Predicted_Sentiment']].head(10))
                    
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Full Analyzed Data",
                        data=csv_data,
                        file_name='sentiment_results_with_dashboard.csv',
                        mime='text/csv',
                    )