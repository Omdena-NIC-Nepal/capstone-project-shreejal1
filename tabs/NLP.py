import streamlit as st
import spacy
from textblob import TextBlob
import random

# Load the spaCy model (assumes it's pre-installed via requirements.txt)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model 'en_core_web_sm' is not installed. Please ensure it's listed in requirements.txt.")
    st.stop()

def display():
    # Page Title
    st.subheader("🌿 NLP Analysis on Agri-Climate Texts")
    
    st.markdown("""
        This section allows you to perform natural language processing (NLP) on agriculture or climate-related text.  
        You can analyze **sentiment**, **token structure**, **lemmatization**, and **part-of-speech (POS)** to gain insight into public perception, reports, or news articles related to weather events, crop health, or policy impact.

        Just enter your text and click **Analyze**. 🌾📝
    """)

    # User input text area
    user_text = st.text_area("Enter agriculture- or climate-related text:", 
                            placeholder="Recent droughts are affecting paddy yields in southern Nepal.")

    # Analyze button
    if st.button("Analyze Text"):
        if not user_text.strip():
            st.warning("Please enter valid text for analysis.")
            return
        
        # Process the text with spaCy NLP model
        doc = nlp(user_text)

        # Tokenization, Lemmatization, and POS Tagging
        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        pos_tags = [token.pos_ for token in doc]

        # Sentiment Analysis using TextBlob
        sentiment = TextBlob(user_text).sentiment
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity

        # Sentiment Result Display
        if polarity > 0.1:
            st.success(f"🌞 **Positive Sentiment**\nPolarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
        elif polarity < -0.1:
            st.error(f"🌧️ **Negative Sentiment**\nPolarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")
        else:
            st.warning(f"🌤️ **Neutral Sentiment**\nPolarity: {polarity:.2f}, Subjectivity: {subjectivity:.2f}")

        # Slightly playful transformations (optional)
        transformed_tokens = [token.capitalize() if random.random() > 0.5 else token.upper() for token in tokens]
        transformed_lemmas = [lemma[::-1] for lemma in lemmas]

        # Display Results in Two Columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Transformed Tokens:**", transformed_tokens)
            st.write("**Transformed Lemmas (reversed):**", transformed_lemmas)
        with col2:
            st.write("**POS Tags:**", pos_tags)

        # Divider and additional message
        st.divider()
        st.markdown("✅ NLP Analysis Complete – Use this to monitor sentiment in agriculture reports or climate discussions.")
