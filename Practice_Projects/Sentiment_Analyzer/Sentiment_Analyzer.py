# IMPORT THE NLTK LIBRARY AND SENTIMENT ANALYSIS MODULE
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# DOWNLOAD THE VADER LEXICON FOR SENTIMENT ANALYSIS
nltk.download('vader_lexicon')

# INITIALIZE THE SENTIMENTINTENSITYANALYZER
sia = SentimentIntensityAnalyzer()

# DEFINE A FUNCTION TO ANALYZE SENTIMENT OF GIVEN TEXT
def analyze_sentiment(text):
    # GET THE POLARITY SCORES FOR THE GIVEN TEXT
    scores = sia.polarity_scores(text)

    # INTERPRET THE COMPOUND SCORE AS SENTIMENT
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# EXAMPLE USAGE
text_to_analyze = "I love using NLTK for sentiment analysis!"
result = analyze_sentiment(text_to_analyze)
print(f"Sentiment: {result}")
