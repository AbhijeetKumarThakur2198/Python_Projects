# MY PRACTICE PROJECT 4

This Python script utilizes the Natural Language Toolkit (NLTK) library to perform sentiment analysis on text. The SentimentIntensityAnalyzer from NLTK is employed to evaluate the sentiment of the provided text.

## Installation

Ensure you have NLTK installed by running:

```bash
pip install nltk
```

## Usage

1. Import the necessary modules and download the VADER lexicon for sentiment analysis:

    ```python
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    nltk.download('vader_lexicon')
    ```

2. Initialize the SentimentIntensityAnalyzer:

    ```python
    sia = SentimentIntensityAnalyzer()
    ```

3. Use the `analyze_sentiment` function to assess the sentiment of a given text:

    ```python
    def analyze_sentiment(text):
    
        scores = sia.polarity_scores(text)
           
        if scores['compound'] >= 0.05:
            return "Positive"
        elif scores['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    ```

4. Example usage:

    ```python
    text_to_analyze = "Enter your text to analyze"
    result = analyze_sentiment(text_to_analyze)
    print(f"Sentiment: {result}")
    ```

## Full code:
```python
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
```

If you find this project interesting, feel free to use and modify it for integration into more complex programs.
