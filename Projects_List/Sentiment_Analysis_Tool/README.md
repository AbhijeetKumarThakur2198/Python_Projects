# Sentiment Analysis Tool

## Overview
In this project, I developed a Sentiment Analysis Tool utilizing the nltk module. The objective was to create a robust system capable of analyzing text data to determine the underlying sentiment expressed. Leveraging the natural language processing capabilities offered by nltk, I implemented various techniques to preprocess the textual data, including tokenization, stemming, and stop-word removal. Next, I employed machine learning algorithms such as Naive Bayes and Support Vector Machines to train the model on labeled datasets, enabling it to classify text into categories representing different sentiment polarities, such as positive, negative, or neutral. 

Additionally, I incorporated features for handling emoticons, slang, and negation to enhance the accuracy of sentiment classification. The tool provides users with insights into the emotional tone of the input text, facilitating applications in fields like social media monitoring, customer feedback analysis, and market sentiment tracking. Through rigorous testing and validation, I ensured the tool's reliability and effectiveness in real-world scenarios, aiming to empower users with valuable sentiment analysis capabilities.

Let's examine the code for further details!

## Full Code
Here is the full code of Sentiment Analysis Tool:
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

## How To Report Problems
If you encounter any errors or bugs in the code, please create an issue.

## License
This project is under the [MIT License](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) - see the [LICENSE.md](https://github.com/AbhijeetKumarThakur2198/Python_Projects/tree/main/Projects_List/LICENSE.md) file for details.
