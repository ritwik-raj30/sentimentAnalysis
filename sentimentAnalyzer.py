import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score  

# Download the necessary resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize tools
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_excel('/content/modified_conversation_dataset (1).xlsx')
print('THE DATAFRAME')
print(df.head())

# Function to preprocess the conversation text
def preprocess_text(text):
    # Tokenize the conversation into words
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    
    # Lemmatize and remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    
    # Join the cleaned tokens back into a single string
    return ' '.join(lemmatized_tokens)

# Function to get sentiment for the cleaned conversation
def get_sentiment_for_conversation(conversation):
    # Preprocess the text first
    cleaned_conversation = preprocess_text(conversation)
    
    # Compute sentiment score for the cleaned conversation
    sentiment_scores = sia.polarity_scores(cleaned_conversation)
    
    # Determine overall sentiment
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply preprocessing and sentiment analysis on each conversation group

# Apply preprocessing and sentiment analysis on each conversation group
df['Cleaned_Conversation'] = df['Conversation'].apply(preprocess_text)
df['Predicted_Sentiment'] = df['Cleaned_Conversation'].apply(get_sentiment_for_conversation)

# Display the result
print('THE RESULTS ARE :')
print(df[['Conversation', 'Cleaned_Conversation', 'Sentiment']].head())

# Compare predicted sentiment with the original sentiment
df['Comparison'] = df['Sentiment'] == df['Predicted_Sentiment']


group_comparison = df.groupby('Conversation Group')['Comparison'].mean()

# Display comparison and group analysis
print('COMPARISON:')
print(df[['Conversation', 'Sentiment', 'Predicted_Sentiment', 'Comparison']].head())

print('GROUP COMPARISON (ACCURACY BY GROUP):')
print(group_comparison)

# Calculate overall accuracy
accuracy = accuracy_score(df['Sentiment'], df['Predicted_Sentiment'])
print(f'Overall Sentiment Prediction Accuracy: {accuracy:.2f}')
# Save the modified dataset with sentiment analysis
df.to_excel('conversation_with_preprocessing_and_sentiments.xlsx', index=False)



