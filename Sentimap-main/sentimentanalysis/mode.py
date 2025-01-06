
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
import base64
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

def word_count(df):
    # Assuming df is already defined
    all_comments = ' '.join(df['Comment'].astype(str))

    # Tokenize and filter stopwords
    words = word_tokenize(all_comments)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Count the occurrences of each word
    word_counts = Counter(filtered_words)

    # Get the most common words and their counts
    most_common_words = word_counts.most_common(10)

    # Create a small DataFrame with the most common words and counts
    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])

    # Join filtered words for word cloud
    filtered_comments = ' '.join(filtered_words)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_comments)

    # Save the word cloud as an image
    wordcloud_image_path = os.path.join(settings.MEDIA_ROOT, 'wordcloud.png')
    wordcloud.to_file(wordcloud_image_path)
    wordcloud_image_url = os.path.join(settings.MEDIA_URL, 'wordcloud.png')

    return wordcloud_image_url, common_words_df

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
# Preprocess the text data
def preprocess_text(text):
    return preprocess_string(text)

df['Processed_Text'] = df['Comment'].apply(preprocess_text)
dictionary = corpora.Dictionary(df['Processed_Text'])
corpus = [dictionary.doc2bow(text) for text in df['Processed_Text']]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
df['Topic'] = df['Processed_Text'].apply(lambda x: lda_model[dictionary.doc2bow(x)][0][0])
print(df[['Comment', 'Topic']])
from gensim import models
import pandas as pd
num_topics = 5
topic_keywords = []
for i in range(num_topics):
    topic_keywords.append([word for word, _ in lda_model.show_topic(i)])
columns = [f"Keyword {j+1}" for j in range(max(len(words) for words in topic_keywords))]
df_topic_keywords = pd.DataFrame(topic_keywords, columns=columns)
print("Top Keywords for Each Topic:")
print(df_topic_keywords)
from textblob import TextBlob
import pandas as pd
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
keyword = input('Enter your keyword: ')
filtered_comments = df[df['Comment'].str.contains(keyword, case=False)]
filtered_comments['Sentiment'] = filtered_comments['Comment'].apply(get_sentiment)
overall_sentiment = filtered_comments['Sentiment'].mean()


print(f"Overall Sentiment on '{keyword}': {overall_sentiment:.2f}")
from textblob import TextBlob
import pandas as pd
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
def compare_polarity(keyword1, keyword2, df):
    filtered_comments1 = df[df['Comment'].str.contains(keyword1, case=False)]
    filtered_comments2 = df[df['Comment'].str.contains(keyword2, case=False)]
    polarity1 = filtered_comments1['Comment'].apply(get_sentiment).mean()
    polarity2 = filtered_comments2['Comment'].apply(get_sentiment).mean()
    if polarity1 > polarity2:
        result = f"{keyword1} has a more positive public opinion than {keyword2}."
    elif polarity1 < polarity2:
        result = f"{keyword2} has a more positive public opinion than {keyword1}."
    else:
        result = f"The public opinion on {keyword1} and {keyword2} is equally positive."
    
    return result
keyword1 = input("Enter the first keyword: ")
keyword2 = input("Enter the second keyword: ")
result_message = compare_polarity(keyword1, keyword2, df)
print(result_message)
sentiment_distribution = df.groupby(['Topic', 'Sentiment']).size().reset_index(name='Count')
sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
sentiment_distribution['Sentiment'] = sentiment_distribution['Sentiment'].map(sentiment_mapping)
fig = px.bar(sentiment_distribution, x='Topic', y='Count', color='Sentiment',
             color_continuous_scale=px.colors.diverging.RdYlBu,
             labels={'Sentiment': 'Sentiment (Positive: 1, Neutral: 0, Negative: -1)'},
             title='Sentiment Distribution by Topic')

fig.show()