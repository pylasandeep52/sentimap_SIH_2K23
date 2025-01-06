import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO
from wordcloud import WordCloud
import base64
from .scrape import scrape_youtube_comments
import pandas as pd
import os
from django.conf import settings
import time
def home(request):
    return render(request,"sentianalysis.html")
report1_data={}
def report(request):
    if request.method == 'POST':
        data = request.POST['url']
        #comment this one for testing purpose
        df = scrape_youtube_comments(data, 100)
        #comment this pd.read_csv() for youtube scraping
        #df=pd.read_csv("salaar.csv")
        df.to_csv("new.csv")

        all_comments = ' '.join(df['Comment'].astype(str))

        # Create WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

        image_filename = f'wordcloud_{time.time()}.png'
        image_path = os.path.join(settings.MEDIA_ROOT, image_filename)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Comments')
        plt.savefig(image_path, format='png')
        plt.close()

        # Get the URL path of the saved image
        image_url = os.path.join(settings.MEDIA_URL, image_filename)
        barplot_url,df=barplot(df)
        pyramid_url = generate_sentiment_pyramid(df)
        #common_words_url,common_words=word_count(df)
        count_df=counts(df)
        dft=df.head(20).copy()     
        #positive_url=positive(df) 
        top_url=top(df)
        negative_msg=negative(df)
        context = {
        'comments': dft.to_dict(orient='records'),
        'data': data,
        'graphic': image_url,
        'pyramid': pyramid_url,
        'barplot': barplot_url,
        'commentcount': count_df,
        #'positive_url':positive_url
        'top_url':top_url,
        'negative_msg':negative_msg


       }
        return render(request, 'report.html', context)
import plotly.express as px
import plotly.graph_objects as go 

def generate_sentiment_pyramid(df):
    # Split DataFrame into positive, negative, and neutral
    positive_df = df[df['Sentiment'] == 'Positive']
    neutral_df = df[df['Sentiment'] == 'Neutral']
    negative_df = df[df['Sentiment'] == 'Negative']

    # Count comments for each sentiment
    positive_comments = len(positive_df)
    neutral_comments = len(neutral_df)
    negative_comments = len(negative_df)

    # Create a sentiment pyramid using Plotly
    fig = go.Figure(go.Funnel(
        y=["Positive", "Neutral", "Negative"],
        x=[positive_comments, neutral_comments, negative_comments],
        textposition="inside",
        textinfo="value+percent total",
        marker={"color": ["lightgreen", "lightblue", "lightcoral"]}
    ))

    fig.update_layout(title="Sentiment Distribution")

    # Save the Plotly figure as an image (PNG)
    pyramid_filename = f'sentiment_pyramid.png'
    pyramid_path = os.path.join(settings.MEDIA_ROOT, pyramid_filename)
    fig.write_image(pyramid_path)
    df.to_csv("new.csv")
    pyramid_url = os.path.join(settings.MEDIA_URL, pyramid_filename)
    
    return pyramid_url

    #sentiment analysis matplotlib
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt1
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64
import pandas as pd
from io import BytesIO
def barplot(df):
    df['Comment'] = df['Comment'].fillna('')
    def analyze_sentiment(comment):
        analysis = TextBlob(comment)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    df['Sentiment'] = df['Comment'].apply(analyze_sentiment)
    sentiment_distribution = df['Sentiment'].value_counts()

    # Save the barplot as an image
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)
    plt.title('Sentiment Distribution of Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')

    # Save the barplot as an image file
    barplot_filename = f'barplot.png'  # Use a unique filename
    barplot_path = os.path.join(settings.MEDIA_ROOT, barplot_filename)
    plt.savefig(barplot_path, format='png')
    plt.close()

    # Get the URL path of the saved barplot image
    barplot_url = os.path.join(settings.MEDIA_URL, barplot_filename)
    return barplot_url,df

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
def counts(df):
    all_comments = ' '.join(df['Comment'].astype(str))
    words = word_tokenize(all_comments)
    stop_words = set(STOPWORDS) 
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(10)
    common_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
    return common_words_df
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
import plotly.express as px
from textblob import TextBlob
import pandas as pd
from gensim import models
import collections
import os
from django.conf import settings

def newreport(request):
    if request.method == 'POST':
       keyword1 = request.POST['keyword1']
       keyword2 = request.POST['keyword2']
    df=pd.read_csv("new.csv")
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
    
    
    num_topics = 5
    topic_keywords = []
    for i in range(num_topics):
        topic_keywords.append([word for word, _ in lda_model.show_topic(i)])
    columns = [f"Keyword {j+1}" for j in range(max(len(words) for words in topic_keywords))]
    df_topic_keywords = pd.DataFrame(topic_keywords, columns=columns)
    print("Top Keywords for Each Topic:")
    print(df_topic_keywords)

    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
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
   
    result_message = compare_polarity(keyword1, keyword2, df)
    print(result_message)
    sentiment_distribution = df.groupby(['Topic', 'Sentiment']).size().reset_index(name='Count')
    sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
    sentiment_distribution['Sentiment'] = sentiment_distribution['Sentiment'].map(sentiment_mapping)
    fig = px.bar(sentiment_distribution, x='Topic', y='Count', color='Sentiment',
                color_continuous_scale=px.colors.diverging.RdYlBu,
                labels={'Sentiment': 'Sentiment (Positive: 1, Neutral: 0, Negative: -1)'},
                title='Sentiment Distribution by Topic')

    image_filename = 'sentiment_distribution.png'
    image_path = os.path.join(settings.MEDIA_ROOT, image_filename)
    fig.write_image(image_path)
    newimage= os.path.join(settings.MEDIA_URL, image_filename)
    dft=df.head(20).copy()
    context = {'comments':dft.to_dict(orient='records'),'image_url': newimage,'result_message':result_message}
    return render(request,'newreport.html',context)
'''
import pandas as pd
import plotly.express as px
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
def positive(df):
    positive_words = []

    for index, row in df.iterrows():
        comment = row['Comment']
        sentiment = row['Sentiment']
        
        # Tokenize the words
        words = word_tokenize(comment)
        
        # Filter out stopwords and non-alphabetic words
        filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stopwords.words('english')]
        
        if sentiment == 'Positive':
            positive_words.extend(filtered_words)

    # Get the most common positive words
    positive_word_counts = Counter(positive_words)

    # Convert word counts to DataFrame
    positive_df = pd.DataFrame(positive_word_counts.most_common(5), columns=['Word', 'Count'])

    fig_positive = px.bar(positive_df, x='Word', y='Count', title='Top 5 Positive Words')

    # Save the Plotly figure as an image (PNG or JPG)
    positive_filename = f'positive_words_{time.time()}.png'
    positive_path = os.path.join(settings.MEDIA_ROOT, positive_filename)
    fig_positive.write_image(positive_path)

    # Get the URL path of the saved image
    positive_url = os.path.join(settings.MEDIA_URL, positive_filename)
    return positive_url'''
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import string

def top(df):
    # Tokenize and lowercase the comments
    words = ' '.join(df['Comment'].astype(str)).lower().split()

    # Remove stopwords and non-alphabetic words
    stop_words = set(['the', 'and', 'to', 'of', 'a', 'in', 'is', 'you', 'that', 'it'])  # Add more stopwords if needed
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Select the top 10 words
    top_words = dict(word_counts.most_common(10))

    # Create Plotly bar graph
    fig2 = go.Figure([go.Bar(x=list(top_words.keys()), y=list(top_words.values()), marker_color='purple')])
    fig2.update_layout(title='Top 10 Words in Comments', xaxis_title='Words', yaxis_title='Frequency')

    # Save the Plotly figure as an image (PNG or JPG)
    top_words_filename = f'top_words_.png'
    top_words_path = os.path.join(settings.MEDIA_ROOT, top_words_filename)
    fig2.write_image(top_words_path)

    # Get the URL path of the saved image
    top_words_url = os.path.join(settings.MEDIA_URL, top_words_filename)
    return top_words_url



def negative(df):
    negative_comments_count = df[df['Sentiment'] == 'Negative'].groupby('Author').size().reset_index(name='Negative_Comments_Count')

    # Set a threshold for the number of negative comments to trigger an alert
    threshold = 2  # You can adjust this based on your criteria

    # Identify authors with suspicious behavior
    suspicious_authors = negative_comments_count[negative_comments_count['Negative_Comments_Count'] > threshold]

    # Generate alerts for suspicious authors
    if not suspicious_authors.empty:
        for index, row in suspicious_authors.iterrows():
            username = row['Author']
            negative_count = row['Negative_Comments_Count']
            alert_message = f"The user {username} has suspicious behavior with {negative_count} negative comments. Track this profile for any suspicious activity."
            return alert_message

    