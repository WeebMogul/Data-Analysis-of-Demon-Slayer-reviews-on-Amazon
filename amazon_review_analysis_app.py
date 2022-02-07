from numpy import empty
import pandas as pd
import streamlit as st
import matplotlib.pyplot as pypl
from wordcloud import WordCloud
import re
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# preprocess the dataset 
def preprocess_dataframe(df,volume = None):

    # Check if the option selected is 'All volumes'
    if volume != 'All volumes':
        df = df[df['book name'] == volume]
    else:
        df = df

    # remove certain text from the rating column
    df.loc[df.rating != '','rating'] = df.loc[:,'rating'].apply(lambda x : str(x).replace('.0 out of 5 stars','').strip())
    df = df.fillna('')

    # remove any non-latin text from the dataset
    df.review_text = df.review_text.str.replace(r'[^\x00-\x7F]+','',regex=True)
    df.title = df.title.str.replace(r'[^\x00-\x7F]+','',regex=True)

    # add length of reviews to the dataset
    df['length'] = df['review_text'].apply(lambda x : len(str(x)))

    return df

def text_preprocess(text):

    # use the lemmatizer to lemmatize the text
    lemmatizer = WordNetLemmatizer()

    # Make a set of stop words
    stop_words = set(get_stop_words('en'))
    stop_words.update(stopwords.words('english'))

    # contains a dictionary full of contracted words
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                        "can't": "cannot","can't've": "cannot have",
                        "'cause": "because","could've": "could have","couldn't": "could not",
                        "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                        "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                        "hasn't": "has not","haven't": "have not","he'd": "he would",
                        "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                        "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                        "I'd": "I would", "I'd've": "I would have","I'll": "I will",
                        "I'll've": "I will have","I'm": "I am","I've": "I have", "isn't": "is not",
                        "it'd": "it would","it'd've": "it would have","it'll": "it will",
                        "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                        "mayn't": "may not","might've": "might have","mightn't": "might not", 
                        "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not",
                        "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                        "she'll": "she will", "she'll've": "she will have","should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                        "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                        "there'd've": "there would have", "they'd": "they would",
                        "they'd've": "they would have","they'll": "they will",
                        "they'll've": "they will have", "they're": "they are","they've": "they have",
                        "to've": "to have","wasn't": "was not","we'd": "we would",
                        "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                        "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                        "what'll've": "what will have","what're": "what are", "what've": "what have",
                        "when've": "when have","where'd": "where did", "where've": "where have",
                        "who'll": "who will","who'll've": "who will have","who've": "who have",
                        "why've": "why have","will've": "will have","won't": "will not",
                        "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                        "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are",
                        "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                        "you'll": "you will","you'll've": "you will have", "you're": "you are",
                        "you've": "you have"}
    
    # replace any contracted word with it's full form
    text = ' '.join([contractions_dict[word] if word in contractions_dict else word for word in text.split(' ')])
    
    # remove any number or symbols
    text = re.sub('\W| |\d',' ',text)
    
    # lowercase the text
    text = text.lower()
    
    # split each text into an array of words
    text = word_tokenize(text)
    
    # remove any stopwords from the text
    text = [word for word in text if word.lower() not in list(stop_words)]
    
    # lemmatize the words in the review text
    text = [lemmatizer.lemmatize(word,pos='v') for word in text]
    
    # join the words back again
    text = ' '.join(text)
    
    return text

# load amazon review data into cache
@st.cache(persist=True, suppress_st_warning=True,allow_output_mutation=True)
def load_data():
    df = pd.read_json(r'data/Demon_Slayer-reviews_cleaned.json')
    return df

df = load_data()

st.title("Data Analysis of every volume of Demon Slayer on Amazon")

# Drop down box of all the available volumes
volume_names = list(df['book name'].unique())
volume_names.insert(0,'All volumes')

volume = st.selectbox('Choose volume ',volume_names)

# send dataset for preprocessing
if volume != 'All volumes':
    df = preprocess_dataframe(df,volume)
else:
    df = preprocess_dataframe(df,volume)

# Graph containing number of ratings per volume
rating_counts = df[df['rating'] != 'None'].rating.value_counts()

rat_plot = px.bar(rating_counts, y='rating', x=rating_counts.index)
rat_plot.update_layout(title = f"Rating of {volume}")
rat_plot.update_xaxes(title = 'Rating')
rat_plot.update_yaxes(title = 'Count')
st.write(rat_plot)

# Drop down box used to analyze the review title or the review text
text_category = st.selectbox('Which category to analyze ?',['review_text','title'])

# Options to modify then number of words for wordcloud and the rating 
num_of_wordcloud_words = st.slider("Number of words to display in the word cloud",100,500,300)
rating_number = st.select_slider("Rating",[1,2,3,4,5],5)

text_to_analyze = df.loc[df['rating'] == str(rating_number),text_category]

# Wordcloud showing the most common words
def wordcloud(text,max_words):

    if text.empty:
        st.info("There are no reviews top plot wordcloud under this rating for this volume")
    else:
        
        # function for displaying wordcloud
        wc = WordCloud(width=400, height=300, max_words=max_words, colormap="Dark2",scale=4, max_font_size=50).generate(text.str.cat(sep="\n"))
        
        # Displaying the wordcloud on streamlit
        fig = pypl.figure(figsize=(20,10))
        pypl.imshow(wc, interpolation='bilinear')
        pypl.axis("off")
        st.pyplot(fig)
    
wordcloud(text_to_analyze,num_of_wordcloud_words)

# Most Frequent Words in review text or slider

num_of_words = st.slider("Number of most frequent words",5,20)

# filter text by rating and preprocess the text data
text_cleaned = text_to_analyze.apply(lambda x : text_preprocess(x))

if text_cleaned.empty:
    st.info("There is no data to plot the most frequent words under this rating for this volume")
else:

    # plot the most frequent words in the text

    most_freq_words = pd.DataFrame(text_cleaned.str.split(expand=True).stack().value_counts(),columns=['Count'])[:num_of_words]   
    review_plot = px.bar(most_freq_words.sort_values(by='Count'), x='Count', y=most_freq_words.index, orientation='h')
    review_plot.update_layout(title = f'Most Frequent Words')
    review_plot.update_xaxes(title = 'Count')
    review_plot.update_yaxes(title = 'Words')
    st.write(review_plot)

# Most relevant topics

def get_topics(n_gram_range,no_of_topics,df,text_type):

    # use countvectorizer to get the number of words per review text
    vec = CountVectorizer(ngram_range=n_gram_range)

    # preprocess the text
    df[text_type] = df[text_type].apply(lambda x : text_preprocess(x))

    # fit the data into the countvectorizer
    x = vec.fit_transform(df[text_type].values)
    
    # use LDA method to find the most relevant topics
    lda10 = LatentDirichletAllocation(n_components=no_of_topics, random_state=42)
    lda10.fit_transform(x)
    
    topic_words = []
    topic_no = []
    top_n = 10
    topic_collection = []

    # get the topics and each word per topic
    for idx, topic in enumerate(lda10.components_):
         
         for i in topic.argsort()[:-top_n-1:-1]:
                topic_words.append((vec.get_feature_names()[i]))
         
         topic_collection.append(topic_words[:])
         topic_no.append(f'Topic {idx+1}')
         topic_words.clear()
    
    return topic_no, topic_collection

if text_to_analyze.empty:
        st.info("There are no topics to extract under this rating for this volume")
else:

    # get the number of topics and the words per topic
    ngram_topic = st.slider("Select how many words for each topic",1,3)
    how_many_topics = st.slider("How many topics to view ?",1,20,8)

    # return the topics and their content
    topic_no, topic_collection = get_topics((ngram_topic,ngram_topic),how_many_topics,df.loc[df['rating'] == str(rating_number)],text_category)

    # create a dataframe based of said topics
    topic_df = pd.DataFrame(dict(zip(topic_no,topic_collection)))

    # filter the topic dataframe and select which topic to use
    selected_topic = st.multiselect('Select topic:',topic_df.columns,topic_df.columns[0])
    st.table(topic_df[selected_topic])



