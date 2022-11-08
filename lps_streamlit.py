import streamlit as st
import pandas as pd

with st.sidebar:
  st.header("PoliPad Local Problem Scoping Tool")
  st.write("Input your US state of residence and receive a problem scope based on your state's newsfeed.")
  query = st.text_input('State')
  button = st.button("Load")

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

if button:
  from pygooglenews import GoogleNews

  gn = GoogleNews()

  def get_titles(search):
    stories = []
    search = gn.search(search)
    newsitem = search['entries']
    for item in newsitem:
      story = item.title
      stories.append(story)
    return stories

  def get_links(search):
    links = []
    search = gn.search(search)
    newsitem = search['entries']
    for item in newsitem:
      link = item.link
      links.append(link)
    return links

  first = get_titles(query)
  first_links = get_links(query)
  df = pd.DataFrame(first, columns = ['titles'])
  df['urls'] = first_links

  #newsapi

  from newsapi import NewsApiClient

  newsapi = NewsApiClient(api_key = 'd569604da580417590780d2eb37580a9')

  all_articles = newsapi.get_everything(q=query,
                                        language='en',
                                        sort_by='relevancy')

  articles = all_articles["articles"]

  article_titles = []

  for article in articles:
    titles = article['title']
    article_titles.append(titles)

  article_urls = []

  for article in articles:
    urls = article['url']
    article_urls.append(urls)

  temp_df = pd.DataFrame({'titles': article_titles,
                          'urls': article_urls})

  df = df.append(temp_df, ignore_index=True)
  df['titles'] = df['titles'].str.split('- ').str[0]
  df = df.drop_duplicates()

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  import re 
  import texthero as hero
  from texthero import preprocessing

  #define the pre-processing pipeline
  clean_text_pipeline = [
                preprocessing.remove_urls, #remove urls
                preprocessing.remove_punctuation, #remove punctuation
                preprocessing.remove_digits, #remove numbers
                preprocessing.remove_diacritics, #remove special characters
                preprocessing.lowercase, #convert to lowercase
                preprocessing.remove_stopwords, #remove stopwords
                preprocessing.remove_whitespace , #remove any extra spaces
                preprocessing.stem #stemming of the words
                ]

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  from sklearn.feature_extraction.text import TfidfVectorizer

  #tfdif vectorizer with 1 and 2 ngrams
  tfidf_vec = TfidfVectorizer(ngram_range=(1,2), 
                              min_df=2, 
                              max_features=1000)

  train_temp = pd.read_csv('Mock LPS Train.csv')
  train_temp['clean_text'] = hero.clean(train_temp['titles'], clean_text_pipeline)
  x_train = train_temp.loc[:,'clean_text']
  train_tfidf = tfidf_vec.fit_transform(x_train)

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  article_links = df.urls
  df.drop(['urls'], axis = 1)

  # cleaning text and vectorizing
  df['clean_text'] = hero.clean(df['titles'],clean_text_pipeline)
  df_vect = tfidf_vec.transform(df['clean_text'])

  import pickle

  lps_model = pickle.load(open('trained_model.sav', 'rb'))

  pred_lr = lps_model.predict_proba(df_vect) 

  # function to save results dataset
  labels = ['business', 'crime', 'education', 'ethics', 'government', 'health', 'infrastructure', 'nutrition', 'sanitation', 'unrest']

  def pred_results_dataset(test_id, predictions, labels, filename):
    pred_df = pd.DataFrame(predictions, columns=labels)
    final_sub = pd.concat([test_id, pred_df], axis = 1)
    final_sub.to_csv(filename, index=False)
    print("Submission file created")
  
  # create results dataset for logistic regression
  pred_results_dataset(test_id=df['titles'], predictions = pred_lr.toarray(), labels = labels,
                        filename = "test_results.csv")

  data = pd.read_csv('test_results.csv')
  data['urls'] = article_links

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
  obj = SentimentIntensityAnalyzer()

  data['scores'] = data['titles'].apply(lambda titles: obj.polarity_scores(titles)) #to get the sentiment score, we first need to get the polarity scores of each title
  data['compound']  = data['scores'].apply(lambda score_dict: score_dict['compound']) #from the polarity scores, we get the compound AKA sentiment score
  data.drop(['scores'], axis=1, inplace=True) #we can get rid of the polarity score column

  data1 = data.copy(deep=True) #data1 would only be the titles with negative sentiment #GET RID OF LATER
  data1 = data1[data1.compound < 0] #keep ONLY the titles/rows with a negative sentiment score below 0

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  from keybert import KeyBERT
  custom_kw_extractor = KeyBERT()

  #configuring keyword extraction function
  def keyword_extraction(dataset):
    text = dataset["titles"].str.replace("\xa0","").tolist() #remove any excess white space in the beginning of our titles
    split_text = [] #input our titles from the 'text' dataset and put into list 'split_text'
    for i in text:
      split_text.append(i.rstrip()) #remove any excess white space in the end of our titles
    actual_text = '. '.join(split_text) #we put all of our titles into one line of text, seperated by periods for each title.
    keywords = custom_kw_extractor.extract_keywords(actual_text, diversity=0.9) #apply keyword function on our text
    result = [] 
    for kw in keywords: #get our keywords and put it into 'result list'
      result.append(kw)
    keyword_dataset = pd.DataFrame(result, columns= ['keywords', 'score']) #now we have dataset with keywords column and then their scores in the next column
    keyword_dataset.applymap(str) #make all keywords into string variables
    keyword_dataset = keyword_dataset.nsmallest(5,'score') #get the top 5 keywords with the highest score and order them in descending order
    words = []
    for i in keyword_dataset['keywords']: #get the keywords from our dataset and put it into list 'words'
      words.append(i)
    words = [word.capitalize() for word in words] #capitalize the first letter for every word for better format
    return words

  import textdistance

  #configuring function to find most relevant articles for each category
  def most_relevant(keyword_list, dataset):
    spec = '|'.join(r"\b{}\b".format(x) for x in keyword_list)
    new_titles = dataset[dataset['titles'].str.contains(spec)]
    test = (new_titles.assign(match=new_titles["titles"].map(lambda x: max([textdistance.cosine(x, text) for text in new_titles["titles"]],key=lambda x: x if x != 1 else 0,))).sort_values(by="match").reset_index(drop=True))
    top = test.nsmallest(5,'match')
    final = top['titles'].tolist()
    return final

#////////////////////////////////////////////////////////////////////////////////////////////////////////#

  #final variables is total number of problem articles
  final = data1.index

  #configure function to calculate and display results
  def display_results(dataset, cat, name):
    dataset1 = dataset[dataset[cat] >= 0.5] #get rid of rows with low confidence scores
    if dataset1.empty:
      st.header(name + ": 0%")
    else:
      keywords = keyword_extraction(dataset1) #find keywords
      urls = most_relevant(keywords, dataset1) #find most relevant articles
      percentage = round(((len(dataset1.index)) / len(data1.index)) * 100) #calculate percentage of articles that fall under category
      st.header(name + ": " + str(percentage) + "%")
      if len(dataset1.index) < 5:
        with st.expander('More Info.'):
          st.subheader("Stories")
          for i in urls:
            st.markdown("- " + i)
          st.write("")
        st.write("")
      else:
        with st.expander('More Info.'):
          st.subheader("Keywords:")
          st.write(str(keywords))
          st.subheader("Most Relevant Stories:")
          for i in urls:
            st.markdown("- " + i)
            st.write("")
          st.write("")

  st.title(query + "'s Problem Scope")

  #we use this function for each category
  display_results(dataset = data1[['titles', 'business', 'urls']], cat = 'business', name = "Business, Economy & Fiscal Policy") #business
  display_results(dataset = data1[['titles', 'crime', 'urls']], cat = 'crime', name = "Crime and Law Enforcement") #crime
  display_results(dataset = data1[['titles', 'education', 'urls']], cat = 'education', name = "Education") #education
  display_results(dataset = data1[['titles', 'ethics', 'urls']], cat = 'ethics', name = "Ethics, Morality & Values") #ethics
  display_results(dataset = data1[['titles', 'government', 'urls']], cat = 'government', name = "Government Services & Processes") #government
  display_results(dataset = data1[['titles', 'health', 'urls']], cat = 'health', name = "Healthcare") #healthcare
  display_results(dataset = data1[['titles', 'infrastructure', 'urls']], cat = 'infrastructure', name = "Infrastructure") #infrastructure
  display_results(dataset = data1[['titles', 'nutrition', 'urls']], cat = 'nutrition', name = "Food & Nutrition") #nutrition
  display_results(dataset = data1[['titles', 'sanitation', 'urls']], cat = 'sanitation', name = "Environment & Sanitation") #sanitation
  display_results(dataset = data1[['titles', 'unrest', 'urls']], cat = 'unrest', name = "Civil Disorder & Social Unrest") #unrest
