import nltk
import re
import pandas as pd
from nltk import word_tokenize
from word2number import w2n
from urllib.request import urlopen,HTTPError
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import streamlit as st
import time
from nltk.corpus import wordnet
from inflect import engine
import torch
from torch import nn


    
#Uncomment below two lines
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
#install this if you haven't done
#pip install word2number
#pip install nltk
from user import recommendations, model_fit
from nlp_extract import extract_features
from feature_model import BertClassifier, predict
#reading the data from the csv file
data = pd.read_csv('data/mobiles_scores.csv',low_memory=False)
data.fillna(0, inplace=True)
data.to_csv('data/mobiles_scores.csv', index=False)
brands = pd.read_csv('data/brands.csv',low_memory=False)

from user import recommendations, model_fit
from nlp_extract import extract_features


#mapping the currency to their values
currency_mappings = {
    'thousand': 1e3,
    'thousands': 1e3,
    'k': 1e3,
    'lakh': 1e5,
    'lakhs': 1e5,
    'l': 1e5,
    'million': 1e6,
    'millions': 1e6,
    'm': 1e6,
    'crore': 1e7,
    'crores': 1e7,
    'cr': 1e7,
    'rupee': 1,
    'rupees': 1,
    'rs': 1,
    '₹': 1,
    'inr': 1,
    'dollar': 86,
    'dollars': 86,
    '$': 86,
    'USD': 86,
    'pounds': 118,
    'pound': 118,
    '£': 118,
    'euro': 102,
    'euros': 102,
    '€': 102,
    'eur': 102,
}

rec_model, trainset = model_fit()
user_id = -1


# model_path = 'model/'
# feature_models = []
# for i in range(9):
#     loaded_model_state_dict = torch.load(model_path + str(f"best_model_{i+1}_checkpoint.pth"))
#     feature_model = BertClassifier(num_classes=5)
#     feature_model.load_state_dict(loaded_model_state_dict)
#     feature_models.append(feature_model)
    
model_path = 'model2/'

loaded_model_state_dict = torch.load(model_path + str(f"best_model_checkpoint.pth"))
feature_model = BertClassifier(num_classes=45)
feature_model.load_state_dict(loaded_model_state_dict)
# feature_models.append(feature_model)
    
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#extracting price from the query if mentioned

def con_query(query):
        for currency_word in currency_mappings:
            if currency_word in query:
                 query = re.sub(r"(\d+)" + currency_word, r"\1 " + currency_word + " ", query)
        return query

def convert_to_int(query):
    query = con_query(query)
    word_tokens = nltk.word_tokenize(query)
    total_value = 1
    for word in word_tokens:
        try:
            total_value *= w2n.word_to_num(word)
        except ValueError:     
            if word.lower() in currency_mappings:
                    total_value*=currency_mappings[word.lower()]
    return int(total_value)

#lemmatizing the query
def lemmatize_text(query):
    p = engine()
    lemmatized_words = []
    words = query.split()
    for word in words:
        if p.singular_noun(word):
            lemmatized_words.append(p.singular_noun(word))
        else:
            lemmatized_words.append(word)
    return ' '.join(lemmatized_words)

#getting data of phones not in the csv file from the 91mobiles website
def wrong_name(query):                                                                                        
    matches = process.extract(query, data['model'], scorer=fuzz.token_sort_ratio, limit=1020)
    best_match = max(matches, key=lambda x: x[1])
    best = best_match[0].lower()
    best = best.replace(' ', '-')
    specs_scrapper(best)

def specs_scrapper(phone_name):
    phone_name = phone_name.replace(' ', '-')
    url = f'https://www.91mobiles.com/{phone_name}-price-in-india#specifications'
    try:
        html = urlopen(url)
    except HTTPError as e:
        if e.code == 404:
            wrong_name("specs of " + phone_name)
            return
        else:     
            typewriter(f"HTTP Error {e.code}: {e.reason}")
        return
    
    soup = BeautifulSoup(html, 'html.parser')
        
    spec_titles = soup.find_all('td', attrs={'class': 'spec_ttle'})
    spec_values = soup.find_all('td', attrs={'class': 'spec_des'})
    typewriter("Here are the specifications of the phone you asked for:")
    spec_data = {"Title": [title.get_text(strip=True) for title in spec_titles],
        "Value": [value.get_text(strip=True) for value in spec_values]}
    spec_data = pd.DataFrame(spec_data)
    st.table(spec_data)
    st.session_state.messages.append({"role": "assistant", "content": spec_data.to_markdown(index=False)})

def get_user_id():
    st.write("Are you an existing user or a new user?")
    user_status = st.selectbox("Select:", ["Existing User", "New User"])
    
    if user_status == "Existing User":
        user_id = st.text_input("Enter your user ID:")
        if user_id.strip():  # Check if the user ID is not empty or just whitespace
            st.session_state.user_id = user_id
            print(f"User ID: {user_id}")  # Print the user ID to the terminal
    else:
        st.session_state.user_id = None

#extracting brand or phone name from the query if mentioned
def get_entities(query):
    words = word_tokenize(query)
    for word in words:
        if word in brands.values:
            return word



def typewriter(assistant_response):
    full_response = ""     
    message_placeholder = st.empty()
    for chunk in assistant_response.split(" "):
        full_response += chunk + " "
        time.sleep(0.1)
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

def linewriter(assistant_response):
    full_response = ""     
    message_placeholder = st.empty()
    
    for chunk in assistant_response.split(':'):
        full_response += chunk + " "
        time.sleep(0.1)
        message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)


def tablewriter(assistant_responses):
    full_response = ""
    
    for index, response in assistant_responses.iterrows():
        if index < 4:
            line = f"{index + 1}) {response['model']}: This is the best phone with a price of {response['price']} and an overall user rating of {response['user_rating']}"
            linewriter(line)
        elif index < 8:
            line = f"{index + 1}) {response['model']}: This is a good phone with a price of {response['price']} and an overall user rating of {response['user_rating']}"
            linewriter(line)
        else:
            line = f"{index + 1}) {response['model']}: This is a decent phone with a price of {response['price']} and an overall user rating of {response['user_rating']}"
            linewriter(line)

        full_response += line + "\n"
    full_response += "Please let me know if you want to know more about any of these phones"
    st.session_state.messages.append({"role": "assistant", "content": full_response})

#delay function for the chatbot while replying
def delay():
    linewriter(":AI is thinking 🤔 ...")
    time.sleep(2)

#finding synonyms for the words in the query
def get_synonyms(query):

    synonyms = []
    words = query.split()
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
    return list(set(synonyms))

#check good or best or bad in query if present
def check_best_good_bad(query, df):
    user_req_data = None  # Initialize user_req_data
    
    if 'best' in query:
        # Filtering the data 
        user_req_data = df[df['user_rating'] >= 4.0]
        user_req_data = user_req_data.sample(frac=1) 
        user_req_data = user_req_data.groupby('user_rating').apply(lambda x: x.sample(5)).reset_index(drop=True)

    elif 'bad' in query:
        user_req_data = df[df['user_rating'] < 3]
        if user_req_data.empty:
            typewriter("Sorry, we don't have any phones with such low ratings")
            st.stop()
        user_req_data = user_req_data.sample(frac=1) 
        user_req_data = user_req_data.groupby('user_rating').apply(lambda x: x.sample(10)).reset_index(drop=True)


    else:
        user_req_data = df[(df['user_rating'] < 4.3) & (df['user_rating'] >= 3.0)]
        user_req_data = user_req_data.sample(frac=1)  
        user_req_data = user_req_data.groupby('user_rating').apply(lambda x: x.sample(10)).reset_index(drop=True)

    return user_req_data

def under_query_check(query, df_rec):

    query = query.strip().lower()
    if "one plus" in query:
        query = query.replace("one plus","oneplus")
    if "one pluses" in query:
        query = query.replace("one pluses","oneplus")

    lemmatized_query = lemmatize_text(query)
    price = convert_to_int(query)
    entities = get_entities(lemmatized_query)    
    if ("cheap" in query):
        price = 30000
    if entities:
        if price > 1:
            user_req_data = check_best_good_bad(query,df_rec)
            user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
            user_req_data = user_req_data[user_req_data['price'] <= price] 
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10))  # Show the first 5 results initially
            st.stop()
        else:
            user_req_data = check_best_good_bad(query,df_rec)
            user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10))  # Show the first 5 results initially
            st.stop()
    else:
        if price > 1:
            user_req_data = check_best_good_bad(query,df_rec)
            user_req_data = user_req_data[user_req_data['price'] <= price] 
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10))  # Show the first 5 results initially
            st.stop()

def over_query_check(query, df_rec):
    query = query.strip().lower()
    if "one plus" in query:
        query = query.replace("one plus","oneplus")
    if "one pluses" in query:
        query = query.replace("one pluses","oneplus")

    lemmatized_query = lemmatize_text(query)
    price = convert_to_int(query)
    entities = get_entities(lemmatized_query)
    if"expensive" in query: 
        price = 40000
    if entities:
        if price > 1:
            user_req_data = check_best_good_bad(query, df_rec)
            user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
            user_req_data = user_req_data[user_req_data['price'] >= price] 
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10)) 
            st.stop() # Show the first 5 results initially
        else:
            user_req_data = check_best_good_bad(query,df_rec)
            user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10))  
            st.stop()# Show the first 5 results initially
    else:
        if price > 1:
            user_req_data = check_best_good_bad(query, df_rec)
            user_req_data = user_req_data[user_req_data['price'] >= price] 
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10)) 
            st.stop()
        else:
            user_req_data = check_best_good_bad(query,df_rec)
            user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
            user_req_data.reset_index(drop=True, inplace=True)
            delay()
            tablewriter(user_req_data.head(10))  
            st.stop()
    


def process_query(query):
 
        query = query.strip().lower()
        entities = get_entities(query)
        price = convert_to_int(query)

        # print(price)

        # features = extract_features(query)
        features = predict(query, feature_model)
        print(entities)
        # print(price)

        features = extract_features(query)
        indices = recommendations(rec_model, features, user_id, trainset)
        df_rec = data.iloc[indices]
        
        if 'exit' == query or 'bye' == query or 'goodbye' == query or 'quit' == query or 'stop' == query or 'end' == query:
            typewriter("Thank you for using our service")
            st.stop()

        if "thanks" in query or "thank you" in query:
            typewriter("You are welcome")
            st.stop()

        if 'exit' in query or 'bye' in query or  'goodbye' in query  or  'quit' in query  or  'stop' in query  or 'end' in query  :
            typewriter("Thank you for using our service")
            st.stop()

        if "thanks" in query  or "thank you" in query :
            typewriter("You are welcome")  
            st.stop()

        if "hi" == query  or "hello" == query or "hey" == query or "who are you" == query or "what you can do " == query or "what is your name" == query:
            typewriter("Hi 👋, I am Gadget Guru. I can help you find the best phone for you.")
            st.stop()
        
        if  "how are you" in query  :
            typewriter("I am fine, Thank you")
            st.stop()

        if "under" in query or "less" in query or "below" in query or "within" in query or "cheap" in query:
            under_query_check(query, df_rec)
            st.stop()
        # else:
        #     under_query_check(query, df_rec)

        if "over" in query or "above" in query or "greater" in query or "more than" in query or "expensive" in query:
            over_query_check(query, df_rec)

        # else:
        #     over_query_check(query, df_rec)


        if "specification" in query :
            spec_dataset = pd.read_csv('data/mobiles.csv',low_memory=False)
            phone_name = query.split("specification of ")[-1].strip()
            spec_data = spec_dataset[spec_dataset['model'] .str.lower().str.contains(phone_name.lower())].copy()
            if not spec_data.empty:
                typewriter("Here are the specifications of the phone you asked for:")
                st.table(spec_data)
                st.session_state.messages.append({"role": "assistant", "content": spec_data.to_markdown(index=False)})
            else:
                specs_scrapper(phone_name)
                st.stop()


        elif "specifications" in query :
            spec_dataset = pd.read_csv('data/mobiles.csv',low_memory=False)
            phone_name = query.split("specifications of ")[-1].strip()
            spec_data = spec_dataset[spec_dataset['model'] .str.lower().str.contains(phone_name.lower())].copy()
            if not spec_data.empty:
                typewriter("Here are the specifications of the phone you asked for:")
                st.table(spec_data)
                st.session_state.messages.append({"role": "assistant", "content": spec_data.to_markdown(index=False)})
            else:
                specs_scrapper(phone_name)
                st.stop()

            
        elif "specs" in query:
            phone_name = query.split("specs of ")[-1].strip()
            spec_data = data[data['model'] == phone_name]   
            if not spec_data.empty:
                typewriter("Here are the specifications of the phone you asked for:")
                st.table(spec_data)
                st.session_state.messages.append({"role": "assistant", "content":spec_data.to_markdown(index=False)})
            else:
                specs_scrapper(phone_name) 
                st.stop()


        elif "spec" in query:
            phone_name = query.split("spec of ")[-1].strip()
            spec_data = data[data['model'] == phone_name]   
            if not spec_data.empty:
                typewriter("Here are the specifications of the phone you asked for:")
                st.table(spec_data)
                st.session_state.messages.append({"role": "assistant", "content":spec_data.to_markdown(index=False)})
            else:
                specs_scrapper(phone_name) 
                st.stop()

        elif entities:
            if(len(entities) == len(query)):
                typewriter("You only entered brand name . Please give me more details")
            else:
                if entities:
                    if price > 1:
                        user_req_data = check_best_good_bad(query,df_rec)
                        user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
                        user_req_data = user_req_data[user_req_data['price'] <= price] 
                        user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
                        user_req_data.reset_index(drop=True, inplace=True)
                        delay()
                        tablewriter(user_req_data.head(10))  # Show the first 5 results initially
                        st.stop()
                    else:
                        user_req_data = check_best_good_bad(query,df_rec)
                        user_req_data = user_req_data[user_req_data['model'].str.lower().str.contains(entities.lower())].copy()
                        user_req_data.sort_values(by=['user_rating'], inplace=True, ascending=False)
                        user_req_data.reset_index(drop=True, inplace=True)
                        delay()
                        tablewriter(user_req_data.head(10))  # Show the first 5 results initially
                        st.stop()

        elif "more" in query or "give me more" in query or "give me some more" in query:
            q1 = st.session_state.messages[0]['content']
            print(q1)
            process_query(q1)
            
        else:
            fin_data = check_best_good_bad(query, df_rec)
            delay()
            tablewriter(fin_data.head(10)) 
            st.stop()
            # pass
     
        


#####################################################################################################################################################################

#CHATBOT STARTS HERE

st.title("Welcome to Gadget Guru")
st.subheader('Lets find the phone that fits you the most', divider='rainbow')
st.warning("Information provided by Gadget Gurus may be inaccurate. Please verify the information before making a purchase")

if "user_id" not in st.session_state:
    user_id = get_user_id()




#storing the messages
if "messages" not in st.session_state:      
    st.session_state.messages = []
                       
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Message Gadget Guru"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        process_query(query)
        



    

