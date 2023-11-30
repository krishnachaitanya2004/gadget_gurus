# Gadget_Guru
Our project is to develop a chatbot prototype designed to assist users in recommending mobile phones based on their preferences and queries.​​ 
## Team Members
|Roll No. | Team Member                  |
|-----|------------------------------|
| 210050008   | Akkapally Shasmith Krishna  |
| 210050057   | Gorle Krishna Chaitanya |
| 210050167 | Vutukuri Vinay Mohan |
| 210050003 | Addanki Sanjeev Varma |

A chat interface that recommends mobile phones based on queries given by the 
Made using the stream-lit interface for this type.

## Instructions
To run this first download our model checkpoint from:
https://drive.google.com/drive/folders/1liLYMqyUZx_8BUpubc1O2sTWOhqOHcxS

or 
train the model using the command
python3 feature_model.py(uncomment all the training, testing lines)

and place the saved pytorch checkpoint in model2 folder(you should create it)
Ensure that streamlit-app.py and user.py, user_prod_ratings.csv are in same directory.
Then run :
pip install -r requirements.txt

Finally, run:
streamlit run streamlit-app.py.

Hope you enjoy our app


First, we use dataprocess.py to generate new processed mobile phone data with the original data.
Then we used generator.py to generate user-phone ratings (user_prod_ratings.csv).
Then we have three main files of the project:

## Code walkthrough
Install all the dependencies given in requirements.txt

-  First, we use dataprocess.py to generate new processed mobile phone data with the original data
-  Then we used generator.py to generate user-phone ratings(user_prod_ratings.csv)
-  Then we have three main files of the project
-  **streamlit-app.py** defines the interface of this project, making the user input their query and receive recommendations. This is done using the Streamlit API library in Python. Here, we process the query given by the user using standard NLP techniques.
- **feature_model.py** trains the BERTClassifier model to extract the required feature vectors from the processed query. This is used as the feature vector for this user to find recommendations. BERTClassifier is a new model which uses base model as BERT-Base and added a linear layer (dimensions - (768 ,5)) with relu activation. It has train, test, predict functions, which does their respective jobs.Loss function used is CrossEntropy. Training has been done for 25 epochs.
-  **user.py** trains a Normalized Matrix Factorization Model for providing recommendations. We use this model to get user-latent vectors and item-latent vectors, then provide recommendations based on the cosine similarities between the user vector and all item-latent vectors. The loss function used is regularized MSE Loss. Training has been done for default values(25 epochs). This also has code for taking the feature-vector generated above and recommending phones based on cosine similarity scores. Both these similarity scores are normalized and weighted average of them is taken as final output of the recommendations.

