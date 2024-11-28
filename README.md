# 📱 Gadget Guru  
**Your Personal Mobile Recommendation Chatbot**  

Gadget Guru is a chatbot prototype designed to assist users in finding the perfect mobile phone based on their preferences and queries. Built with an interactive **Streamlit** interface, it combines machine learning models and intuitive UI to deliver tailored recommendations.  

---

## 🚀 Features
- **Smart Recommendations**: Personalized suggestions using advanced NLP techniques.  
- **Interactive Interface**: Built with Streamlit for a smooth and engaging user experience.  
- **Custom Models**: Powered by BERT-based classification and matrix factorization for accurate results.  

---

## 🧑‍💻 Team Members  
| Roll No.     | Team Member                  |  
|--------------|------------------------------|  
| 210050008    | Akkapally Shasmith Krishna   |  
| 210050057    | Gorle Krishna Chaitanya      |  
| 210050167    | Vutukuri Vinay Mohan         |  
| 210050003    | Addanki Sanjeev Varma        |  

---

## 🛠️ Setup Instructions  

### Prerequisites
1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
2. To run this first download our model checkpoint from:
https://drive.google.com/drive/folders/1liLYMqyUZx_8BUpubc1O2sTWOhqOHcxS
_or_ train the model using (instructsions below)

### Taining the Models
1️⃣ **Normalized Matrix Factorization (NMF) Model**
- Train the model
`python3 user.py`
2️⃣ **BERT-Based Model**
- Train the feature extractor:
`python3 feature_model.py(uncomment all the training, testing lines)`


## Running the App:
1. Ensure all required files are in the same directory:
<br />
- **streamlit-app.py** <br />
- **user.py** <br />
- **user_prod_ratings.csv** <br />
2. Create a folder named model2 and place the saved PyTorch checkpoint in it.
3 . Install all requirements by running 
`pip install -r requirements.txt`
4. Start the app
`streamlit run streamlit-app.py`

## 📜 Code Walkthrough
### Data Preparation
- **dataprocess.py**: Processes original mobile phone data into a clean dataset.
- **generator.py**: Generates user-phone ratings stored in user_prod_ratings.csv.

### Key Files
1. **streamlit-app.py**:
   - Defines the app interface using the Streamlit API.
   - Processes user queries with NLP techniques and provides recommendations.
2. **feature_model.py**:
   - Trains a BERT-based classifier for feature extraction.
   - **Architecture**:
      - **Base**: BERT-Base.
      - **Added Layer**: Linear (768, 5) with ReLU activation.
   - Functions: Train, Test, Predict.
   - Loss Function: Cross-Entropy.
   - Trained for 25 epochs.
-  **user.py** trains a Normalized Matrix Factorization Model for providing recommendations. We use this model to get user-latent vectors and item-latent vectors, then provide recommendations based on the cosine similarities between the user vector and all item-latent vectors. The loss function used is regularized MSE Loss. Training has been done for default values(25 epochs). This also has code for taking the feature-vector generated above and recommending phones based on cosine similarity scores. Both these similarity scores are normalized and weighted average of them is taken as final output of the recommendations.
