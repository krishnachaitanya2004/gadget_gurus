# 📱 Mobile Gurus 
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
1. Ensure the following files are located in the same directory:

- **`streamlit-app.py`**
- **`user.py`**
- **`user_prod_ratings.csv`**
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
3. **user.py**:
    - This trains train a **Normalized Matrix Factorization (NMF)** model for generating recommendations.
    **Process**:
   - The model learns **user-latent vectors** and **item-latent vectors** by optimizing a **regularized Mean             Squared Error (MSE) Loss** function.
   - Training is conducted for **default settings** (e.g., 25 epochs).
   
   **Recommendation Generation**:
   - **Cosine Similarity** is used to compute:
     - Similarities between the **user vector** and all **item-latent vectors**.
   - Recommendations are generated based on these similarity scores.
   
   **Additional Features**:
   - Includes a method to take a **feature vector** (generated earlier) and recommend **phones** based on their         **cosine similarity scores**.
   - **Normalization** and a **weighted average** of multiple similarity scores are used to produce the final           recommendation output.
  
## Recommendation Workflow
1. Extract feature vectors from user queries using feature_model.py.
2. Compute recommendations using cosine similarity between the user vector and item-latent vectors.
3. Normalize the similarity scores for both models.
4. Calculate a weighted average of the normalized scores to finalize the recommendations.

## Future Work
In future we are thinking to extend our model for variours **gadget recommendations** such as **Laptops**,**Earbuds**,**Washing Machines** etc

## 🎉 Enjoy the App!
Your feedback and contributions are welcome!

💡 Built for everyone who loves finding the perfect Mobile.
