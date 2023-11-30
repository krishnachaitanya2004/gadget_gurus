import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from surprise import Dataset, Reader, KNNBasic
from surprise import SVD, NMF
from surprise.model_selection import train_test_split


def model_fit():
    file_path = 'user_prod_rating.csv'
    df = pd.read_csv(file_path)

    reader = Reader(rating_scale=(0, 5))  # Adjust the rating_scale if needed

    # Load the dataset from the pandas DataFrame
    data = Dataset.load_from_df(df[['user_id', 'index', 'user_row']], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.05)

    # Use the KNNBasic collaborative filtering algorithm
    # You can experiment with other algorithms as well
    # algo = KNNBasic(sim_options={'user_based': True})
    algo = NMF()
    # algo = SVD()#use the SVD collaborative filtering algorithm

    algo.fit(trainset)
    
    return algo, trainset
# Replace 'your_user_id' with the actual user ID for whom you want to make recommendations

def recommendations(algo, query, user_id, trainset):
    feature_space = query
    if(user_id==-1):
        df = pd.read_csv('./data/mobiles_scores.csv')
        df1 = df[['price_range', 'sim_count', 'processor_speed', 'ram_size', 'storage_size', 'battery_score', 'os_score', 'cam_score', 'display_score']]
        print(df1.shape)
        cosine_sim = cosine_similarity([feature_space], df1)
        # k = 100  # Replace with your desired value of k
        indices = cosine_sim.argsort()[0][::-1]
        return indices
        
    else:
        
        # Get a list of tuples (product_id, estimated_rating) for the specified user
        user_id_to_recommend = user_id
        user_ratings = [(product_id, algo.predict(user_id_to_recommend, product_id).est) for product_id in trainset.all_items()]

        # Sort the recommendations by the estimated rating in descending order
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # user_ratings = user_ratings[:10]
        # Print the recommended products and their estimated ratings
        # for product_id, estimated_rating in user_ratings[:10]:
        #     print(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

        # print("--------------------------------------------------------------------------")

        df = pd.read_csv('./data/mobiles_scores.csv')
        df1 = df[['price_range', 'sim_count', 'processor_speed', 'ram_size', 'storage_size', 'battery_score', 'os_score', 'cam_score', 'display_score']]
        # print(df1.shape)
        cosine_sim = cosine_similarity([feature_space], df1)
        df1['product_id'] = df1.index
        # k = 5  # Replace with your desired value of k
        indices = cosine_sim.argsort()[0][::-1]
        
        indices_k = indices
        k_nearest_phones = df1.iloc[indices]
        # print(k_nearest_phones)

        # print("--------------------------------------------------------------------------")
        # Normalize ratings
        scaler = MinMaxScaler()
        normalized_ratings = scaler.fit_transform([[rating] for _, rating in user_ratings])

        # Normalize similarity scores
        normalized_similarities = scaler.fit_transform(cosine_sim.T)
        print(normalized_similarities)
        weight_user_ratings = 0.5
        weight_similarity_scores = 0.5

        weighted_sums = {}

        for (product_id, rating), normalized_rating in zip(user_ratings, normalized_ratings):
            weighted_sums[product_id] = weight_user_ratings * normalized_rating[0]

        for i, row in k_nearest_phones.iterrows():
            product_id = row[9]  # Replace with the actual column name
            similarity_score = normalized_similarities[i]
            weighted_sums.setdefault(product_id, 0)
            weighted_sums[product_id] += weight_similarity_scores * similarity_score
        # Sort the products based on the weighted sum
        sorted_products = sorted(weighted_sums.items(), key=lambda x: x[1], reverse=True)
        # top_indices = np.argsort(list(weighted_sums.values()))[::-1]
        # print(top_indices)
        # for product_id, weighted_sum in sorted_products:
        #     print(f"Product ID: {product_id}, Weighted Sum: {weighted_sum}")
        return [prod[0] for prod in sorted_products]
            
# Print the recommended products and their weighted sums
# def main():
#     while True:
#         questionnaire = Questionnaire()
#         questionnaire.ask_questions()
#         user_id, query = questionnaire.results()
#         recommendations(query, user_id)
        
        
        # print(user_id, " and ", query)
        # if(user_id==-1):
        #     feature_space = [2,1,5,3,4,1,3,3,3]
        #     df = pd.read_csv('./data/mobiles_scores.csv')
        #     df1 = df[['price_range', 'sim_count', 'processor_speed', 'ram_size', 'storage_size', 'battery_score', 'os_score', 'cam_score', 'display_score']]
        #     print(df1.shape)
        #     cosine_sim = cosine_similarity([feature_space], df1)
        #     k = 5  # Replace with your desired value of k
        #     indices = cosine_sim.argsort()[0][-k:][::-1]
        #     k_nearest_phones = df1.iloc[indices]
        #     print(k_nearest_phones)

        #     print("--------------------------------------------------------------------------")

        # else:
        #     # product_path = 
        #     file_path = 'user_prod_rating.csv'
        #     df = pd.read_csv(file_path)

        #     reader = Reader(rating_scale=(0, 5))  # Adjust the rating_scale if needed

        #     # Load the dataset from the pandas DataFrame
        #     data = Dataset.load_from_df(df[['user_id', 'index', 'user_row']], reader)

        #     # Split the data into training and testing sets
        #     trainset, testset = train_test_split(data, test_size=0.25)

        #     # Use the KNNBasic collaborative filtering algorithm
        #     # You can experiment with other algorithms as well
        #     # algo = KNNBasic(sim_options={'user_based': True})
        #     algo = SVD()#use the SVD collaborative filtering algorithm

        #     # Train the algorithm on the training set
        #     algo.fit(trainset)
        #     print(algo.pu.shape) #usersxlatentfactors
        #     print(algo.qi.shape) #latentfactorsxitems
        #     # Replace 'your_user_id' with the actual user ID for whom you want to make recommendations
        #     user_id_to_recommend = user_id

        #     # Get a list of tuples (product_id, estimated_rating) for the specified user
        #     user_ratings = [(product_id, algo.predict(user_id_to_recommend, product_id).est) for product_id in trainset.all_items()]

        #     # Sort the recommendations by the estimated rating in descending order
        #     user_ratings.sort(key=lambda x: x[1], reverse=True)
        #     # user_ratings = user_ratings[:10]
        #     # Print the recommended products and their estimated ratings
        #     for product_id, estimated_rating in user_ratings[:10]:
        #         print(f"Product ID: {product_id}, Estimated Rating: {estimated_rating}")

        #     print("--------------------------------------------------------------------------")


        #     feature_space = [2,1,5,3,4,1,3,3,3]
        #     df = pd.read_csv('./data/mobiles_scores.csv')
        #     df1 = df[['price_range', 'sim_count', 'processor_speed', 'ram_size', 'storage_size', 'battery_score', 'os_score', 'cam_score', 'display_score']]
        #     # print(df1.shape)
        #     cosine_sim = cosine_similarity([feature_space], df1)
        #     df1['product_id'] = df1.index
        #     k = 5  # Replace with your desired value of k
        #     indices = cosine_sim.argsort()[0][::-1]
        #     indices_k = indices[:k]
        #     k_nearest_phones = df1.iloc[indices]
        #     print(k_nearest_phones)

        #     print("--------------------------------------------------------------------------")
        #     # Normalize ratings
        #     scaler = MinMaxScaler()
        #     normalized_ratings = scaler.fit_transform([[rating] for _, rating in user_ratings])

        #     # Normalize similarity scores
        #     normalized_similarities = scaler.fit_transform(cosine_sim.T)
        #     print(normalized_similarities)
        #     weight_user_ratings = 0.5
        #     weight_similarity_scores = 0.5

        #     weighted_sums = {}

        #     for (product_id, rating), normalized_rating in zip(user_ratings, normalized_ratings):
        #         weighted_sums[product_id] = weight_user_ratings * normalized_rating[0]

        #     for i, row in k_nearest_phones.iterrows():
        #         product_id = row[9]  # Replace with the actual column name
        #         similarity_score = normalized_similarities[i]
        #         weighted_sums.setdefault(product_id, 0)
        #         weighted_sums[product_id] += weight_similarity_scores * similarity_score
        #     # Sort the products based on the weighted sum
        #     sorted_products = sorted(weighted_sums.items(), key=lambda x: x[1], reverse=True)[:5]

        #     # Print the recommended products and their weighted sums
        #     for product_id, weighted_sum in sorted_products:
        #         print(f"Product ID: {product_id}, Weighted Sum: {weighted_sum}")

# if __name__ == "__main__":
#     main()