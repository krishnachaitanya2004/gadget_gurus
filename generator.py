# import numpy as np
# import pandas as pd

# # Read mean values from a CSV file
# means_df = pd.read_csv('ratings.csv')

# # Set standard deviation
# std_deviation = 0.5
# size_per_mean = 1000

# def generate_random_integers(mean, n):
#     # Generate random numbers from a uniform distribution
#     random_numbers = np.random.uniform(1, 5, n)
    
#     # Adjust the numbers to achieve the desired mean
#     adjusted_numbers = random_numbers + (mean - np.mean(random_numbers))
    
#     # Clip values to the range [1, 5]
#     clipped_numbers = np.clip(adjusted_numbers, 1, 5)
    
#     # Round to the nearest integer
#     rounded_numbers = np.round(clipped_numbers).astype(int)
    
#     return rounded_numbers


# # Create a new DataFrame with columns for each user
# result_df = pd.DataFrame()
# rec_df = pd.DataFrame()
# # Iterate through mean values and generate random integers for each
# user_columns = []
# n = 1000
# # Iterate through rows in means_df and generate random integers
# for index, row in means_df.iterrows():
#     user_column = generate_random_integers(row['user_rating'], n)
#     # user_columns.append(pd.DataFrame({f"item_{index + 1}": user_column}))
#     result_df = pd.concat([result_df, user_column])

# # result_df = pd.concat(user_columns, axis=0)


# # for index, row in means_df.iterrows():
# #     n =  np.random.randint(0, 3000)
# #     user_column = generate_random_integers(row['user_rating'], n)
# #     for in range(n):
# #     user_columns.append(pd.DataFrame({f"user_{index + 1}": user_column}))
# # Concatenate the list of DataFrames into the final result_df

# # Print the resulting DataFrame
# result_df.to_csv('user_ratings.csv', index=True)

import numpy as np
import pandas as pd

# Read mean values from a CSV file
means_df = pd.read_csv('ratings.csv')

# Set standard deviation
std_deviation = 0.5
size_per_mean = 1000

def generate_random_integers(mean, n):
    # Generate random numbers from a normal distribution with the given mean and standard deviation
    random_numbers = np.random.normal(mean, std_deviation, n)
    
    # Clip values to the range [1, 5]
    clipped_numbers = np.clip(random_numbers, 1, 5)
    
    # Round to the nearest integer
    rounded_numbers = np.round(clipped_numbers).astype(int)
    
    return rounded_numbers

# Create a new DataFrame with rows for each user
result_df = pd.DataFrame()
rec_df = pd.DataFrame()


# Iterate through rows in means_df and generate random integers
# n = 1000
# for index, row in means_df.iterrows():
#     # print(index)
#     user_row = generate_random_integers(row['user_rating'], n)
#     result_df = pd.concat([result_df, pd.DataFrame([user_row])], ignore_index=True)
# print("done1")
# result_df.to_csv('user_ratings.csv', index=True)
user_rows = []
# for index, row in means_df.iterrows():
#     print(index)
#     n = np.random.randint(0, 1000)
#     user_id = np.random.randint(0, 1001, n)
#     user_row = generate_random_integers(row['user_rating'], n)
#     for i in range(n):
#         data = {'user_id': user_id[i], 'index': index, 'user_row': user_row[i]}
#         user_rows.append(pd.DataFrame({'user_id': user_id[i], 'index': index, 'user_row': user_row[i]}))
        
# rec_df = pd.concat(user_rows,axis = 1)
# rec_df.to_csv('user_prod_rating.csv', index=True)

for index, row in means_df.iterrows():
    n = np.random.randint(0, 1000)
    user_id = np.random.randint(0, 1001, n)
    user_row = generate_random_integers(row['user_rating'], n)
    
    data = {'user_id': user_id, 'index': index, 'user_rating': user_row}
    user_rows.append(pd.DataFrame(data))

rec_df = pd.concat(user_rows)
rec_df.to_csv('user_prod_rating.csv', index=True)
