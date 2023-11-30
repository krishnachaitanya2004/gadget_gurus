import pandas as pd
import re
import numpy as np


df = pd.read_csv('data/mobiles_lowercase.csv')
mean_value=df['rating'].mean() 
  
# Replace NaNs in column S2 with the 
# mean of values in the same column 
df['rating'].fillna(value=mean_value, inplace=True) 
df['user_rating'] = df['rating']/20
df['user_rating'] = df['user_rating'].apply(lambda x: round(x, 2))
# Price ranges 0-10000: 1; 10000 - 30000 : 2; 30000 - 50000: 3; 50000 - 80000: 4; 80000+ :5
df['price_range'] = df['price'].apply(lambda x: 1 if x <= 10000 else (2 if x <= 30000 else (3 if x <= 50000 else (4 if x <= 80000 else 5))))
# Split the contents in df['sim'] into list of words by splitting by commas and store it in other python list
sim_list = []
for i in df['sim']:
    sim_list.append(i.split(','))
# Create a new column in df called 'sim_count' and store the length of the list in it
df['sim_count'] = [len(i) for i in sim_list]
#Sim count -> 6, then 3; if 5 then 2, less than 5 then 1, if 7 then 4, more than 7 then 5
df['sim_count'] = df['sim_count'].apply(lambda x:3  if x == 6 else (2 if x == 5 else (1 if x < 5 else (4 if x == 7 else 5))))

# Function to extract and convert processor speed to float
def extract_and_convert_processor_speed(sentence):
    # Using regular expression to find the GHz value
    match = re.search(r'(\d+(\.\d+)?)\s*ghz', sentence, re.IGNORECASE)
    if match:
        processor_speed_str = match.group(1)
        try:
            return float(processor_speed_str)
        except ValueError:
            return 0
    else:
        return 0
    
# # Apply the function to create a new column 'processor_speed' as float
# less than 2.2 : 1, 2.2-2.5 : 2, 2.5 - 3 :3, 3 - 3.2: 4, more than 3.2 : 5   
df['processor'].fillna(value = '',inplace = True )
df['processor_speed'] = df['processor'].apply(extract_and_convert_processor_speed)
df['processor_speed'] = df['processor_speed'].apply(lambda x: 1 if x < 2.2 else (2 if x <= 2.5 else (3 if x <= 3.0 else (4 if x <= 3.2 else 5))))


def extract_ram_and_storage(description):
    description = re.sub(r'(\d+)\s*mb', lambda x: str(int(x.group(1)) / 1024) + 'gb', description, flags=re.IGNORECASE)

    ram_match = re.search(r'(\d+)\s*gb\s*ram', description, re.IGNORECASE)
    storage_match = re.search(r'(\d+)\s*gb\s*storage|\b(\d+)\s*gb\s*inbuilt', description, re.IGNORECASE)
    
    ram = float(ram_match.group(1)) if ram_match else 0
    storage = float(storage_match.group(1) or storage_match.group(2)) if storage_match else 0

    return ram, storage
df['ram'].fillna(value = '',inplace = True )
df['ram_size'], df['storage_size'] = zip(*df['ram'].apply(extract_ram_and_storage))
# Apply the function to create new columns 'ram' and 'storage'
# Ram Size -- less than 2: 1, less than 4: 2, less than 6: 3, less than 8: 4, more than 8: 5
df['ram_size'] = df['ram_size'].apply(lambda x: 1 if x <= 2 else (2 if x <= 4 else (3 if x <= 6 else (4 if x <= 8 else 5))))
# Storage_size Size -- les than 32: 1, less than 64: 2, less than 128: 3, less than 256: 4, more than 256: 5
df['storage_size'] = df['storage_size'].apply(lambda x: 1 if x <= 32 else (2 if x <= 64 else (3 if x <= 128 else (4 if x <= 256 else 5))))


def extract_and_add(row):
    capacity_match = re.search(r'(\d+)\s*mah', row)
    charging_match = re.search(r'(\d+)\s*w fast charging', row)

    capacity = float(capacity_match.group(1)) if capacity_match else 0
    charging = float(charging_match.group(1)) if charging_match else 0
    
    return capacity + charging

# Applying the custom function to each row
df['battery'].fillna(value='', inplace=True) 
df['battery_score'] = df['battery'].apply(extract_and_add) 

# Calculate mean excluding NaN values
mean_batt_score = df['battery_score'].mean()
df['battery_score'].replace(0, mean_batt_score, inplace = True) 
# Battery Score -- less than 4050 - 1; less than 5050 2; less than 6550 3; less than 7050 4; more than 7050 5
df['battery_score'] = df['battery_score'].apply(lambda x: 1 if x <= 4050 else (2 if x <= 5050 else (3 if x <= 6550 else (4 if x <= 7050 else 5))))


# if android v12 or iosv14 or ios v13-- 3, android v13 or ios 15 -- 4, android v14 or ios 16/17 -- 5, android v11, android v10, ios v12 -- 2, else 1
android_versions = []

for os in df['os']:
    if "android" in str(os):
        val = os.find('v')
        if val == -1:
            android_versions.append(1)
        else:
            android_version = os[val+1: val+3]
            android_versions.append((android_version))
    else:
        android_versions.append(0)

android_df = pd.DataFrame({'android_version': android_versions})
df['android_version'] = android_df['android_version']
df['android_version'] = df['android_version'].apply(lambda x: 2 if x == '10' or x == '11' else (3 if x == '12' else (4 if x == '13' else (5 if x == '14' else (0 if x == 0 else 1)))))

ios_versions = []
for os in df['os']:
    if "ios" in str(os):
        val = os.find('v')
        if val == -1:
            ios_versions.append(1)
        else:
            ios_version = os[val+1: val+3]
            ios_versions.append((ios_version))
    else:
        ios_versions.append(0)

ios_df = pd.DataFrame({'ios_version': ios_versions})
df['ios_version'] = ios_df['ios_version']
df['ios_version'] = df['ios_version'].apply(lambda x: 2 if x == '12' else (3 if x == '13' or x == '14' else (4 if x == '15' else (5 if x == '16' or x == '17' else (0 if x == 0 else 1)))))

df['os_score'] = df['android_version'].fillna(0) + df['ios_version'].fillna(0)
df['os_score'].to_csv('data/os_score.csv', index=False)



# df['model'] = df['model'].apply(lambda x: x.replace(' 5g', ''))
# df['model'] = df['model'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))


df['camera_score'] = df['camera'].apply(lambda x: str(x)[:str(x).find('mp')].strip())
df['camera_score'].replace('na', '0', inplace=True)
df['camera_score'].replace('no rear camera', '0', inplace=True)
df['camera_score'] = df['camera_score'].str.replace(r'\D', '', regex=True)
df['camera_score'] = pd.to_numeric(df['camera_score'], errors='coerce')
df['camera_score'] = df['camera_score'].apply(lambda x: 1 if x <= 10 else (2 if x > 10 and x < 40 else (3 if x >= 40 and x<= 60 else (4 if x >60 and x < 80 else 5))))

df['cam_score'] = (0.2*df['camera_score']) + 0.25*df['price_range'] + 0.25*df['processor_speed'] + 0.3*df['os_score']
df['cam_score'] = df['cam_score'].apply(lambda x: round(x, 2))


min_value = df['cam_score'].min()
max_value = df['cam_score'].max()
df['cam_score'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df['cam_score'] = ((df['cam_score'] - min_value) / (max_value - min_value))*4 + 1
df['cam_score'] = df['cam_score'].round(0).astype(int)


ppi_score = []
punc_score = []
refresh_rate_score = []
for display in df['display']:
    str1 = display.split(' inches')[0]
    start = display.find(',')
    end_width = display.find('x')
    end_height = display.find('px')
    width = display[start+1:end_width].strip()
    height = display[end_width+1:end_height].strip()
    ppi = int(np.sqrt((int(width) ** 2) + (int(height)**2))/float(str1))
    ppi_score.append(ppi)

    if("punch hole" in display):
        punc_score.append(90)
    elif("notch" in display):
        punc_score.append(90)
    else:
        punc_score.append(50)

    refresh_start = display.find('px,')
    refresh_end = display.find('hz')
    if refresh_start != -1 and refresh_end != -1:
        refresh_rate = int(display[refresh_start + 4:refresh_end].strip())
    else:
        refresh_rate = 60
    refresh_rate_score.append(refresh_rate)

df['refresh_rate_score'] = refresh_rate_score
df['punc_score'] = punc_score
df['ppi_score'] = ppi_score
df['ppi_score'] = df['ppi_score'].apply(lambda x: 1 if x <= 200 else (2 if x <= 300 else (3 if x <= 400 else (4 if x <= 500 else 5))))
df['refresh_rate_score'] = df['refresh_rate_score'].apply(lambda x: 1 if x <= 30 else (2 if x <= 60 else (3 if x <= 90 else (4 if x <= 120 else 5))))
df['punc_score'] = df['punc_score'].apply(lambda x: 1 if x == 50 else (2 if x == 80 else (3)))
df['display_score'] = (0.4*df['ppi_score']) + 0.4*df['punc_score'] + 0.2*df['refresh_rate_score']
df['display_score'] = df['display_score'].apply(lambda x: round(x, 2))

min_value = df['display_score'].min()
max_value = df['display_score'].max()
df['display_score'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df['display_score'] = ((df['display_score'] - min_value) / (max_value - min_value))*4 + 1
df['display_score'] = df['display_score'].round(0).astype(int)
df['display_score'].to_csv('data/display_score.csv', index=False)


android_df.drop(columns=['android_version'], inplace=True)
ios_df.drop(columns=['ios_version'], inplace=True)
df.drop(columns=[ 'rating', 'sim','processor', 'ram', 'battery','camera_score','android_version','ios_version','punc_score','refresh_rate_score','ppi_score'], inplace=True)
df.to_csv('data/mobiles_scores.csv', index=False)



