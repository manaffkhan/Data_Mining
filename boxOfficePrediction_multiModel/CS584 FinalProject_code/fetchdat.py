import requests
import json
import pandas as pd
import time

#creating a imdb dataframe
movies_df = pd.read_csv("data/IMDb_movies.csv", skip_blank_lines=True, low_memory=False)

# getting the movie id of all the movies as list
imdb_id_list = movies_df["imdb_title_id"].tolist()

len(imdb_id_list)

# splitting the list into batches of 10k

imdb_id_list_0_10000 = imdb_id_list[:9999] #Done
imdb_id_list_10000_20000 = imdb_id_list[10000:19999]
imdb_id_list_20000_30000 = imdb_id_list[20000:29999]
imdb_id_list_30000_40000 = imdb_id_list[30000:39999]
imdb_id_list_40000_50000 = imdb_id_list[40000:49999]
imdb_id_list_50000_60000 = imdb_id_list[50000:59999]
imdb_id_list_60000_70000 = imdb_id_list[60000:69999]
imdb_id_list_70000_80000 = imdb_id_list[70000:79999]
imdb_id_list_80000_90000 = imdb_id_list[80000:]

# function to fetch the omdb data for an every id in the above lists
def fetch_records(list1):
    URL = "http://www.omdbapi.com/?&apikey=74338615&plot=full&i=" #tt0120689
    data =list()        
#     print("Entering data in file.\n",file_name)
    print("Entering data in file.\n")
    cnt = 0
    with open('movies_json_10k_20k.json', 'w') as f:
        # sending get request and saving the response as response object        
        for i in list1:
            cnt = cnt +1
#             print(cnt)
            if i == "tt0120689" or i == "tt0120690":
                continue
            else: 
                try:
                    print("MovieId:",i,"\t count:",cnt)
                    r = requests.get(url = URL + i)
#                     the response is in the form of json which is converted to a dictionary
                    temp = r.json()
#                     this dictionary is then appended to a list
                    data.append(temp)
                except:
                    print("error encountered")
                    continue
        json.dump(data, f) #finally the list is dumped into a json
        return data
    
    
start_time = time.time()
data_list = fetch_records(imdb_id_list_10000_20000)
print("Execution time:",time.time()-start_time)


