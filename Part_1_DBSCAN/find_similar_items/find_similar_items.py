import pandas as pd
from random import sample

def find_similar_items():
    def check_id(user_input):
        try:
            item_id = int(user_input)
            assert 1 <= item_id <= 500
            return item_id
        except Exception as e:
            print("There seems to be something wrong with the input. Make sure you are inputting a whole number between 1 and 500.")
            return check_id(input("Try again: "))
    
    checked_id = check_id(input("Please enter a product ID between 1 and 500 to find out about similar items: "))
        
    data_df = pd.read_csv("Data/best_labels_data.csv")
    cluster_id = data_df[data_df.id == checked_id]["labels"].to_list()[0]

    if cluster_id == -1:
        print("Great choice! This is quite the unique item you've got there. There should be something for you among those:")
        # Cluster 7 in best_labels_data.csv is almost exclusively T-shirts, which is a safe bet, on the basis of having looked at other clusters
        cluster_7_ids = [id for id in data_df[data_df.labels == 7]["id"]]
        for id in sample(cluster_7_ids, 5):
            desc_header = data_df[data_df.id == id]["description"].to_list()[0].split("<br>")[0]
            print(f"ID: {id}\nDescription:\n{desc_header}\n\n")
    else:
        print("Nice!\nCheck this out:\n")
        other_cluster_ids = [id for id in data_df[data_df.labels == cluster_id]["id"] if id != checked_id]
        for id in sample(other_cluster_ids, min(5, len(other_cluster_ids))):
            desc_header = data_df[data_df.id == id]["description"].to_list()[0].split("<br>")[0]
            print(f"ID: {id}\nDescription:\n{desc_header}\n\n")



if __name__ == "__main__":
    find_similar_items()