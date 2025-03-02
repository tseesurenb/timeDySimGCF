import numpy as np
import argparse
import pandas as pd

# seed = 2020
np.random.seed(2020)

def split_by_all_user(df, train_frac=0.8, random_state=None):
    """
    Splits data into training and test sets by selecting 80% of users for training
    and 20% of users for testing.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with at least 'user_id' and 'item_id' columns.
    - train_frac (float): Fraction of users to use for training (default is 0.8).
    - random_state (int): Seed for reproducibility (default is None).

    Returns:
    - df_train (pd.DataFrame): Training dataset with user_id and item_id columns.
    - df_test (pd.DataFrame): Test dataset with user_id and item_id columns.
    """
    # Get unique user IDs
    unique_users = df['user_id'].unique()
    
    # Shuffle and split user IDs into train and test sets
    train_users = pd.Series(unique_users).sample(frac=train_frac, random_state=random_state).values
    test_users = list(set(unique_users) - set(train_users))
    
    # Split DataFrame based on user IDs
    df_train = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    df_test = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    # Keep only 'user_id' and 'item_id'
    df_train = df_train[['user_id', 'item_id']]
    df_test = df_test[['user_id', 'item_id']]

    return df_train, df_test


def split_by_each_user(df, train_frac=0.8, random_state=None):
    """
    Splits data into training and test sets for each user.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with at least 'user_id' and 'item_id' columns.
    - train_frac (float): Fraction of data to use for training (default is 0.8).
    - random_state (int): Seed for reproducibility (default is None).

    Returns:
    - df_train (pd.DataFrame): Training dataset with user_id and item_id columns.
    - df_test (pd.DataFrame): Test dataset with user_id and item_id columns.
    """
    train_data = []
    test_data = []

    # Group by each user and split their interactions
    for user_id, user_data in df.groupby('user_id'):
        # Shuffle the user's interactions
        user_data = user_data.sample(frac=1, random_state=random_state)
        # Determine the split point
        train_size = int(train_frac * len(user_data))
        # Split into train and test sets
        train_data.append(user_data.iloc[:train_size])
        test_data.append(user_data.iloc[train_size:])

    # Combine all user splits into single DataFrames
    df_train = pd.concat(train_data).reset_index(drop=True)
    df_test = pd.concat(test_data).reset_index(drop=True)

    # Keep only 'user_id' and 'item_id'
    df_train = df_train[['user_id', 'item_id']]
    df_test = df_test[['user_id', 'item_id']]

    return df_train, df_test


def create_train_test_data(dataset = "ml-100k", seed=2020):
    
    df_selected = None
    
    if dataset == 'ml-1m':
        # Paths for ML-1M data files
        ratings_path = f'{dataset}/ratings.dat'
        movies_path = f'{dataset}/movies.dat'
        users_path = f'{dataset}/users.dat'

        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='::', names=["user_id", "item_id", "rating", "timestamp"], engine='python', encoding='latin-1')
        
        # # Load the entire movie dataframe into memory
        # item_df = pd.read_csv(movies_path, sep='::', names=["item_id", "title", "genres"], engine='python', encoding='latin-1')
        # item_df = item_df.set_index('item_id')
        
        # # Load the entire user dataframe into memory UserID::Gender::Age::Occupation::Zip-code -> 1::F::1::10::48067 
        # user_df = pd.read_csv(users_path, sep='::', encoding='latin-1', names=["user_id", "sex", "age_group", "occupation", "zip_code"], engine='python')
        # user_df = user_df.set_index('user_id')
        
    elif dataset == 'ml-100k':
        # Paths for ML-100k data files
        ratings_path = f'{dataset}/u.data'
        movies_path = f'{dataset}/u.item'
        users_path = f'{dataset}/u.user'
        
        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='\t', names=["user_id", "item_id", "rating", "timestamp"])

        # # Load the entire movie dataframe into memory
        # genre_columns = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
        # item_df = pd.read_csv(movies_path, sep='|', encoding='latin-1', names=["itemId", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_columns)
        
        # # Create the genres column by concatenating genre names where the value is 1
        # item_df['genres'] = item_df[genre_columns].apply(lambda row: '|'.join([genre for genre, val in row.items() if val == 1]), axis=1)
    
        # # Keep only the necessary columns
        # item_df = item_df[["itemId", "title", "genres"]]
        # item_df = item_df.set_index('itemId')
        
        # # Load the entire user dataframe into memory 1|24|M|technician|85711
        # user_df = pd.read_csv(users_path, sep='|', encoding='latin-1', names=["userId", "age_group", "sex", "occupation", "zip_code"], engine='python')
        # user_df = user_df.set_index('userId')

    if df_selected is not None:
            
        df_train, df_test = split_by_all_user(df_selected, train_frac=0.8, random_state=seed)
        
        # # set headers: user_id item_id
        df_train.columns = ['user_id', 'item_id']
        df_test.columns = ['user_id', 'item_id']
        
        df_train.to_csv(f'{dataset}_2/train_coo.txt', sep=' ', index=False, header=True)
        df_test.to_csv(f'{dataset}_2/test_coo.txt', sep=' ', index=False, header=True)
        
        # Clear memory of large DataFrames that are no longer needed
        del df_selected
        del df_train
        del df_test
        
    else:
        print(f'No data is loaded for dataset: {dataset} !!!')
        
    return None

def main():
    # Load the dataset
    dataset = 'ml-100k'
    create_train_test_data(dataset)
    

if __name__ == "__main__":
    main()
