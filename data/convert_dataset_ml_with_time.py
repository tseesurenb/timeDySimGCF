import numpy as np
import argparse
import pandas as pd

timestamp = 1

def split_by_all_user(df, train_frac=0.8, random_state=None):
    """
    Splits data into training and test sets by selecting 80% of users for training
    and 20% of users for testing.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with at least 'user_id', 'item_id', and 'timestamp' columns.
    - train_frac (float): Fraction of users to use for training (default is 0.8).
    - random_state (int): Seed for reproducibility (default is None).

    Returns:
    - df_train (pd.DataFrame): Training dataset with user_id, item_id, and timestamp columns.
    - df_test (pd.DataFrame): Test dataset with only user_id and item_id columns.
    """
    # Get unique user IDs
    unique_users = df['user_id'].unique()
    
    # Shuffle and split user IDs into train and test sets
    train_users = pd.Series(unique_users).sample(frac=train_frac, random_state=random_state).values
    test_users = list(set(unique_users) - set(train_users))
    
    # Split DataFrame based on user IDs
    df_train = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    df_test = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    # Keep only relevant columns
    if timestamp == 1:
        df_train = df_train[['user_id', 'item_id', 'timestamp']]
    else:
        df_train = df_train[['user_id', 'item_id']]  # Keep timestamp for training
        
    df_test = df_test[['user_id', 'item_id']]  # No timestamp for testing

    return df_train, df_test


def split_by_each_user(df, train_frac=0.8, random_state=None):
    """
    Splits data into training and test sets for each user.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with at least 'user_id', 'item_id', and 'timestamp' columns.
    - train_frac (float): Fraction of data to use for training (default is 0.8).
    - random_state (int): Seed for reproducibility (default is None).

    Returns:
    - df_train (pd.DataFrame): Training dataset with user_id, item_id, and timestamp columns.
    - df_test (pd.DataFrame): Test dataset with only user_id and item_id columns.
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

    # Keep only relevant columns
    if 'timestamp' in df.columns:
        df_train = df_train[['user_id', 'item_id', 'timestamp']]
    else:
        df_train = df_train[['user_id', 'item_id']]

    df_test = df_test[['user_id', 'item_id']]  # No timestamp for testing

    return df_train, df_test


def create_train_test_data(dataset="ml-100k", seed=2020):
    
    df_selected = None
    
    if dataset == 'ml-1m':
        # Paths for ML-1M data files
        ratings_path = f'{dataset}/ratings.dat'
        movies_path = f'{dataset}/movies.dat'
        users_path = f'{dataset}/users.dat'

        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='::', names=["user_id", "item_id", "rating", "timestamp"], engine='python', encoding='latin-1')
        
    elif dataset == 'ml-100k':
        # Paths for ML-100k data files
        ratings_path = f'{dataset}/u.data'
        movies_path = f'{dataset}/u.item'
        users_path = f'{dataset}/u.user'
        
        # Load the entire ratings dataframe into memory
        df_selected = pd.read_csv(ratings_path, sep='\t', names=["user_id", "item_id", "rating", "timestamp"])

    if df_selected is not None:
            
        df_train, df_test = split_by_each_user(df_selected, train_frac=0.8, random_state=seed)

        # Set headers appropriately based on what columns are actually in df_train
        if timestamp == 1 and len(df_train.columns) == 3:
            df_train.columns = ['user_id', 'item_id', 'timestamp']  # Keep timestamp for training
        elif len(df_train.columns) == 2:
            df_train.columns = ['user_id', 'item_id']
        else:
            # If df_train has 3 columns but timestamp is 0, we need to drop the timestamp column
            if len(df_train.columns) == 3 and timestamp == 0:
                df_train = df_train[['user_id', 'item_id']]
                df_train.columns = ['user_id', 'item_id']
            
        df_test.columns = ['user_id', 'item_id']  # No timestamp for testing
        
        # Save to files
        df_train.to_csv(f'{dataset}/train_coo.txt', sep=' ', index=False, header=True)
        df_test.to_csv(f'{dataset}/test_coo.txt', sep=' ', index=False, header=True)
        
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