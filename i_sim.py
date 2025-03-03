'''
Created on Sep 1, 2024
Pytorch Implementation of DySimGCF:  A Similarity-Centric Graph Convolutional Network for Adaptive Collaborative Filtering
'''

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse import csr_matrix
from world import config

# ANSI escape codes for bold and red
br = "\033[1;31m"
b = "\033[1m"
bg = "\033[1;32m"
bb = "\033[1;34m"
rs = "\033[0m"

def jaccard_sim(matrix, top_k=20, self_loop=False, verbose=-1):
    if verbose > 0:
        print('Computing Jaccard similarity by top-k...')
    
    # Ensure the matrix is binary and of type int
    binary_matrix = csr_matrix((matrix > 0).astype(int))

    # Compute the intersection using dot product
    intersection = binary_matrix.dot(binary_matrix.T).toarray()  # Convert to dense format to avoid dtype issues

    # Compute the row sums (number of interactions)
    row_sums = np.array(binary_matrix.sum(axis=1)).flatten()
    
    # Compute the union
    union = row_sums[:, None] + row_sums[None, :] - intersection
    
    # Ensure intersection and union are of type float to avoid dtype issues
    intersection = intersection.astype(np.float32)
    union = union.astype(np.float32)

    # Compute Jaccard similarity
    similarity_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=union != 0)

    #full_similarity_matrix = similarity_matrix.copy()  # Keep the full similarity matrix
    
    # If self_loop is False, set the diagonal to zero
    if self_loop:
        np.fill_diagonal(similarity_matrix, 1)
    else:
        np.fill_diagonal(similarity_matrix, 0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing {br}jaccard{rs} similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix[i]
        if np.count_nonzero(row) == 0:
            continue
        
        # Sort indices based on similarity values (in descending order) and select top K
        top_k_idx = np.argsort(-row)[:top_k]
        
        # Store the top K similarities
        filtered_data.extend(row[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(top_k_idx)

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del binary_matrix, intersection, row_sums, union, similarity_matrix, filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr() #, csr_matrix(full_similarity_matrix)

def cosine_sim_old(matrix, top_k=20, self_loop=False, verbose=-1):
    
    if verbose > 0:
        print('Computing cosine similarity by top-k...')
    
    # Convert the sparse matrix to a binary sparse matrix
    sparse_matrix = csr_matrix(matrix)
    sparse_matrix.data = (sparse_matrix.data > 0).astype(int)

    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # If self_sim is False, set the diagonal to zero
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing {br} cosine {rs} similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col

        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del sparse_matrix, similarity_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr()


def pearson_sim(matrix, top_k=20, threshold = 0.0, self_loop=False, verbose=-1):
       
    if verbose > 0:
        print('Computing Pearson similarity by top-k...')
    
    # Convert the input matrix to a sparse format (CSR)
    sparse_matrix = csr_matrix(matrix)
    
    # Row-wise mean centering: subtract the mean from non-zero entries
    row_means = np.array(sparse_matrix.mean(axis=1)).flatten()
    sparse_matrix.data -= row_means[sparse_matrix.nonzero()[0]]
    
    if verbose > 0:
        print('Data mean-centered for Pearson similarity.')

    # Compute cosine similarity on the mean-centered data
    similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
    
    if verbose > 0:
        print('Pearson similarity computed.')
    
    # If self_loop is True, set the diagonal to 1; otherwise, set it to 0
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    #full_similarity_matrix = similarity_matrix.copy()  # Keep the full similarity matrix
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f'Preparing {br}pearson{rs} similarity matrix | Top-K: {top_k}')
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Apply the threshold filter
        valid_idx = row_data > threshold
        row_data = row_data[valid_idx]
        row_indices = row_indices[valid_idx]

        # Sort indices based on similarity values (in descending order) and select top K
        if row_data.size > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])

    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del sparse_matrix, similarity_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr() #, full_similarity_matrix


def cosine_sim_exp(matrix, top_k=20, self_loop=False, verbose=-1):
    if verbose > 0:
        print('Computing time-weighted cosine similarity by top-k...')
    
    # Convert the original matrix to a sparse matrix (preserving timestamps)
    sparse_matrix = csr_matrix(matrix)
    
    # Create a user-wise normalized version of the timestamps using softmax
    # First, we'll need to process each user's interactions separately
    normalized_matrix = sparse_matrix.copy()
    
    if verbose > 0:
        print('Normalizing timestamps with softmax per user...')
    
    # Process each user (row) to normalize their timestamps
    for i in range(sparse_matrix.shape[0]):
        # Get the row for user i
        user_row = sparse_matrix.getrow(i)
        
        if user_row.nnz == 0:  # Skip if user has no interactions
            continue
        
        # Extract indices and timestamps for this user
        user_data = user_row.data
        user_indices = user_row.indices
        
        # Calculate softmax for this user's timestamps
        # First shift values to prevent numerical overflow
        shifted_data = user_data - np.max(user_data)
        exp_data = np.exp(shifted_data)
        softmax_data = exp_data / np.sum(exp_data)
        
        # Update the normalized matrix with softmax values
        normalized_matrix.data[normalized_matrix.indptr[i]:normalized_matrix.indptr[i+1]] = softmax_data
    
    if verbose > 0:
        print('Computing cosine similarity with time weights...')
    
    # Compute sparse cosine similarity (output will be sparse)
    similarity_matrix = cosine_similarity(normalized_matrix, dense_output=False)
    
    if verbose > 0:
        print('Cosine similarity computed.')
    
    # If self_sim is False, set the diagonal to zero
    if self_loop:
        similarity_matrix.setdiag(1)
    else:
        similarity_matrix.setdiag(0)
    
    # Prepare to filter top K values
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(similarity_matrix.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing {'br'} cosine {'rs'} similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = similarity_matrix.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])
    
    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=similarity_matrix.shape)
    
    del sparse_matrix, similarity_matrix, normalized_matrix
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr()

def cosine_sim_2(matrix, binary_matrix=None, alpha=0.5, top_k=20, self_loop=False, verbose=-1):
    """
    Computes a weighted combination of binary and temporal similarity matrices.
    
    Parameters:
    -----------
    matrix : scipy.sparse.csr_matrix or numpy.ndarray
        Matrix with timestamps as values, where rows are users and columns are items.
    binary_matrix : scipy.sparse.csr_matrix or numpy.ndarray, optional
        Binary interaction matrix. If None, a binary version will be created from the timestamp matrix.
    alpha : float, default=0.5
        Weight parameter for combining similarities. 
        final_sim = alpha * binary_sim + (1-alpha) * temporal_sim
    top_k : int, default=20
        Number of most similar items to keep for each item.
    self_loop : bool, default=False
        Whether to include self-similarity.
    verbose : int, default=-1
        Verbosity level. Use 0 or higher for more verbose output.
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Combined similarity matrix with only top_k values per row.
    """
    
    if verbose > 0:
        print('Computing weighted combination of binary and temporal similarities...')
    
    # Convert the original matrix to a sparse matrix (preserving timestamps)
    sparse_matrix = csr_matrix(matrix)
    
    # Create binary matrix if not provided
    if binary_matrix is None:
        if verbose > 0:
            print('Creating binary matrix from timestamp matrix...')
        # Create a copy of the matrix with binary values (1 where timestamp exists)
        binary_matrix = sparse_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
    else:
        binary_matrix = csr_matrix(binary_matrix)
    
    # --- Step 1: Compute binary similarity ---
    if verbose > 0:
        print('Computing binary cosine similarity...')
    binary_sim = cosine_similarity(binary_matrix, dense_output=False)
    
    # --- Step 2: Compute temporal similarity ---
    # Create a user-wise normalized version of the timestamps using softmax
    normalized_matrix = sparse_matrix.copy()
    
    if verbose > 0:
        print('Normalizing timestamps with softmax per user...')
    
    # Process each user (row) to normalize their timestamps
    for i in range(sparse_matrix.shape[0]):
        # Get the row for user i
        user_row = sparse_matrix.getrow(i)
        
        if user_row.nnz == 0:  # Skip if user has no interactions
            continue
        
        # Extract indices and timestamps for this user
        user_data = user_row.data
        
        # Calculate softmax for this user's timestamps
        # First shift values to prevent numerical overflow
        shifted_data = user_data - np.max(user_data)
        exp_data = np.exp(shifted_data)
        softmax_data = exp_data / np.sum(exp_data)
        
        # Update the normalized matrix with softmax values
        normalized_matrix.data[normalized_matrix.indptr[i]:normalized_matrix.indptr[i+1]] = softmax_data
    
    if verbose > 0:
        print('Computing temporal cosine similarity...')
    
    # Compute sparse cosine similarity with time weights
    temporal_sim = cosine_similarity(normalized_matrix, dense_output=False)
    
    alpha = config["alpha"]
     # --- Step 3: Combine the two similarity matrices ---
    if verbose > 0:
        print(f'Combining similarities with alpha={alpha}...')
    
    # We'll combine them directly using their sparse representations
    # alpha * binary_sim + (1-alpha) * temporal_sim
    combined_sim = binary_sim.multiply(alpha) + temporal_sim.multiply(1-alpha)
    
    # If self_sim is False, set the diagonal to zero
    if self_loop:
        combined_sim.setdiag(1)
    else:
        combined_sim.setdiag(0)
    
    # --- Step 4: Filter to top K values ---
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(combined_sim.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing combined similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = combined_sim.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])
    
    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=combined_sim.shape)
    
    # Clean up to free memory
    del sparse_matrix, binary_matrix, normalized_matrix
    del binary_sim, temporal_sim, combined_sim
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr()


def cosine_sim_3(matrix, binary_matrix=None, alpha=0.5, decay_rate=0.01, top_k=20, self_loop=False, verbose=-1):
    """
    Computes a weighted combination of binary and temporal similarity matrices with time decay.
    
    Parameters:
    -----------
    matrix : scipy.sparse.csr_matrix or numpy.ndarray
        Matrix with timestamps as values, where rows are users and columns are items.
    binary_matrix : scipy.sparse.csr_matrix or numpy.ndarray, optional
        Binary interaction matrix. If None, a binary version will be created from the timestamp matrix.
    alpha : float, default=0.5
        Weight parameter for combining similarities. 
        final_sim = alpha * binary_sim + (1-alpha) * temporal_sim
    decay_rate : float, default=0.1
        Controls how quickly the importance of older interactions decays.
        Higher values mean faster decay.
    top_k : int, default=20
        Number of most similar items to keep for each item.
    self_loop : bool, default=False
        Whether to include self-similarity.
    verbose : int, default=-1
        Verbosity level. Use 0 or higher for more verbose output.
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Combined similarity matrix with only top_k values per row.
    """
    
    if verbose > 0:
        print('Computing weighted combination of binary and temporal similarities...')
    
    decay_rate = config["t_decay"]
    
    # Convert the original matrix to a sparse matrix (preserving timestamps)
    sparse_matrix = csr_matrix(matrix)
    
    # Create binary matrix if not provided
    if binary_matrix is None:
        if verbose > 0:
            print('Creating binary matrix from timestamp matrix...')
        # Create a copy of the matrix with binary values (1 where timestamp exists)
        binary_matrix = sparse_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
    else:
        binary_matrix = csr_matrix(binary_matrix)
    
    # --- Step 1: Compute binary similarity ---
    if verbose > 0:
        print('Computing binary cosine similarity...')
    binary_sim = cosine_similarity(binary_matrix, dense_output=False)
    
    # --- Step 2: Compute temporal similarity with time decay ---
    # Create a user-wise normalized version of the timestamps using time decay
    normalized_matrix = sparse_matrix.copy()
    
    if verbose > 0:
        print('Applying time decay to timestamps...')
    
    # Find the maximum timestamp in the dataset to use as reference
    max_timestamp = np.max(sparse_matrix.data)
    
    # Apply time decay: weight = exp(-decay_rate * (max_time - interaction_time))
    # This gives higher weights to more recent interactions
    time_diff = max_timestamp - sparse_matrix.data
    decay_weights = np.exp(-decay_rate * time_diff)
    
    # Update the normalized matrix with decay weights
    normalized_matrix.data = decay_weights
    
    if verbose > 0:
        print('Computing temporal cosine similarity...')
    
    # Compute sparse cosine similarity with time weights
    temporal_sim = cosine_similarity(normalized_matrix, dense_output=False)
    
    # --- Step 3: Combine the two similarity matrices ---
    # Use alpha from parameters, not from config
    if verbose > 0:
        print(f'Combining similarities with alpha={alpha}...')
    
    # We'll combine them directly using their sparse representations
    # alpha * binary_sim + (1-alpha) * temporal_sim
    alpha = config["alpha"]
    combined_sim = binary_sim.multiply(alpha) + temporal_sim.multiply(1-alpha)
    
    # If self_loop is False, set the diagonal to zero
    if self_loop:
        combined_sim.setdiag(1)
    else:
        combined_sim.setdiag(0)
    
    # --- Step 4: Filter to top K values ---
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    from tqdm import tqdm
    pbar = tqdm(range(combined_sim.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing combined similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = combined_sim.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])
    
    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=combined_sim.shape)
    
    # Clean up to free memory
    del sparse_matrix, binary_matrix, normalized_matrix
    del binary_sim, temporal_sim, combined_sim
    del filtered_data, filtered_rows, filtered_cols
    
    return filtered_similarity_matrix.tocsr()


def cosine_sim(matrix, binary_matrix=None, alpha=0.5, decay_mode='both', decay_rate=0.01, beta=1.0, top_k=20, self_loop=False, verbose=-1):
    """
    Computes a weighted combination of binary and temporal similarity matrices with time decay.
    
    Parameters:
    -----------
    matrix : scipy.sparse.csr_matrix or numpy.ndarray
        Matrix with timestamps as values, where rows are users and columns are items.
    binary_matrix : scipy.sparse.csr_matrix or numpy.ndarray, optional
        Binary interaction matrix. If None, a binary version will be created from the timestamp matrix.
    alpha : float, default=0.5
        Weight parameter for combining similarities. 
        final_sim = alpha * binary_sim + (1-alpha) * temporal_sim
    decay_mode : str, default='global'
        Mode for time decay calculation:
        - 'global': Uses global reference (absolute decay from earliest timestamp)
        - 'user': Uses user-specific reference (relative decay from each user's first interaction)
        - 'both': Combines both global and user-specific decay
    decay_rate : float, default=0.1
        Controls how quickly the importance of older interactions decays.
        Higher values mean faster decay.
    beta : float, default=1.0
        Scaling factor for time distance when using 'both' decay_mode.
        Controls the balance between global and user-specific decay.
    top_k : int, default=20
        Number of most similar items to keep for each item.
    self_loop : bool, default=False
        Whether to include self-similarity.
    verbose : int, default=-1
        Verbosity level. Use 0 or higher for more verbose output.
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Combined similarity matrix with only top_k values per row.
    """
    # import numpy as np
    # from scipy.sparse import csr_matrix, coo_matrix
    # from sklearn.metrics.pairwise import cosine_similarity
    # from tqdm import tqdm
    
    if verbose > 0:
        print('Computing weighted combination of binary and temporal similarities...')
    
    # Convert the original matrix to a sparse matrix (preserving timestamps)
    sparse_matrix = csr_matrix(matrix)
    
    # Create binary matrix if not provided
    if binary_matrix is None:
        if verbose > 0:
            print('Creating binary matrix from timestamp matrix...')
        # Create a copy of the matrix with binary values (1 where timestamp exists)
        binary_matrix = sparse_matrix.copy()
        binary_matrix.data = np.ones_like(binary_matrix.data)
    else:
        binary_matrix = csr_matrix(binary_matrix)
    
    # --- Step 1: Compute binary similarity ---
    if verbose > 0:
        print('Computing binary cosine similarity...')
    binary_sim = cosine_similarity(binary_matrix, dense_output=False)
    
    # --- Step 2: Compute temporal similarity with time decay ---
    # Create a decay-weighted version of the timestamps
    time_weighted_matrix = sparse_matrix.copy()
    
    # Time window unit (24 hours in seconds) for scaling
    _win_unit = 24 * 3600
    
    if decay_mode == 'global' or decay_mode == 'both':
        if verbose > 0:
            print('Applying global (absolute) time decay...')
        
        # Global minimum timestamp (earliest interaction)
        _start = np.min(sparse_matrix.data)
        
        # Calculate absolute decay: (timestamp - earliest) / day_in_seconds
        abs_decay = (sparse_matrix.data - _start) / _win_unit
        
        # Apply exponential decay to absolute time differences
        global_weights = np.exp(decay_rate * abs_decay)
        
        if decay_mode == 'global':
            time_weighted_matrix.data = global_weights
    
    if decay_mode == 'user' or decay_mode == 'both':
        if verbose > 0:
            print('Applying user-specific (relative) time decay...')
        
        # Create a user-specific time-weighted matrix
        user_time_weighted_matrix = sparse_matrix.copy()
        user_weights = np.zeros_like(sparse_matrix.data)
        
        # Process each user (row) to calculate user-specific decay
        for i in range(sparse_matrix.shape[0]):
            # Get the row for user i
            user_row = sparse_matrix.getrow(i)
            
            if user_row.nnz == 0:  # Skip if user has no interactions
                continue
            
            # Extract timestamps for this user
            user_timestamps = user_row.data
            
            # Calculate the user's earliest timestamp
            user_start = np.min(user_timestamps)
            
            # Calculate relative decay: (timestamp - user_earliest) / day_in_seconds
            user_rel_decay = (user_timestamps - user_start) / _win_unit
            
            # Apply exponential decay to user-relative time differences
            user_row_weights = np.exp(decay_rate * user_rel_decay)
            
            # Update the weights for this user
            user_weights[sparse_matrix.indptr[i]:sparse_matrix.indptr[i+1]] = user_row_weights
        
        if decay_mode == 'user':
            time_weighted_matrix.data = user_weights
    
    # If using both decay types, combine them with beta as scaling factor
    if decay_mode == 'both':
        if verbose > 0:
            print(f'Combining global and user-specific decay with beta={beta}...')
        
        # Combined decay = global_decay + beta * user_decay
        combined_weights = global_weights + beta * user_weights
        time_weighted_matrix.data = combined_weights
    
    if verbose > 0:
        print('Computing temporal cosine similarity...')
    
    # Compute sparse cosine similarity with time weights
    temporal_sim = cosine_similarity(time_weighted_matrix, dense_output=False)
    
    # --- Step 3: Combine the two similarity matrices ---
    if verbose > 0:
        print(f'Combining similarities with alpha={alpha}...')
    
    alpha = config["alpha"]
    
    # Combine using sparse matrix operations: alpha * binary_sim + (1-alpha) * temporal_sim
    combined_sim = binary_sim.multiply(alpha) + temporal_sim.multiply(1-alpha)
    
    # If self_loop is False, set the diagonal to zero
    if self_loop:
        combined_sim.setdiag(1)
    else:
        combined_sim.setdiag(0)
    
    # --- Step 4: Filter to top K values ---
    filtered_data = []
    filtered_rows = []
    filtered_cols = []
    
    if verbose > 0:
        print('Filtering top-k values...')
    
    pbar = tqdm(range(combined_sim.shape[0]), bar_format='{desc}{bar:30} {percentage:3.0f}% | {elapsed}{postfix}', ascii="░❯")
    pbar.set_description(f"Preparing combined similarity matrix | Top-K: {top_k}")
    
    for i in pbar:
        # Get the non-zero elements in the i-th row
        row = combined_sim.getrow(i).tocoo()
        if row.nnz == 0:
            continue
        
        # Extract indices and values of the row
        row_data = row.data
        row_indices = row.col
        
        # Sort indices based on similarity values (in descending order) and select top K
        if row.nnz > top_k:
            top_k_idx = np.argsort(-row_data)[:top_k]
        else:
            top_k_idx = np.argsort(-row_data)
        
        # Store the top K similarities
        filtered_data.extend(row_data[top_k_idx])
        filtered_rows.extend([i] * len(top_k_idx))
        filtered_cols.extend(row_indices[top_k_idx])
    
    # Construct the final filtered sparse matrix
    filtered_similarity_matrix = coo_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=combined_sim.shape)
    
    # Clean up to free memory
    del sparse_matrix, binary_matrix, time_weighted_matrix
    del binary_sim, temporal_sim, combined_sim
    del filtered_data, filtered_rows, filtered_cols
    
    if decay_mode == 'user' or decay_mode == 'both':
        del user_weights
        if 'user_time_weighted_matrix' in locals():
            del user_time_weighted_matrix
    
    if decay_mode == 'global' or decay_mode == 'both':
        del global_weights
    
    return filtered_similarity_matrix.tocsr()