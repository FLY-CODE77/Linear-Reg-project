def get_data_batch(dataset, batch_idx, batch_size, n_batch):
    if batch_idx is n_batch -1:
        batch = dataset[batch_idx * batch_size:]

    else:
        batch = dataset[batch_idx * bach_size : (batch_id + 1) * batch_size]
        
    return batch 
