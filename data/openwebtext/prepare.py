import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets

# Number of workers in .map() call
num_proc = 8

# Number of workers in load_dataset() call
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load OpenWebText dataset
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # Subsample the train split to make it 1/3 of its original size
    train_size = len(dataset["train"])  # Original train size
    reduced_train_size = train_size // 2  # One-third of the original train size
    reduced_train_split = dataset["train"].train_test_split(test_size=1 - (reduced_train_size / train_size), seed=2357, shuffle=True)
    reduced_train_dataset = reduced_train_split['train']

    # Create a validation split
    split_dataset = reduced_train_split['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename 'test' to 'val'

    # Tokenize the dataset using GPT-2 BPE encoding
    def process(example):
        ids = enc.encode_ordinary(example['text'])  # Encode without special tokens
        ids.append(enc.eot_token)  # Add end-of-text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize each split
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Save each split into its own binary file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        # In Colab, set a custom path for the file, e.g., saving in current directory
        filename = os.path.join(os.getcwd(), f'{split}.bin')  # Use os.getcwd() for current directory
        dtype = np.uint16  # GPT-2 has a max token value of 50256, which is < 2^16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into memory-mapped array
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Example to load the binary file later:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
