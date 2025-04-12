import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer
from tqdm import tqdm
#from fuzzy_match import match
#from fuzzy_match import algorithims

# Normalize a 1D tensor
def normalize_1d(input_tensor):
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean) / std
    return input_tensor

# Function to prepare input samples from ZuCo dataset
def get_input_sample(sent_obj, tokenizer, eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'], max_len=56, add_CLS_token=False, test_input="noise"):
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type + band])
        word_eeg_embedding = np.concatenate(frequency_features)

        if len(word_eeg_embedding) != 105 * len(bands):
            print(f'Expect word EEG embedding dim to be {105 * len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None

        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean' + band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])

        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105 * len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        return None

    input_sample = {}
    # Get target label (text)
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask=True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # Get sentence-level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None

    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # Get sentiment label (dummy value for now)
    input_sample['sentiment_label'] = torch.tensor(-100)  # Dummy value

    # Get input embeddings (EEG data for each word)
    word_embeddings = []
    if add_CLS_token:
        word_embeddings.append(torch.ones(105 * len(bands)))

    for word in sent_obj['word']:
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        if word_level_eeg_tensor is None or torch.isnan(word_level_eeg_tensor).any():
            return None
        word_embeddings.append(word_level_eeg_tensor)

    # Pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))

    if test_input == 'noise':
        rand_eeg = torch.randn(torch.stack(word_embeddings).size())
        input_sample['input_embeddings'] = rand_eeg  # Random noise for testing
    else:
        input_sample['input_embeddings'] = torch.stack(word_embeddings)  # Actual EEG data

    # Mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len)  # 0 is masked out
    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word']) + 1] = torch.ones(len(sent_obj['word']) + 1)  # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word']))  # 1 is not masked

    # Mask out padding tokens (inverted)
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)  # 1 is masked out
    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word']) + 1] = torch.zeros(len(sent_obj['word']) + 1)  # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word']))  # 0 is not masked

    # Mask out target padding for loss computation
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    # Discard zero-length data
    if input_sample['seq_len'] == 0:
        print('Discard length zero instance: ', target_string)
        return None

    return input_sample

# Dataset class for ZuCo
class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL', eeg_type='GD', bands=['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2'], setting='unique_sent', is_add_CLS_token=False, test_input='noise'):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO] Loading {len(input_dataset_dicts)} task datasets')

        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO] Using subjects: ', subjects)
            else:
                subjects = [subject]

            total_num_sentence = len(input_dataset_dict[subjects[0]])
            train_divider = int(0.8 * total_num_sentence)
            dev_divider = train_divider + int(0.1 * total_num_sentence)

            print(f'Train divider = {train_divider}')
            print(f'Dev divider = {dev_divider}')

            if setting == 'unique_sent':
                if phase == 'train':
                    print('[INFO] Initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO] Initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider, dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO] Initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider, total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input)
                            if input_sample is not None:
                                self.inputs.append(input_sample)

            print('++ Adding task to dataset, now we have:', len(self.inputs))

        print('[INFO] Input tensor size:', self.inputs[0]['input_embeddings'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'],
            input_sample['seq_len'],
            input_sample['input_attn_mask'],
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'],
            input_sample['target_mask'],
            input_sample['sentiment_label'],
        )

# Main block for testing
if __name__ == '__main__':
    whole_dataset_dicts = []

    # Load ZuCo dataset
    dataset_path_task1 = '/home/student/code/EEG-to-Text/Data/pickle_file/task1-SR-dataset.pickle'
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    dataset_path_task2 = '/home/student/code/EEG-to-Text/Data/pickle_file/task2-NR-dataset.pickle'
    with open(dataset_path_task2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    # Uncomment if you want to load task2_v2 dataset
    # dataset_path_task2_v2 = '/path/to/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle'
    # with open(dataset_path_task2_v2, 'rb') as handle:
    #     whole_dataset_dicts.append(pickle.load(handle))

    # Initialize tokenizer and dataset
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset_setting = 'unique_sent'
    subject_choice = 'ALL'
    eeg_type_choice = 'GD'
    bands_choice = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)

    print('Train set size:', len(train_set))
    print('Dev set size:', len(dev_set))
    print('Test set size:', len(test_set))
