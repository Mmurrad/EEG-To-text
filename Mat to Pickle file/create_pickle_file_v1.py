import os
import pickle
import scipy.io as io
import h5py
from glob import glob
from tqdm import tqdm
import argparse

def load_mat_file(mat_file, version):
    """Load .mat file based on version."""
    if version == 'v1':
        return io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
    elif version == 'v2':
        return h5py.File(mat_file, 'r')
    
def process_sentence_data(sent, task_name):
    """Process the sentence level data and return the structured dictionary."""
    sent_obj = {'content': sent.content}
    sent_obj['sentence_level_EEG'] = {
        'mean_t1': sent.mean_t1, 'mean_t2': sent.mean_t2,
        'mean_a1': sent.mean_a1, 'mean_a2': sent.mean_a2,
        'mean_b1': sent.mean_b1, 'mean_b2': sent.mean_b2,
        'mean_g1': sent.mean_g1, 'mean_g2': sent.mean_g2
    }

    if task_name == 'task1-SR':
        sent_obj['answer_EEG'] = {
            'answer_mean_t1': sent.answer_mean_t1, 'answer_mean_t2': sent.answer_mean_t2,
            'answer_mean_a1': sent.answer_mean_a1, 'answer_mean_a2': sent.answer_mean_a2,
            'answer_mean_b1': sent.answer_mean_b1, 'answer_mean_b2': sent.answer_mean_b2,
            'answer_mean_g1': sent.answer_mean_g1, 'answer_mean_g2': sent.answer_mean_g2
        }
    
    return sent_obj

def process_word_data(word, sent_obj, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask):
    """Process word-level EEG data and add to sentence object."""
    word_obj = {'content': word.content, 'nFixations': word.nFixations}
    
    if word.nFixations > 0:
        word_obj['word_level_EEG'] = {
            'FFD': {f'FFD_{freq}': getattr(word, f'FFD_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']},
            'TRT': {f'TRT_{freq}': getattr(word, f'TRT_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']},
            'GD': {f'GD_{freq}': getattr(word, f'GD_{freq}') for freq in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']}
        }
        sent_obj['word'].append(word_obj)
        word_tokens_has_fixation.append(word.content)
        word_tokens_with_mask.append(word.content)
    else:
        word_tokens_with_mask.append('[MASK]')
    
    word_tokens_all.append(word.content)

def process_mat_files(mat_files, task_name, version):
    """Process all .mat files into a structured dataset."""
    dataset_dict = {}
    
    for mat_file in tqdm(mat_files):
        subject_name = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
        dataset_dict[subject_name] = []

        matdata = load_mat_file(mat_file, version)
        
        for sent in matdata:
            if isinstance(sent.word, float):
                dataset_dict[subject_name].append(None)
                continue
            
            sent_obj = process_sentence_data(sent, task_name)
            sent_obj['word'] = []
            
            word_tokens_all = []
            word_tokens_has_fixation = []
            word_tokens_with_mask = []
            
            for word in sent.word:
                process_word_data(word, sent_obj, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask)

            sent_obj.update({
                'word_tokens_has_fixation': word_tokens_has_fixation,
                'word_tokens_with_mask': word_tokens_with_mask,
                'word_tokens_all': word_tokens_all
            })
            
            dataset_dict[subject_name].append(sent_obj)
    
    return dataset_dict

def save_to_pickle(dataset_dict, task_name, output_dir):
    """Save the dataset to a pickle file."""
    output_name = f'{task_name}-dataset.pickle'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, output_name), 'wb') as handle:
        pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to {os.path.join(output_dir, output_name)}")

def main():
    parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
    parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
    args = vars(parser.parse_args())
    
    task_name = args['task_name']
    version = 'v1'  # You can change this to 'v2' if necessary
    
    input_dir = '/home/student/code/EEG-Code/Data/task3-TSR/'
    output_dir = '/home/student/code/EEG-Code/Data/pickle_file/temp_result/'
    mat_files = sorted(glob(os.path.join(input_dir, '*.mat')))
    
    if not mat_files:
        print(f'No mat files found for {task_name}')
        return
    
    dataset_dict = process_mat_files(mat_files, task_name, version)
    save_to_pickle(dataset_dict, task_name, output_dir)

if __name__ == '__main__':
    main()

