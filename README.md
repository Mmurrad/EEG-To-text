# EEG-to-Text Translation: A Model for Deciphering Human Brain Activity 


## Overview

The project aims to bridge the gap between the human brain and language processing by translating EEG signals directly into text. We propose the **R1 Translator**, a novel model combining a bidirectional LSTM encoder with a pretrained transformer-based decoder. This approach processes EEG embeddings to capture sequential dependencies via the LSTM, feeding these contextualized representations into the transformer decoder for effective text generation. This work compares the R1 Translator against baseline models like T5 and Brain Translator.

## Features

* Implementation of the R1 Translator model (BiLSTM Encoder + Transformer Decoder).
* Scripts for data preprocessing (e.g., converting .mat files to pickle).
* Scripts for training and evaluating EEG-to-text models.
* Comparison with T5 and Brain Translator baseline models.
`

## Implementaion Process

The implementation follows a pipeline starting with raw EEG data preprocessing (filtering, artifact removal, epoching). Features are then extracted from the cleaned signals and stored in pickle files alongside the corresponding text data. These prepared files are used to train and evaluate the R1 Translator model for the task of text generation from EEG features, with performance measured by metrics like BLEU, WER, CER, and ROUGE.

![R1Translator model architecture and training principle.](figures/EEG_methodology.jpg)

* Caption: R1Translator model architecture and training principle.


## Result

![Model comparison](figures/Model_comaparison.png)

* Caption: Evaluation of EEG-to-Text Decoding Performance on the ZuCo Dataset Using Reading
task SR V1, NR V1 and TSR V1: Results with and without Teaching Force (w/tf) During Model
Evaluation


## Acknowledgements

The core code structure implemented in this project is based on the work presented in the following research article:

* **Are EEG-to-Text Models Working?**
    * Author(s): Hyejeong Jo, Yiqian Yang, Juhyeok Han, Yiqun Duan, Hui Xiong, Won Hee Lee
    * Link: https://arxiv.org/abs/2405.06459

We are grateful for their foundational work which inspired this implementation.

* The original codebase can be found here: https://github.com/NeuSpeech/EEG-To-Text?tab=readme-ov-file

* This work utilized the ZuCo dataset. We thank the dataset creators.
