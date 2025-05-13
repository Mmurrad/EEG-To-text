# EEG-to-Text Translation: A Model for Deciphering Human Brain Activity 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) This repository contains the code and resources for the **R1 Translator** model, designed for decoding human brain activity (EEG signals) into text, as presented in the paper *"EEG-to-Text Translation: A Model for Deciphering Human Brain Activity"*.

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
