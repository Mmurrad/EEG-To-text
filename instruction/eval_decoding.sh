python eval_decoding.py \
    --checkpoint_path checkpoints/decoding/best/task1v1_task2v1_task3v1_finetune_r1Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.pt \
    --config_path config/decoding/task1v1_task2v1_task3v1_finetune_R1Translator_skipstep1_b32_20_30_2e-05_2e-05_unique_sent_EEG.json \
    --test_input EEG \
    --train_input EEG \
    -cuda cuda:0