# Transcribe Dataset (Optional)

```bash
python transcribe.py /mnt/md0/synvoices/data/hausa_asr_dataset/synthesized hau 24000 --device 1 --output hausa_transcriptions.jsonl
```

# Preprocess Dataset

```bash
python preprocess.py /mnt/md0/synvoices/data/chichewa_asr_dataset/synthesized /mnt/md0/synvoices/data/chichewa_asr_dataset/manifest.jsonl --output chichewa_processed_manifest.jsonl
```

To filter the dataset based on the ratio of the length of the transcription to the length of text, you can use the following command:

```bash
python preprocess.py /mnt/md0/synvoices/data/hausa_asr_dataset/synthesized /mnt/md0/synvoices/data/hausa_asr_dataset/manifest.jsonl --transcription_manifest hausa_transcriptions.jsonl --min 0.85 --max 1.06 --output hausa_filtered_manifest.jsonl
```

# Download RIRS Noises dataset

```bash
python download_rirs_noises.py
```

# Create Augmented Dataset

```bash
python create_augmented_dataset.py \
    --manifest filtered_manifest.jsonl \
    --rirs_noises_path noise_samples/RIRS_NOISES \
    --output_dir /mnt/md0/synvoices/data/hausa_asr_filtered_augmented
```

