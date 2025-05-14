import argparse
import os
from tqdm import tqdm
import json

from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import torchaudio


def transcribe_audio(audio_file, sample_rate, model, device, processor, resampler):
    sample, sr = torchaudio.load(audio_file)
    if sr != sample_rate:
        raise ValueError(f"Sample rate of {audio_file} is {sr}, but expected {sample_rate}")

    sample = resampler(sample)[0]

    inputs = processor(sample, sampling_rate=16_000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits
    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)

    return transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe audio files')
    parser.add_argument('dir', type=str, help='Directory containing audio files')
    parser.add_argument('lang_iso', type=str, help='Language ISO code')
    parser.add_argument('sample_rate', type=int, help='Sample rate of audio files')
    parser.add_argument('--device', type=int, help='GPU device to use, -1 for CPU', default=-1)
    parser.add_argument('--output', type=str, help='Output file', default='transcriptions.jsonl')
    args = parser.parse_args()

    lang_iso = args.lang_iso

    # set device
    if args.device == -1:
        device = 'cpu'
    else:
        device = f'cuda:{args.device}'

    # load model
    model_id = "facebook/mms-1b-all"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    processor.tokenizer.set_target_lang(lang_iso)
    model.load_adapter(lang_iso)

    # resample to 16 kHz
    resampler = torchaudio.transforms.Resample(orig_freq=args.sample_rate, new_freq=16_000)

    transcribed_files = set()
    if os.path.exists(args.output):
        print(f"Loading existing transcriptions from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                item = json.loads(line)
                transcribed_files.add(item['audio_file'])

    total_files = 0
    skipped_files = 0
    processed_files = 0

    with open(args.output, 'a') as f:
        # transcribe audio files
        for audio_file in tqdm(os.listdir(args.dir)):
            total_files += 1
            if audio_file in transcribed_files:
                skipped_files += 1
                continue

            transcription = transcribe_audio(
                os.path.join(args.dir, audio_file),
                args.sample_rate,
                model,
                device,
                processor,
                resampler
                )

            item = {
                'audio_file': audio_file,
                'transcription': transcription
            }

            # append to jsonl file
            f.write(json.dumps(item) + '\n')
            processed_files += 1

    print(f"Total files: {total_files}")
    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Transcriptions saved to {args.output}")
