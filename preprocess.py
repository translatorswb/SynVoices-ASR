import argparse
import os
import json
import pandas as pd
import concurrent.futures
import soundfile as sf


PUNCT = [',', '.', '/', ':', ';', '?', '!', '‘', '"', '“', '”']


def handle_punctuation(text):
    for p in PUNCT:
        text = text.replace(p, '')
    text = text.replace('-', ' ')
    return text


def get_duration(filepath):
    info = sf.info(filepath)
    return info.duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter synthetic TTS data based on transcription length and prepare final ASR transcriptions')
    parser.add_argument('dir', type=str, help='Path to the directory containing the synthesized clips')
    parser.add_argument('original_manifest', type=str, help='Path to the original manifest file containing the text')
    parser.add_argument('--transcription_manifest', type=str, default=None, help='Path to the manifest file containing the transcriptions')
    parser.add_argument('--output', type=str, default='processed_manifest.jsonl', help='Path to the output manifest file')
    parser.add_argument('--min', type=float, default=None, help='Minimum ratio of transcription length to original text length')
    parser.add_argument('--max', type=float, default=None, help='Maximum ratio of transcription length to original text length')
    args = parser.parse_args()

    with open(args.original_manifest, 'r') as f:
        orig_items = [json.loads(line) for line in f]

    orig_df = pd.DataFrame(orig_items)
    
    # Process with transcription filtering if transcription manifest is provided
    if args.transcription_manifest:
        with open(args.transcription_manifest, 'r') as f:
            trans_items = [json.loads(line) for line in f]
        
        trans_df = pd.DataFrame(trans_items)
        
        # join the two dataframes on the 'audio_file' column
        merged = orig_df.merge(trans_df, on='audio_file')
        
        # rename 'transcription' to 'pred_text' and 'text' to 'original_text'
        merged = merged.rename(columns={'transcription': 'pred_text', 'text': 'original_text'})
        
        # handle punctuation
        merged['text'] = merged['original_text'].apply(handle_punctuation)
        
        # lowercase the text
        merged['text'] = merged['text'].str.lower()
        
        # calculate the ratio of transcription length to the original text length
        merged['ratio'] = merged['pred_text'].str.len() / merged['text'].str.len()
        
        # filter based on the ratio, only if min or max ratio is provided
        if args.min is not None or args.max is not None:
            print('Filtering...')
            min_ratio = args.min if args.min is not None else 0
            max_ratio = args.max if args.max is not None else float('inf')
            merged = merged[(merged['ratio'] >= min_ratio) & (merged['ratio'] <= max_ratio)]
    else:
        # If no transcription manifest, just use the original data
        merged = orig_df.rename(columns={'text': 'original_text'})
        merged['text'] = merged['original_text'].apply(handle_punctuation).str.lower()
        merged['pred_text'] = None
        merged['ratio'] = None

    # add 'audio_filepath' column containing the absolute path to the audio file
    merged['audio_filepath'] = merged['audio_file'].apply(lambda x: os.path.join(args.dir, x))

    # add 'duration' column containing the duration of the audio file
    print('Calculating durations...')

    filepaths = merged['audio_filepath'].tolist()
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        durations = list(executor.map(get_duration, filepaths))
    merged['duration'] = durations

    # save to a new jsonl file
    cols = ['audio_filepath', 'text', 'original_text', 'duration']
    if args.transcription_manifest:
        cols = cols + ['pred_text', 'ratio']
    merged[cols].to_json(args.output, orient='records', lines=True)

    print(f'Filtered manifest saved to {args.output}')
    if args.transcription_manifest and (args.min is not None or args.max is not None):
        print(f'Original manifest size: {len(orig_items)}')
        print(f'Filtered manifest size: {len(merged)}')
        print(f'Ratio: {len(merged) / len(orig_items)}')
