import argparse
import os
import pandas as pd
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")


def get_duration(func_args):
    """
    Calculate the duration of an audio file in seconds.
    """
    audio_file, sample_rate = func_args
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file, sr=sample_rate)
        # Calculate duration
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def find_audio_files(directory):
    """
    Find all audio files in a directory.
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate audio file durations.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--metadata', type=str, help='CSV file with metadata')
    input_group.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--sample_rate', type=int, required=True, help='Sample rate of audio files')
    parser.add_argument('--output', type=str, help='Output file', default='durations.csv')
    args = parser.parse_args()

    # Get filepaths either from CSV or directory
    if args.metadata:
        df = pd.read_csv(args.metadata)
        filepaths = df['audio_filepath'].tolist()
    else:
        filepaths = find_audio_files(args.audio_dir)
        print(f"Found {len(filepaths)} audio files in {args.audio_dir}")

    args_list = [(filepath, args.sample_rate) for filepath in filepaths]

    with Pool(os.cpu_count()) as p:
        durations = list(tqdm(p.imap(get_duration, args_list), total=len(filepaths)))

    data = []
    errors = 0
    total_duration = 0
    
    for filepath, duration in zip(filepaths, durations):
        if duration is not None:
            data.append({
                'audio_filepath': filepath,
                'duration': duration
            })
            total_duration += duration
        else:
            errors += 1

    if errors > 0:
        print(f"Warning: {errors} files could not be processed.")

    # Save durations to CSV
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print(f"Durations saved to {args.output}")
    
    # Print total duration in hours, minutes, seconds format
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total duration: {int(hours)}:{int(minutes):02d}:{seconds:.2f} (HH:MM:SS.ms)")
    print(f"Total duration in seconds: {total_duration:.2f}")
