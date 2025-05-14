import os
import subprocess
import wget
import glob
from zipfile import ZipFile
from tqdm import tqdm


def main():
    # This is where the noise samples will be placed.
    noise_samples = 'noise_samples'
    if not os.path.exists(noise_samples):
        os.makedirs(noise_samples)

    # Download noise samples
    rirs_noises_path = noise_samples + '/rirs_noises.zip'
    if not os.path.exists(rirs_noises_path):
        rirs_noises_url = 'https://www.openslr.org/resources/28/rirs_noises.zip'  
        wget.download(rirs_noises_url, noise_samples)
        print(f"Dataset downloaded at: {rirs_noises_path}")
    else:
        print("Zipfile already exists.")

    # Unzip the downloaded file
    if not os.path.exists(noise_samples + '/RIRS_NOISES'):
        with ZipFile(rirs_noises_path, "r") as zipObj:
            zipObj.extractall(noise_samples)
            print("Extracting noise data complete")
        # Convert 8-channel audio files to mono-channel
        wav_list = glob.glob(noise_samples + '/RIRS_NOISES/**/*.wav', recursive=True)
        for wav_path in tqdm(wav_list, desc="Converting 8-channel noise data to mono-channel"):
            mono_wav_path = wav_path[:-4] + '_mono.wav'
            cmd = f"sox {wav_path} {mono_wav_path} remix 1"
            subprocess.call(cmd, shell=True)
        print("Finished converting the 8-channel noise data .wav files to mono-channel")
    else: 
        print("Extracted noise data already exists. Proceed to the next step.")


if __name__ == '__main__':
    main()
