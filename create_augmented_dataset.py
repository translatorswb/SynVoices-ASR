import os
import librosa
import soundfile as sf
import numpy as np
import argparse
import json
import glob
from tqdm import tqdm
import multiprocessing as mp


MU_SNR = 50
SIGMA_SNR = 15
MU_TGT_VOL = -20
SIGMA_TGT_VOL = 5

def audioread(path, sample_rate = 16000):
   path = os.path.abspath(path)
   if not os.path.exists(path):
      raise ValueError("[{}] does not exist!".format(path))

   x, sr = librosa.load(path, sr=sample_rate)

   assert len(x.shape) == 1, "Audio file is not mono"

   return x, sr


def audiowrite(data, fs, destpath):
   destpath = os.path.abspath(destpath)

   sf.write(destpath, data, fs)


def snr_mixer(clean, noise, snr, tgt=-25):
   # Normalizing
   rmsclean = (clean**2).mean()**0.5
   if rmsclean == 0:
      rmsclean = 1

   scalarclean = 10 ** (tgt / 20) / rmsclean
   clean = clean * scalarclean
   rmsclean = (clean**2).mean()**0.5

   rmsnoise = (noise**2).mean()**0.5
   if rmsnoise == 0:
      rmsnoise = 1

   scalarnoise = 10 ** (tgt / 20) / rmsnoise
   noise = noise * scalarnoise
   rmsnoise = (noise**2).mean()**0.5
   if rmsnoise == 0:
      rmsnoise = 1

   # Set the noise level for a given SNR
   noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
   noisenewlevel = noise * noisescalar
   noisyspeech = clean + noisenewlevel
   return clean, noisenewlevel, noisyspeech


def concatenate_noise_sample(noise, fs, len_clean):
   silence_length = 0.5
   while len(noise) <= len_clean:
      noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))
      noise = np.append(noiseconcat, noise)

   if noise.size > len_clean:
      noise = noise[0:len_clean]

   return noise


def load_noise_samples(noise_path):
   pointsource_noises = glob.glob(noise_path + '/pointsource_noises/*_mono.wav', recursive=True)
   real_rirs_isotropic_noises = glob.glob(noise_path + '/real_rirs_isotropic_noises/*_mono.wav', recursive=True)

   return pointsource_noises + real_rirs_isotropic_noises


def process_item(item, noise_samples, snr_value, tgt_vol_value, manifest_dir, output_dir):
   file_path = os.path.join(manifest_dir, item['audio_filepath'])
   clean, fs = audioread(file_path)
   noise, n_fs = audioread(np.random.choice(noise_samples))

   if fs != n_fs:
      # Resample noise to match clean speech sampling rate
      noise = librosa.resample(noise, orig_sr=n_fs, target_sr=fs)

   if len(noise) > len(clean):
      noise = noise[0:len(clean)]
   elif len(noise) < len(clean):
      noise = concatenate_noise_sample(noise, fs, clean.size)

   clean_snr, noise_snr, noisy_snr = snr_mixer(clean, noise, snr_value, tgt_vol_value)

   duration = len(noisy_snr) / fs
   output_path = os.path.join(output_dir, os.path.basename(file_path))
   audiowrite(noisy_snr, fs, output_path)
   
   return {
      **item,
      'duration': duration,
      'audio_filepath': output_path
   }


if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Add noise to clean speech data")
   parser.add_argument("--manifest", type=str, required=True, help="Path to manifest file")
   parser.add_argument("--rirs_noises_path", type=str, required=True, help="Path to RIRs noises dataset")
   parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
   parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of worker processes")

   args = parser.parse_args()

   items = []
   with open(args.manifest, "r") as f:
      for line in f:
         items.append(json.loads(line))
   total = len(items)
   print("Total items: {}".format(total))

   noise_samples = load_noise_samples(args.rirs_noises_path)
   print("Total noise samples: {}".format(len(noise_samples)))

   # randomly sample SNR and target volume from a normal distribution
   snr = np.random.normal(MU_SNR, SIGMA_SNR, total)
   tgt_vol = np.random.normal(MU_TGT_VOL, SIGMA_TGT_VOL, total)
   print("SNR mean: {:.2f}, std: {:.2f}".format(np.mean(snr), np.std(snr)))
   print("Target volume mean: {:.2f}, std: {:.2f}".format(np.mean(tgt_vol), np.std(tgt_vol)))

   # create output directory if it doesn't exist
   output_dir = os.path.join(args.output_dir, "clips")
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)

   manifest_dir = os.path.dirname(args.manifest)
   
   def process_worker(work_tuple):
      item, snr_val, tgt_vol_val = work_tuple
      return process_item(
         item=item,
         noise_samples=noise_samples,
         snr_value=snr_val,
         tgt_vol_value=tgt_vol_val,
         manifest_dir=manifest_dir,
         output_dir=output_dir
      )
   
   # Prepare parameters for each worker - now as a single argument tuple
   work_items = [
      (item, snr[i], tgt_vol[i]) 
      for i, item in enumerate(items)
   ]
   
   # Process files in parallel - use map instead of starmap
   print(f"Processing audio files using {args.num_workers} workers")
   with mp.Pool(processes=args.num_workers) as pool:
      processed_items = list(tqdm(
         pool.map(process_worker, work_items),
         total=len(items),
         desc="Processing audio files"
      ))

   # write the new manifest file
   output_manifest = os.path.join(args.output_dir, "manifest.jsonl")
   with open(output_manifest, "w") as f:
      for item in processed_items:
         f.write(json.dumps(item) + "\n")

   print("Done! Augmented dataset created at {}".format(args.output_dir))
