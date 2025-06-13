from tqdm import tqdm
import os
import sys
import torchaudio as ta
import subprocess
import torch as th
from .audio import AudioFile, convert_audio

def get_size(file_path, unit='kb'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)
    
class DemucsDataSet:
    def __init__(self, input_path, audio_channels, samplerate, out, model_name, ext, audiolength, drop_kb = 180):
        self.path = input_path
        self.file_list = list(self.path.rglob('**/*.mp3')) + list(self.path.rglob('**/*.wav')) + list(self.path.rglob('**/*.m4a'))

        print("Number of initially loaded files : ", len(self.file_list))
        ffiles = []
        for file in tqdm(self.file_list):    
            if (out / model_name / file.parent.relative_to(self.path) / (str(file.name.rsplit(".", 1)[0]) + '.' + ext)).exists() == False and get_size(file) > drop_kb:
                ffiles.append(file)
        self.file_list = ffiles
        print("Number of files which will be separated : ", len(self.file_list))

        self.audio_channels = audio_channels
        self.samplerate = samplerate
        self.audiolength = audiolength
    
    def __getitem__(self, idx):
        fn = self.file_list[idx]

        wav = load_track(fn, self.audio_channels, self.samplerate)

        if len(wav.shape) == 1 :
            th.stack([wav,wav], dim=0)
        if wav.shape[-1] >= self.audiolength :
            wav = wav[:, : self.audiolength]
        else :
            wav = th.concat([wav,th.zeros(wav.shape[0],self.audiolength - wav.shape[1])], dim = -1)

        is_zero = wav == 0
        wav = wav + is_zero * 1e-7 #adding eps to zeros so that can be devided by mean value at a line below.

        ref = wav.mean(0)
        wav -= ref.mean()
        wav /= ref.std()
        return (wav, ref.mean(), ref.std(), str(fn))
    
    def __len__(self):
        return len(self.file_list)
    
def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'FFmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav
