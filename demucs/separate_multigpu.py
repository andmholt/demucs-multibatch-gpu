# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

from dora.log import fatal
import torch as th
import librosa
from torch.utils.data import DataLoader

from tqdm import tqdm

from .apply_multigpu import apply_model
from .apply import BagOfModels
from .audio import save_audio
from .htdemucs import HTDemucs
from .pretrained import get_model_from_args, add_model_flags, ModelLoadingError
from .data_utils import get_size, DemucsDataSet, load_track

import gc

def get_parser():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("input_path", type=Path, help='Path to tracks')
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
    parser.add_argument("-c",
                        "--clone_subdir",
                        default=None,
                        help="Cloning sub-directories to the output directory. Get the base Input directory path as input. If None, not cloning.")
    parser.add_argument("-b", "--n_batch",
                        default=1,
                        type=int,
                        help="Batch mode True/False")
    parser.add_argument("-l",
                        "--audiolength",
                        type =int,
                        default=1324800,
                        help="Length of the audio(sr) based on model's sr. (44100 based default)")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        default = None,
                        help="Only separate audio into {STEM} and no_{STEM}. If 'inst' only no_vocal will be saved.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--int24", action="store_true",
                       help="Save wav output as 24 bits wav.")
    group.add_argument("--float32", action="store_true",
                       help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action="store_true",
                              help="Convert the output wavs to flac.")
    format_group.add_argument("--mp3", action="store_true",
                              help="Convert the output wavs to mp3.")
    parser.add_argument("-sr",
                        "--sample_rate",
                        type =int,
                        default=None,
                        help="Output sample rate. Resampleing from 44100Hz")
    parser.add_argument("--mp3-bitrate",
                        default=128,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("--mp3-preset", choices=range(2, 8), type=int, default=2,
                        help="Encoder preset of MP3, 2 for highest quality, 7 for "
                        "fastest speed. Default is 2")
    parser.add_argument("-j", "--jobs",
                        default=64,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")
    parser.add_argument("--drop_kb",
                        default=180,
                        type=int,
                        help="Files with size under drop_kb will be omitted, for corrputed file omission.")
    parser.add_argument("--num_worker",
                        default=8,
                        type=int,
                        help="num_worker for DataLoader")

    return parser


def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)

    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if th.cuda.device_count() > 1:
        model = th.nn.DataParallel(model)
        args.device='cuda'

    max_allowed_segment = float('inf')
    if isinstance(model, HTDemucs):
        max_allowed_segment = float(model.module.segment)
    elif isinstance(model, BagOfModels):
        max_allowed_segment = model.module.max_allowed_segment
    if args.segment is not None and args.segment > max_allowed_segment:
        fatal("Cannot use a Transformer model with a longer segment "
              f"than it was trained for. Maximum segment is: {max_allowed_segment}")

    if isinstance(model, BagOfModels):
        print(f"Selected model is a bag of {len(model.models)} models. "
              "You will see that many progress bars per track.")


    model.cpu()
    model.eval()

    if args.stem is not None and (args.stem not in model.module.sources and args.stem != 'inst'):
        fatal(
            'error: stem "{stem}" is not in selected model. STEM must be one of {sources}.'.format(
                stem=args.stem, sources=', '.join(model.module.sources)))
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")

    if args.mp3:
        ext = "mp3"
    elif args.flac:
        ext = "flac"
    else:
        ext = "wav"
    
    kwargs = {
    'samplerate': model.module.samplerate,
    'bitrate': args.mp3_bitrate,
    'preset': args.mp3_preset,
    'clip': args.clip_mode,
    'as_float': args.float32,
    'bits_per_sample': 24 if args.int24 else 16,
    }

    if args.sample_rate is not None : 
            kwargs['samplerate'] = args.sample_rate

    dataset = DemucsDataSet(args.input_path, model.module.audio_channels, model.module.samplerate, args.out, args.name, ext, args.audiolength, drop_kb = 180)
    dataloader = DataLoader(dataset, batch_size = args.n_batch, num_workers = args.num_worker, shuffle = False, pin_memory = True)

    for batch, means, stds, tracks in tqdm(dataloader):
        b_sources = apply_model(model, batch.to(args.device), device=args.device, shifts=args.shifts,
                            split=args.split, overlap=args.overlap, progress=True,
                            num_workers=args.jobs, segment=args.segment)
                    
        for k, sources in enumerate(list(b_sources)):
            sources *= stds[k]
            sources += means[k]
            track = Path(tracks[k])
            print('Saving ',str(track.name))
            subdir = track.relative_to(Path(args.clone_subdir)).parent
            if args.stem is None:
                for source, name in zip(sources, model.module.sources):
                    stem = out / subdir / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                    trackext=track.name.rsplit(".", 1)[-1],
                                                    stem=name, ext=ext)
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    if args.sample_rate is not None :
                        source = librosa.resample(source.detach().cpu().numpy(), orig_sr = model.module.samplerate, target_sr = args.sample_rate)
                    save_audio(th.Tensor(source), str(stem), **kwargs)
            elif args.stem == 'inst':
                args.filename = "{track}.{ext}"
                sources = list(sources)
                stem = out / subdir / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                trackext=track.name.rsplit(".", 1)[-1],
                                                stem=args.stem, ext=ext)
                stem.parent.mkdir(parents=True, exist_ok=True)
                sources.pop(model.module.sources.index('vocals'))
                # Warning : after poping the stem, selected stem is no longer in the list 'sources'
                other_stem = th.zeros_like(sources[0])
                for i in sources:
                    other_stem += i
                stem = out / subdir / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                trackext=track.name.rsplit(".", 1)[-1],
                                                stem="no_"+args.stem, ext=ext)
                stem.parent.mkdir(parents=True, exist_ok=True)
                if args.sample_rate is not None :
                    other_stem = librosa.resample(other_stem.detach().cpu().numpy(), orig_sr = model.module.samplerate, target_sr = args.sample_rate)
                save_audio(th.Tensor(other_stem), str(stem), **kwargs)
            else:
                sources = list(sources)
                stem = out / subdir / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                trackext=track.name.rsplit(".", 1)[-1],
                                                stem=args.stem, ext=ext)
                stem.parent.mkdir(parents=True, exist_ok=True)
                source = sources.pop(model.module.sources.index(args.stem))
                if args.sample_rate is not None :
                    source = librosa.resample(source.detach().cpu().numpy(), orig_sr = model.module.samplerate, target_sr = args.sample_rate)
                save_audio(th.Tensor(source), str(stem), **kwargs)
                # Warning : after poping the stem, selected stem is no longer in the list 'sources'
                other_stem = th.zeros_like(sources[0])
                for i in sources:
                    other_stem += i
                stem = out / subdir / args.filename.format(track=track.name.rsplit(".", 1)[0],
                                                trackext=track.name.rsplit(".", 1)[-1],
                                                stem="no_"+args.stem, ext=ext)
                stem.parent.mkdir(parents=True, exist_ok=True)
                if args.sample_rate is not None :
                    other_stem = librosa.resample(other_stem.detach().cpu().numpy(), orig_sr = model.module.samplerate, target_sr = args.sample_rate)
                save_audio(th.Tensor(other_stem), str(stem), **kwargs)
        del b_sources, sources, other_stem, batch
        gc.collect()

if __name__ == "__main__":
    main()
