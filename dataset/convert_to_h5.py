import argparse
import os
import sys
import shutil
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from .h5 import HDF5Maker


def get_seq(video_dirs, extension="png", target_size=(640, 480)):
    for f in video_dirs:
        images = sorted(glob(os.path.join(f, f"*.{extension}")))
        image_seq = []
        for image_name in images:
            img = Image.open(image_name).convert("RGB").resize(target_size)
            arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
            image_seq.append(arr)
        yield image_seq


def make_h5(data_root, split, target_size=(640, 480), extension="png", out_dir='./h5_ds',
            vids_per_shard=100000, force_h5=False, override=False):
    # If the output directory exists and override is enabled, remove it.
    if os.path.exists(out_dir) and override:
        print(f"Overriding existing output directory: {out_dir}")
        shutil.rmtree(out_dir)

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5, video=True)

    data_dir = os.path.join(data_root, split)
    video_dirs = [os.path.join(data_dir, vd) for vd in sorted(os.listdir(data_dir))]

    seq_generator = get_seq(video_dirs, extension=extension, target_size=target_size)

    for seq in tqdm(seq_generator, total=len(video_dirs), desc=f"Processing '{split}' split"):
        try:
            h5_maker.add_data(seq, dtype='uint8')
        except StopIteration:
            break
        except (KeyboardInterrupt, SystemExit):
            print("Ctrl+C detected! Aborting...")
            break
        except Exception:
            e = sys.exc_info()[0]
            print("ERROR:", e)

    h5_maker.close()


def run_make_h5(data_dir, out_dir, target_size=(640, 480), extension="png",
                vids_per_shard=100000, force_h5=False, override=False):
    """
    Runs the HDF5 conversion pipeline for train, val, and test splits.
    
    This function can be imported and called directly from a notebook.
    """
    print("=== Starting HDF5 Conversion ===")
    for split in ["train", "val", "test"]:
        split_out_dir = os.path.join(out_dir, split)
        print(f"--- Processing split: {split} ---")
        make_h5(
            data_root=data_dir,
            split=split,
            target_size=target_size,
            extension=extension,
            out_dir=split_out_dir,
            vids_per_shard=vids_per_shard,
            force_h5=force_h5,
            override=override
        )
    print("=== HDF5 Conversion Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files", required=True)
    parser.add_argument('--data_dir', type=str, help="Directory with videos", required=True)
    parser.add_argument('--width', type=int, default=640, help="Output image width")
    parser.add_argument('--height', type=int, default=480, help="Output image height")
    parser.add_argument('--extension', type=str, help="Video frames extension", default="png")
    parser.add_argument('--override', action="store_true", help="Override output directories if they exist")
    args = parser.parse_args()

    target_size = (args.width, args.height)
    
    run_make_h5(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        target_size=target_size,
        extension=args.extension,
        override=args.override
    )
