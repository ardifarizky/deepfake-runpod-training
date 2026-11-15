"""Convert locally built index paths to the RunPod volume layout.

Expected local inputs (do not retrain on them before exporting):
  D:/deepfake_project/index_v2.csv
  D:/deepfake_project/df40_external.csv
  D:/deepfake_project/df40_face_index.csv (optional diagnostic only)
On RunPod, copy the three existing image roots to /workspace/data/:
  cache/           <- C:/Users/USER/deepfake_cache (or D:/deepfake_cache)
  df40_mirror/     <- D:/Dataset/df40_mirror/*_extracted
"""
import argparse
from pathlib import Path
import pandas as pd


def convert(path: str, output: str, prefix: str):
    df = pd.read_csv(path)
    # Original local path can point at either D: or C: cache.
    df['path'] = df['path'].str.replace(r'^[A-Za-z]:/Users/USER/deepfake_cache/', prefix + 'cache/', regex=True)
    df['path'] = df['path'].str.replace(r'^[A-Za-z]:/deepfake_cache/', prefix + 'cache/', regex=True)
    df['path'] = df['path'].str.replace(r'^D:/Dataset/df40_mirror/', prefix + 'df40_mirror/', regex=True)
    missing = ~df.path.str.startswith(prefix)
    if missing.any():
        raise ValueError(f'Unmapped paths in {path}: {df.loc[missing,"path"].head().tolist()}')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f'{output}: {len(df)} rows')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old-index', default='D:/deepfake_project/index_v2.csv')
    p.add_argument('--df40-index', default='D:/deepfake_project/df40_external.csv')
    p.add_argument('--output-dir', default='D:/deepfake_project/runpod_package/manifests')
    p.add_argument('--prefix', default='/workspace/data/')
    a = p.parse_args()
    convert(a.old_index, str(Path(a.output_dir) / 'index_v2.csv'), a.prefix)
    convert(a.df40_index, str(Path(a.output_dir) / 'df40_external.csv'), a.prefix)


if __name__ == '__main__':
    main()
