"""Validate a RunPod manifest without training or downloading model weights."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', required=True)
    p.add_argument('--df40-external', required=True)
    p.add_argument('--sdxl-external', required=True)
    p.add_argument('--check-files', action='store_true')
    args = p.parse_args()
    df = pd.read_csv(args.index)
    assert len(df) == 310000, len(df)
    assert set(df.split) == {'train', 'val', 'test'}
    assert set(df.label) == {0, 1}
    assert not df.path.duplicated().any()
    dff = df[df.source.str.startswith('dff_')]
    assert dff.groupby('identity').split.nunique().max() == 1
    dfdc = df[df.source.eq('dfdc')]
    groups = {s: set(g.identity) for s, g in dfdc.groupby('split')}
    for a in groups:
        for b in groups:
            if a < b:
                assert not groups[a] & groups[b], f'DFDC overlap {a}/{b}'
    for file, expected_source, expected_count in [
        (args.df40_external, 'df40_external', 6424),
        (args.sdxl_external, 'sdxl_faces_external', 1920),
    ]:
        ext = pd.read_csv(file)
        assert len(ext) == expected_count, (file, len(ext))
        assert ext.source.eq(expected_source).all()
        assert ext.split.eq('external').all()
        assert not ext.path.duplicated().any()
        assert set(ext.path).isdisjoint(set(df.path))
    if args.check_files:
        missing = [p for p in df.path if not Path(p).is_file()]
        assert not missing, f'Missing files: {len(missing)}; first={missing[:3]}'
    print('MANIFEST VALIDATION PASSED: 310000 old images, 6424 DF40 external, 1920 SDXL held-out.')


if __name__ == '__main__':
    main()
