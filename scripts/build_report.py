"""Aggregate full RunPod model results into CSV/XLSX/Markdown and PNG charts."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='/workspace/outputs')
    args = p.parse_args()
    root = Path(args.results)
    files = sorted(root.glob('*_metrics.json'))
    if not files:
        raise SystemExit('No completed metrics. Check training logs before reporting.')
    rows, external, sources = [], [], []
    for path in files:
        m = json.loads(path.read_text())
        test = m['in_domain']
        rows.append({'model': m['model'], 'best_epoch': m['best_epoch'], 'params_M': m['params_M'],
                     'train_n': m['train_n'], 'test_n': m['test_n'],
                     **{f'in_domain_{k}': test[k] for k in ['accuracy', 'auc', 'fake_recall', 'real_recall', 'f1']}})
        for source, v in m.get('external', {}).items():
            external.append({'model': m['model'], 'source': source,
                             **{k: v[k] for k in ['n', 'accuracy', 'auc', 'fake_recall', 'real_recall', 'f1']}})
        for source, v in m.get('per_source', {}).items():
            sources.append({'model': m['model'], 'source': source,
                            **{k: v[k] for k in ['n', 'accuracy', 'auc', 'fake_recall', 'real_recall']}})
    comparison = pd.DataFrame(rows).sort_values('model')
    external_df = pd.DataFrame(external)
    sources_df = pd.DataFrame(sources)
    comparison.to_csv(root / 'model_comparison.csv', index=False)
    external_df.to_csv(root / 'external_comparison.csv', index=False)
    sources_df.to_csv(root / 'per_source_metrics.csv', index=False)
    with pd.ExcelWriter(root / 'full_training_results.xlsx', engine='openpyxl') as writer:
        comparison.to_excel(writer, sheet_name='In-domain', index=False)
        external_df.to_excel(writer, sheet_name='External', index=False)
        sources_df.to_excel(writer, sheet_name='Per-source', index=False)

    if not external_df.empty:
        pivot = external_df.pivot(index='source', columns='model', values='accuracy')
        ax = pivot.plot(kind='bar', ylim=(0, 1), figsize=(9, 5), ylabel='Accuracy',
                        title='External generalization: DF40 and held-out SDXL')
        ax.legend(title='Model'); plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(root / 'external_comparison.png', dpi=180); plt.close()
    report = ['# Full-dataset RunPod results', '', comparison.to_markdown(index=False), '',
              '## External evaluation', external_df.to_markdown(index=False), '',
              'External sets were never used for training or checkpoint selection.',
              'DFDC split is disjoint by video ID; DFF aligned variants share splits.']
    (root / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(comparison.to_string(index=False))
    print(external_df.to_string(index=False))
    print('REPORT_COMPLETE', root / 'full_training_results.xlsx')


if __name__ == '__main__':
    main()
