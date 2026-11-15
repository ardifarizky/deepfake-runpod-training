"""Full-dataset deepfake training for RunPod (Linux + NVIDIA GPU).

- No multi-process DataLoader lambda functions; works with Linux workers.
- Best validation checkpoint + resumable last checkpoint per model.
- Fixed identity/video split supplied by index_v2.csv.
- DF40 val/test and SDXL are always external; never used for fitting/selection.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


class FaceDataset(Dataset):
    def __init__(self, frame, train=False, size=224):
        self.paths = frame['path'].astype(str).tolist()
        self.labels = frame['label'].astype(int).tolist()
        augment = [transforms.RandomHorizontalFlip(0.5),
                   transforms.RandomApply([transforms.GaussianBlur(3)], p=0.12)] if train else []
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            *augment,
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        try:
            with Image.open(self.paths[index]) as im:
                image = im.convert('RGB')
        except Exception:
            # Do not silently substitute random labels: corrupt data is an explicit error.
            raise RuntimeError(f'Unreadable training image: {self.paths[index]}')
        return self.transform(image), self.labels[index]


def make_loader(frame, train, batch, workers):
    return DataLoader(FaceDataset(frame, train), batch_size=batch,
                      shuffle=train, num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0, drop_last=train,
                      prefetch_factor=4 if workers > 0 else None)


@torch.no_grad()
def predict(model, frame, batch, workers, amp=True):
    model.eval()
    probabilities, labels = [], []
    for x, y in make_loader(frame, False, batch, workers):
        x = x.cuda(non_blocking=True)
        with torch.autocast('cuda', enabled=amp):
            output = model(x)
        probabilities.append(torch.softmax(output.float(), 1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def metrics(labels, probabilities):
    pred = (probabilities >= .5).astype(int)
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        'n': int(len(labels)),
        'accuracy': float(accuracy_score(labels, pred)),
        'precision': float(precision_score(labels, pred, zero_division=0)),
        'fake_recall': float(recall_score(labels, pred, zero_division=0)),
        'real_recall': float(((~pred.astype(bool)) & real_mask).sum() / max(real_mask.sum(), 1)),
        'f1': float(f1_score(labels, pred, zero_division=0)),
        'auc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None,
        'confusion_matrix': confusion_matrix(labels, pred).tolist(),
    }


def per_source(frame, labels, probabilities):
    out = {}
    for source, group in frame.assign(_y=labels, _p=probabilities).groupby('source'):
        out[source] = metrics(group._y.to_numpy(), group._p.to_numpy())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--model', choices=['efficientnet', 'swin'], required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    t = cfg['training']
    torch.manual_seed(t['seed']); np.random.seed(t['seed'])
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available. Run on the RunPod GPU template.')

    frame = pd.read_csv(cfg['data']['old_index'])
    # Explicitly guard against using external sources for training.
    assert not frame.source.isin(['df40_external', 'sdxl_faces_external']).any()
    train_df = frame[frame.split.eq('train')].reset_index(drop=True)
    val_df = frame[frame.split.eq('val')].reset_index(drop=True)
    test_df = frame[frame.split.eq('test')].reset_index(drop=True)
    assert len(train_df) + len(val_df) + len(test_df) == len(frame)

    output_dir = Path(cfg['output_dir']); output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'; checkpoint_dir.mkdir(exist_ok=True)
    model = timm.create_model(cfg['models'][args.model], pretrained=True, num_classes=2).cuda()
    if t['channels_last']:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    train_loader = make_loader(train_df, True, t['batch_size'], t['workers'])
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * t['epochs']
    warmup = min(t['warmup_steps'], total_steps // 10)

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return .5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = torch.amp.GradScaler('cuda', enabled=t['amp'])
    criterion = nn.CrossEntropyLoss()
    start_epoch, step, best_auc = 1, 0, -1.0
    history, epochs_without_improvement = [], 0
    last_path = checkpoint_dir / f'{args.model}_last.pt'
    best_path = checkpoint_dir / f'{args.model}_best.pt'

    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model']); opt.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1; step = checkpoint['step']; best_auc = checkpoint['best_auc']
        history = checkpoint['history']; epochs_without_improvement = checkpoint['epochs_without_improvement']
        print(f'Resuming from epoch {start_epoch}, step {step}', flush=True)

    print(json.dumps({'model': args.model, 'train': len(train_df), 'val': len(val_df), 'test': len(test_df),
                      'batch': t['batch_size'], 'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps,
                      'parameters_M': sum(p.numel() for p in model.parameters()) / 1e6}), flush=True)

    for epoch in range(start_epoch, t['epochs'] + 1):
        model.train(); total_loss = correct = seen = 0; start = time.time()
        for x, y in train_loader:
            if step >= total_steps:
                break
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=t['amp']):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip_norm'])
            scaler.step(opt); scaler.update(); sched.step()
            total_loss += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); seen += len(y); step += 1
            if step % 100 == 0:
                print(f'epoch={epoch} step={step}/{total_steps} loss={total_loss/seen:.4f} acc={correct/seen:.4f} {seen/(time.time()-start):.0f} img/s', flush=True)
        y_val, p_val = predict(model, val_df, t['batch_size'], t['workers'], t['amp'])
        val_metrics = metrics(y_val, p_val)
        record = {'epoch': epoch, 'step': step, 'train_loss': total_loss / seen,
                  'train_acc': correct / seen, 'val': val_metrics, 'seconds': time.time() - start}
        history.append(record)
        print('EPOCH_RESULT', json.dumps(record), flush=True)
        epochs_without_improvement += 1
        if val_metrics['auc'] is not None and val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']; epochs_without_improvement = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'step': step,
                        'best_auc': best_auc}, best_path)
            print(f'SAVED_BEST val_auc={best_auc:.6f}', flush=True)
        # Atomic last checkpoint permits safe resume after interruption.
        tmp = last_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'step': step, 'best_auc': best_auc,
                    'history': history, 'epochs_without_improvement': epochs_without_improvement}, tmp)
        tmp.replace(last_path)
        (output_dir / f'{args.model}_history.json').write_text(json.dumps(history, indent=2))
        if epochs_without_improvement >= t['early_stopping_patience']:
            print('EARLY_STOPPING', flush=True)
            break

    if not best_path.exists():
        raise RuntimeError('No best checkpoint produced; check validation data and logs.')
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(best['model'])
    result = {'model': args.model, 'config': cfg, 'history': history,
              'best_epoch': best['epoch'], 'best_val_auc': best['best_auc'],
              'train_n': len(train_df), 'val_n': len(val_df), 'test_n': len(test_df),
              'params_M': sum(p.numel() for p in model.parameters()) / 1e6}
    y_test, p_test = predict(model, test_df, t['batch_size'], t['workers'], t['amp'])
    result['in_domain'] = metrics(y_test, p_test)
    result['per_source'] = per_source(test_df, y_test, p_test)
    pd.DataFrame({'path': test_df.path, 'source': test_df.source, 'label': y_test, 'prob_fake': p_test}).to_csv(output_dir / f'{args.model}_in_domain_preds.csv', index=False)

    if cfg['external']['enabled']:
        external_files = {'df40_external': str(Path(cfg['data']['old_index']).with_name('df40_external.csv')),
                          'sdxl_faces_external': cfg['data']['sdxl_manifest']}
        result['external'] = {}
        for source, file in external_files.items():
            external = pd.read_csv(file)
            if not (external.source.eq(source)).all():
                raise ValueError(f'Unexpected external source in {file}')
            y, prob = predict(model, external, t['batch_size'], t['workers'], t['amp'])
            result['external'][source] = metrics(y, prob)
            pd.DataFrame({'path': external.path, 'source': external.source, 'label': y, 'prob_fake': prob}).to_csv(output_dir / f'{args.model}_{source}_preds.csv', index=False)
            print('EXTERNAL', source, json.dumps(result['external'][source]), flush=True)

    (output_dir / f'{args.model}_metrics.json').write_text(json.dumps(result, indent=2))
    print('TRAINING_COMPLETE', args.model, flush=True)


if __name__ == '__main__':
    main()
