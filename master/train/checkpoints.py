"""Checkpoint utilities for saving and loading model + optimizer state.

Provides atomic save and convenience helpers for keeping `last` and `best`
checkpoints.
"""

import os
import tempfile
import torch
import re, ast

    # saving file as ini format
def save_file_as_ini(var, path):
    """Save dictionary or list `var` to `path` in INI format."""
    # ensure directory exists
    dirpath = os.path.dirname(path) or '.'
    os.makedirs(dirpath, exist_ok=True)
    
    # write INI file with [config] header for dict or list
    if isinstance(var, dict):
        with open(path, 'w') as f:
            f.write("[defaults]\n")
            for key, value in var.items():
                f.write(f"{key} = {value}\n")
        return
    
    if isinstance(var, list):
        with open(path, 'w') as f:
            for i, value in enumerate(var):
                f.write(f"{i} = {value}\n")
        return
    
def read_file_from_ini(path, ftype=dict):
    """Read INI file from `path` and return as a dictionary or list."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"INI file not found at {path}")

    if ftype == list:
        result = []
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(';') or s.startswith('#') or s.startswith('['):
                    continue
                if '=' in line:
                    _, value = line.split('=', 1)
                    vstr = value.strip()
                    # Try to parse numeric or tuple/list values like '(0, 0.001)'
                    try:
                        v = ast.literal_eval(vstr)
                    except Exception:
                        try:
                            v = float(vstr)
                        except Exception:
                            v = vstr
                    result.append(v)
        return result

    if ftype == dict:
        # First, try a simple key = value parse with type conversion
        try:
            result = {}
            with open(path, 'r') as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith(';') or s.startswith('#') or s.startswith('['):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        vstr = value.strip()
                        # Try to convert to appropriate type
                        try:
                            v = ast.literal_eval(vstr)
                        except Exception:
                            try:
                                v = float(vstr)
                            except Exception:
                                v = vstr
                        result[key.strip()] = v
            if result:
                return result
        except Exception:
            # fall through to more advanced parsing
            pass

        # Fallback: parse numeric-indexed entries like "0 = (1, 2, 3)" etc.
        pattern = re.compile(r'^\s*(\d+)\s*=\s*(.+)$')
        data = {}
        with open(path, 'r') as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith(';') or s.startswith('#') or s.startswith('['):
                    continue
                m = pattern.match(s)
                if not m:
                    continue
                key_s, rhs = m.groups()
                try:
                    val = ast.literal_eval(rhs)
                except Exception:
                    # try forcing parentheses, then fallback to numeric split
                    try:
                        val = ast.literal_eval('(' + rhs + ')')
                    except Exception:
                        parts = re.split(r'[,\s]+', rhs.strip('() '))
                        parts = [p for p in parts if p != '']
                        conv = []
                        for p in parts:
                            if re.search(r'[.eE]', p):
                                try:
                                    conv.append(float(p))
                                except ValueError:
                                    conv.append(p)
                            else:
                                try:
                                    conv.append(int(p))
                                except ValueError:
                                    conv.append(p)
                        val = tuple(conv) if len(conv) > 1 else (conv[0] if conv else rhs)
                data[int(key_s)] = val

        return dict(sorted(data.items()))

    raise ValueError("ftype must be dict or list")

def save_checkpoint(state, path):
    """Save `state` (dict) to `path`, atomically overwriting any existing file."""
    # ensure directory exists
    dirpath = os.path.dirname(path) or '.'
    os.makedirs(dirpath, exist_ok=True)
    # write to a temp file in the same directory and atomically replace the target
    tmp_file = None
    fd = None
    try:
        # mkstemp returns an OS-level fd plus the path; write using the fd
        fd, tmp_file = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path) + '.', suffix='.tmp')
        # open fd as a binary file object and let torch save to it directly
        with os.fdopen(fd, 'wb') as f:
            # writing via the open file avoids reopen/locking issues on Windows
            torch.save(state, f)
        # fd was closed by os.fdopen context manager; avoid double-close in finally
        fd = None
        # atomic replace will overwrite existing checkpoint if present
        os.replace(tmp_file, path)
        tmp_file = None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_file and os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass


def load_checkpoint(path, map_location=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location)


def save_last_and_best(state, ckpt_dir, is_best):
    """Save checkpoint as 'last.pth' and optionally 'best.pth'."""
    # Konverter til CPU én gang
    cpu_state = {
        'epoch': state['epoch'],
        'model_state_dict': _to_cpu(state['model_state_dict']),
        'optimizer_state_dict': _to_cpu(state['optimizer_state_dict']),
        'train_loss': state['train_loss'],
        'val_loss': state['val_loss'],
        'train_mean': state['train_mean'],
        'train_std': state['train_std'],
        'learning_rate': state['learning_rate']
    }

    last_path = os.path.join(ckpt_dir, 'last.pth')
    save_checkpoint(cpu_state, last_path)

    if is_best:
        best_path = os.path.join(ckpt_dir, 'best.pth')
        # Only save model state for best checkpoint to save space, i.e. exclude optimizer state
        # You can no longer resume training from 'best.pth'
        best_state = {k: v for k, v in cpu_state.items() if k != 'optimizer_state_dict'}
        save_checkpoint(best_state, best_path)

def _to_cpu(obj):
    """Recursively move torch tensors inside obj to CPU."""
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj

__all__ = ['save_checkpoint', 'load_checkpoint', 'save_last_and_best', 'save_file_as_ini']
