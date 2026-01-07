import signal
import sys
import threading
import time
import queue
import select
import os
import signal
from tqdm import tqdm
import contextlib

@contextlib.contextmanager
def input_listener_context(daemon: bool = True, join_timeout: float = 2.0,
                           show_banner: bool = True, add_epochs_by: int = 10,
                           command_map: dict = None, verbose: bool = True):
    """
    Starts listener thread and ensures it stops on exit.
    """
    if show_banner and verbose:
        print("\n" + "="*60)
        print("Interactive commands available during training:")
        print("  'h' or 'halve' - Halve the learning rate for next epoch")
        print(f"  'a' or 'add'   - Add {add_epochs_by} more epochs to training")
        print("  'q' or 'quit'  - Stop training after current epoch")
        print("(Type command and press Enter at any time)")
        print("="*60 + "\n")
    command_queue = queue.Queue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=_input_listener,
        args=(command_queue, stop_event, add_epochs_by, command_map),
        daemon=daemon
    )
    input_thread.start()
    try:
        yield command_queue, stop_event, input_thread
    finally:
        stop_event.set()
        try:
            input_thread.join(timeout=join_timeout)
        except Exception:
            pass

# New: encapsulate the command handling loop so it can be reused/imported
def handle_input_commands(command_queue, optimizer, total_epochs, epoch):
    """
    Process all pending commands from the input thread.

    Args:
        command_queue: queue.Queue instance used by input_listener
        optimizer: optimizer instance (will be updated for halve_lr)
        total_epochs: current total_epochs (int) - may be increased by add_epochs
        epoch: current epoch index (used only for messaging)

    Returns:
        (updated_total_epochs, early_stop_bool)
    """
    early_stop = False
    while not command_queue.empty():
        cmd, value = command_queue.get()
        if cmd == 'halve_lr':
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = current_lr / 2.0
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"✓ Learning rate manually halved: {current_lr:.2e} → {new_lr:.2e}")
            tqdm.write(f"{'='*60}\n")
        elif cmd == 'add_epochs':
            total_epochs += value
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"✓ Added {value} more epochs: now training until epoch {total_epochs}")
            tqdm.write(f"{'='*60}\n")
        elif cmd == 'stop_training':
            early_stop = True
    return total_epochs, early_stop

def _input_listener(command_queue, stop_event, add_epochs_by=None, command_map=None):
    """
    Background thread that listens for user commands.
    command_map: optional dict mapping input strings -> (cmd_name, value_or_None)
    """
    buffer = ""
    while not stop_event.is_set():
        try:
            if sys.platform != 'win32':
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    char = sys.stdin.read(1)
                    if char == '\n':
                        user_input = buffer.strip().lower()
                        buffer = ""
                        if command_map and user_input in command_map:
                            cmd, val = command_map[user_input]
                            command_queue.put((cmd, val))
                            tqdm.write(f"\n✓ Command received: {user_input}\n")
                        else:
                            # built-in defaults
                            if user_input in ['h', 'halve']:
                                command_queue.put(('halve_lr', None))
                                tqdm.write("\n✓ Command received: Learning rate will be halved after current epoch completes\n")
                            elif user_input in ['a', 'add']:
                                command_queue.put(('add_epochs', add_epochs_by))
                                tqdm.write(f"\n✓ Command received: {add_epochs_by} more epochs will be added to training\n")
                            elif user_input in ['q', 'quit']:
                                command_queue.put(('stop_training', None))
                                tqdm.write("\n✓ Command received: Training will stop after current epoch completes\n")
                                break
                            elif user_input:
                                tqdm.write(f"\n⚠ Unknown command: '{user_input}'. Use 'h', 'a', or 'q'.\n")
                    else:
                        buffer += char
            else:
                # Windows fallback - optionally replace with msvcrt-based non-blocking read
                time.sleep(0.1)
        except (EOFError, KeyboardInterrupt):
            # signal stop and exit
            stop_event.set()
                       # Print a user-visible message and stop the listener.
            # Use tqdm.write so progress bars are not corrupted.
            tqdm.write("\n✓ Keyboard interrupt (Ctrl-C) received — stopping input listener.\n")
            stop_event.set()
            # KeyboardInterrupt: ensure the main process gets interrupted.
            # Send SIGINT to the process so the main thread receives KeyboardInterrupt.
            tqdm.write("\n✓ Keyboard interrupt (Ctrl-C) received — forwarding to main program.\n")
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except Exception:
                # If os.kill fails for some reason, re-raise to at least stop this thread.
                raise
            break
        except Exception:
            # ignore other errors to avoid crashing training
            pass

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n" + "="*60)
    print("Training interrupted by user (Ctrl+C)")
    print("="*60)
    sys.exit(130)

def install_quiet_signal_handlers(quiet_exceptions: bool = True):
    """
    Install SIGINT handler and optional silent excepthook.
    Call this once from the main thread before starting training.
    """
    # SIGINT -> clean exit without traceback
    signal.signal(signal.SIGINT, signal_handler)

    if quiet_exceptions:
        def _silent_excepthook(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                sys.exit(getattr(exc_value, "code", 1) or 0)
            # Short one-line message for other exceptions (or return to silence completely)
            print(f"{exc_type.__name__}: {exc_value}", file=sys.stderr)
        sys.excepthook = _silent_excepthook


___all__ = ['input_listener_context', 'handle_input_commands', 'signal_handler', 'install_quiet_signal_handlers']