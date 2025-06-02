import os
import multiprocessing as mp
import time
import signal
from typing import Callable, Tuple


def work() -> None:
    process_name = mp.current_process().name
    process_id = os.getpid()

    online = True

    def handler(signum, frame):
        nonlocal online
        online = False
        raise KeyboardInterrupt("Triggering KeyboardInterrupt in worker.")

    signal.signal(signal.SIGINT, handler)

    try:
        while online:
            time.sleep(2)
    except KeyboardInterrupt:
        print(f"Worker '{process_name}' handler caught signal.")
    finally:
        print(f"Worker '{process_name}' shutting down.")


def spawn(
    name: str, target: Callable[..., None] = work, args: Tuple = ()
) -> mp.Process:
    print(f"Attempting to spawn process named '{name}'.")
    process = mp.Process(name=name, target=target, args=args)
    process.start()
    print(f"Process '{process.name}' has been started successfully.")
    return process


def kill(process_obj: mp.Process, signal_to_send=signal.SIGINT) -> bool:
    process_name = process_obj.name if process_obj.name else "unknownProcess"
    if not process_obj.is_alive():
        print(f"Process '{process_name}' is not alive. No signal sent.")
        return False

    pid_to_kill = process_obj.pid

    if pid_to_kill is None:
        print(
            f"Error: Process '{process_name}' is alive but its PID is not available. Cannot send signal."
        )
        return False

    try:
        os.kill(pid_to_kill, signal_to_send)
        print(
            f"Signal {signal.Signals(signal_to_send).name} sent to process '{process_name}'."
        )
        return True
    except ProcessLookupError:
        print(f"Process '{process_name}' likely exited just now.")
        return False
    except PermissionError:
        print(f"Permission denied to send signal to process '{process_name}'.")
        return False
    except Exception as e:
        print(
            f"An unexpected error occurred while trying to signal process '{process_name}': {e}"
        )
        return False
