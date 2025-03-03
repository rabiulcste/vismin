import os
import random
import time
import json
import fcntl
from contextlib import contextmanager
from datetime import date, datetime
from datetime import datetime, date
import numpy as np
import pandas as pd
import torch
import threading

def set_random_seed():
    # Use SLURM_ARRAY_TASK_ID if it's available, otherwise default to 0
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
    seed = int(time.time() * 1000) + os.getpid() + int(task_id)
    random.seed(seed)


def load_json_data(fpath: str):
    """Load JSON data from a file."""
    try:
        with open(fpath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {fpath}")
    except json.JSONDecodeError:
        logger.error(f"Failed to load JSON data from {fpath}")
    except Exception as e:
        logger.error(f"Unexpected error loading {fpath}: {e}")

    return None


def save_to_json(data, filepath):
    """Save data to a JSON file."""
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved data to {filepath}")


def format_caption(caption: str):
    caption = caption.strip()
    caption = caption[0].upper() + caption[1:]
    return caption if caption.endswith(".") else caption + "."




class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy numbers
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        # Handle pandas DataFrame and Series
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        if hasattr(obj, "tolist"):  # Convert tensors to lists
            return obj.tolist()
        elif hasattr(obj, "name"):  # Convert PyTorch device to its name string
            return obj.name
        elif isinstance(obj, type):  # If it's a type/class object
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, torch.device):  # Handling torch.device objects
            return str(obj)
        elif isinstance(obj, torch.dtype):  # Handling torch.dtype objects
            return str(obj)

        # Handle other non-serializable objects or custom classes
        # By default, convert them to string (change this if needed)
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)
    
class FileLocker:
    def __init__(self):
        self.locks = {}
        self.locks_lock = threading.Lock()

    def acquire_lock(self, directory_path):
        """
        Acquire lock for a specific directory to prevent other scripts from accessing it.
        """
        lock_file_path = directory_path + ".lock"
        if os.path.exists(lock_file_path):
            raise RuntimeError(f"Lock file already exists for directory: {directory_path}")

        lock_file = open(lock_file_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            with self.locks_lock:
                self.locks[directory_path] = lock_file
        except Exception:
            lock_file.close()
            os.remove(lock_file_path)
            raise

    def release_lock(self, directory_path):
        """
        Release the lock for a specific directory.
        """
        with self.locks_lock:
            lock_file = self.locks.pop(directory_path, None)
            if lock_file:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                except Exception as e:
                    # Log the error but continue cleanup
                    print(f"Error releasing lock for directory: {directory_path}. Error: {e}")
                finally:
                    lock_file.close()
                    lock_file_path = directory_path + ".lock"
                    try:
                        os.remove(lock_file_path)
                    except Exception as e:
                        # Log the error but continue cleanup
                        print(f"Error removing lock file: {lock_file_path}. Error: {e}")

    def __del__(self):
        """
        Ensure locks are released when the FileLocker instance is destroyed.
        """
        for directory_path in list(self.locks.keys()):
            self.release_lock(directory_path)

    @contextmanager
    def locked(self, directory_path):
        lock_acquired = False
        try:
            self.acquire_lock(directory_path)
            lock_acquired = True
        except Exception as e:
            print(f"Error while trying to acquire lock on {directory_path}: {e}")
            lock_acquired = False  # Ensure we communicate failure to acquire the lock.

        try:
            yield lock_acquired  # Yield the state of lock acquisition.
        finally:
            # Cleanup: release the lock if it was acquired.
            if lock_acquired:
                self.release_lock(directory_path)