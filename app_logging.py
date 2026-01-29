import os
import traceback
from datetime import datetime


LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_error.log")
DISTRIBUTION_LOG_FILE = os.path.join(os.path.expanduser("~"), "renamer_distribution.log")


def log_exception(e: Exception):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(datetime.now().isoformat() + "\n")
        f.write(str(e) + "\n")
        f.write(traceback.format_exc())
        f.flush()
        os.fsync(f.fileno())


def log_info(message: str):
    entry = f"{datetime.now().isoformat()} INFO {message}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(entry)


def append_distribution_log(entry: str):
    with open(DISTRIBUTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
        f.flush()
        os.fsync(f.fileno())
