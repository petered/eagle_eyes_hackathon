import datetime
import os
import shutil
from typing import Optional


def make_backup_file_copy(file_path: str, backup_folder: Optional[str] = None):
    # In format suitable for filename
    backup_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if backup_folder is not None:
        os.makedirs(backup_folder, exist_ok=True)
        backup_path = os.path.join(backup_folder, f"{os.path.split(file_path)[1]}.{backup_time}.backup")
    else:
        backup_path = f"{file_path}.{backup_time}.backup"

    if os.path.isdir(file_path):
        shutil.copytree(file_path, backup_path)
    else:
        shutil.copyfile(file_path, backup_path)
    print(f'Saved backup to {backup_path}')
