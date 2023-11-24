import os
import urllib
import zipfile

from hackathon.data_utils.data_loading import DEFAULT_DATASET_FOLDER


def download_dataset(dataset_zip_file_url: str, local_folder_path: str = DEFAULT_DATASET_FOLDER):
    """
    Download the dataset from the zip file and extract it into the local path
    """
    local_folder_path = os.path.expanduser(local_folder_path)
    os.makedirs(local_folder_path, exist_ok=True)
    dataset_zip_file_path = os.path.join(local_folder_path, 'dataset.zip')
    if not os.path.exists(dataset_zip_file_path):
        print(f"Downloading dataset from {dataset_zip_file_url} to {dataset_zip_file_path}")
        urllib.request.urlretrieve(dataset_zip_file_url, dataset_zip_file_path)
    else:
        print(f"Found existing dataset at {dataset_zip_file_path}")
    print(f"Extracting dataset from {dataset_zip_file_path} to {local_folder_path}")
    with zipfile.ZipFile(dataset_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(local_folder_path)
    print(f"Finished extracting dataset from {dataset_zip_file_path} to {local_folder_path}")




