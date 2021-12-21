import urllib
import logging
import zipfile
import shutil
import py7zr
from os import remove
from pathlib import Path


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def download_file(url, filename=None):
    log = logging.getLogger(__name__)
    filename = filename or url.split("/")[-1]
    chunk_size = 8192
    filesize = 0
    last_filesize = 0
    with urllib.request.urlopen(url) as Response:
        with open(filename, "wb") as f:
            while True:
                chunk = Response.read(chunk_size)
                if not chunk:
                    break
                datasize = f.write(chunk)
                filesize += datasize
                if (filesize - last_filesize) > chunk_size * 1024:
                    log.info(f"Downloaded {sizeof_fmt(filesize)}")
                    last_filesize = filesize


def extract_zip(filepath):
    dirpath = filepath.with_suffix("")
    extension = filepath.name.split(".")[-1]
    if extension == "zip":
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(dirpath)
    elif extension == "7z":
        with py7zr.SevenZipFile(filepath, mode="r") as zip_ref:
            zip_ref.extractall(dirpath)
    else:
        raise ValueError(f"Unknown zip extension: {extension}")
    return dirpath


def remove_dir(tmp_dir):
    shutil.rmtree(tmp_dir)


def read_text_file(filepath, **kwargs):
    return open(filepath, **kwargs).read()


def download_and_extract_dataset(
    url,
    extract=None,
    read_func=read_text_file,
    data_dir=Path("data"),
    read_kwargs={},
):
    filename = url.split("/")[-1]
    filepath = data_dir / filename
    download_file(url, filepath)
    dirpath = extract_zip(filepath) if extract else data_dir
    extract_files = extract if extract else filename

    if isinstance(extract_files, list):
        text_datasets = [
            read_func(dirpath / name, **read_kwargs) for name in extract_files
        ]
    else:
        text_datasets = read_func(dirpath / extract_files, **read_kwargs)

    if extract:
        remove_dir(dirpath)
    remove(filepath)
    return text_datasets
