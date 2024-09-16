# Script to download mnist dataset

import os
import urllib.request
from urllib.error import HTTPError
import gzip
import struct

import numpy as np


def download(url: str, filename: str) -> None:
    """Download the file from the specifies URL if it's not already present."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as response, open(
                filename, "wb"
            ) as out_file:
                data = response.read()  # a `bytes` object
                out_file.write(data)

        except HTTPError as e:
            print(f"HTTP Error: {e.code} {e.reason}")
            print(
                "Failed to download. Please check the URL or your internet connection."
            )
            raise

    else:
        print(f"{filename} already exists. Skipping download.")


def unzip_gz(filename: str, unzipped_filename: str) -> None:
    """Unzip a .gz file."""

    if not os.path.exists(unzipped_filename):
        print(f"Unzipping {filename}...")
        with gzip.open(filename, "rb") as f_in:
            with open(unzipped_filename, "wb") as f_out:
                f_out.write(f_in.read())

        # Delete zipped file
        os.remove(filename)

    else:
        print(f"{unzipped_filename} already exists. Skipping unzipping.")


def read_idx(filename: str) -> np.ndarray:
    """Read IDX file format and return NumPy array."""

    with open(filename, "rb") as f:
        magic_number = struct.unpack(">I", f.read(4))[0]
        data_type = (magic_number >> 8) & 0xFF
        num_dims = magic_number & 0xFF

        dims = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(num_dims))

        if data_type == 0x08:  # unsigned byte
            dtype = np.uint8
        elif data_type == 0x09:  # signed byte
            dtype = np.int8
        elif data_type == 0x0B:  # short (2 bytes)
            dtype = np.int16
        elif data_type == 0x0C:  # int (4 bytes)
            dtype = np.int32
        elif data_type == 0x0D:  # float (4 bytes)
            dtype = np.float32
        elif data_type == 0x0E:  # double (8 bytes)
            dtype = np.float64
        else:
            raise ValueError(f"Unknown data type 0x{data_type:X} in {filename}")

        data = np.frombuffer(f.read(), dtype=dtype)
        data = data.reshape(dims)

    return data


def mnist(dir: str | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    # URLs for the MNIST dataset
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    if dir is None:
        dir = ""
    else:
        os.makedirs(dir, exist_ok=True)

    # Download and unzip files
    for key in files:
        url = base_url + files[key]
        filename = os.path.join(dir, files[key])
        unzipped_filename = filename.replace(".gz", "")
        if not os.path.exists(unzipped_filename):
            download(url, filename)
            unzip_gz(filename, unzipped_filename)
        files[key] = unzipped_filename  # Update to unzipped filename

    # Read data into NumPy arrays
    train_images = read_idx(files["train_images"])
    train_labels = read_idx(files["train_labels"])
    test_images = read_idx(files["test_images"])
    test_labels = read_idx(files["test_labels"])

    if verbose:
        # Display the shapes of the dataset
        print("Training images shape:", train_images.shape)
        print("Training labels shape:", train_labels.shape)
        print("Test images shape:", test_images.shape)
        print("Test labels shape:", test_labels.shape)

    dataset = {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }

    return dataset
