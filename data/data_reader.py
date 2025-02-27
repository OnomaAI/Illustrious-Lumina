import os
import json
import requests
import logging
import time
from io import BytesIO
from typing import Union

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)

PRIMARY_BASE_URL = (
    "https://huggingface.co/datasets/AngelBottomless/Danbooru2025-test/resolve/main"
)
PRIMARY_CACHE_DIR = "./cache_json_primary"
SECONDARY_TAR_BASE = (
    "https://huggingface.co/datasets/KBlueLeaf/danbooru2023-webp-4Mpixel/resolve/main/images"
)
SECONDARY_JSON_BASE = (
    "https://huggingface.co/datasets/deepghs/danbooru2023-webp-4Mpixel_index/resolve/main/images"
)
SECONDARY_CACHE_DIR = "./cache_json_secondary"

os.makedirs(PRIMARY_CACHE_DIR, exist_ok=True)
os.makedirs(SECONDARY_CACHE_DIR, exist_ok=True)

def primary_subfolder_from_id(x: int) -> str:
    """ Returns a string like '0000', '0001' etc., used in your primary dataset. """
    return f"{x % 1000:04d}"

def secondary_chunk_from_id(x: int, chunk_size=1000) -> int:
    """ Returns the chunk index for the fallback dataset. """
    return x // chunk_size

def download_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    """
    Downloads [start, end] (inclusive) bytes from `url` using a Range request,
    returns as raw bytes.
    """
    headers = {"Range": f"bytes={start}-{end}"}
    r = session.get(url, headers=headers, stream=True)
    r.raise_for_status()
    # Expect status_code==206, but some servers might still return 200
    # if they do not support partial requests. So we won't strictly enforce 206.
    return r.content

def load_primary_json_index(session: requests.Session, folder_name: str) -> dict:
    """
    Downloads/caches <folder_name>.json from the primary dataset and returns it as a dict.
    Example: { "7501000.jpg": [start_offset, end_offset], ... }
    """
    json_url = f"{PRIMARY_BASE_URL}/{folder_name}.json"
    local_path = os.path.join(PRIMARY_CACHE_DIR, f"{folder_name}.json")

    # Check if cached:
    if not os.path.isfile(local_path):
        resp = session.get(json_url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_secondary_json_index(session: requests.Session, chunk_index: int) -> dict:
    """
    Downloads/caches data-000X.json from the secondary dataset index.
    Returns the entire JSON object:
      {
        "files": {
          "1000.webp": {"offset": 1536, "size": 36416, "sha256": "..."},
          ...
        }
      }
    """
    data_name = f"data-{chunk_index:04d}"
    json_url = f"{SECONDARY_JSON_BASE}/{data_name}.json"
    local_path = os.path.join(SECONDARY_CACHE_DIR, f"{data_name}.json")

    if not os.path.isfile(local_path):
        resp = session.get(json_url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    with open(local_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def find_in_primary(session: requests.Session, x: int):
    """
    Tries to find offsets for ID=x in the primary dataset.
    Returns (tar_url, start_offset, end_offset, filename) or None if not found.
    """
    folder_name = primary_subfolder_from_id(x)
    json_index = load_primary_json_index(session, folder_name)
    if json_index is None:
        return None

    filename = f"{x}.jpg"
    if filename not in json_index:
        return None

    start_offset, end_offset = json_index[filename]
    tar_url = f"{PRIMARY_BASE_URL}/{folder_name}.tar"
    return (tar_url, start_offset, end_offset, filename)

def find_in_secondary(session: requests.Session, x: int):
    """
    Tries to find offsets for ID=x in the secondary dataset.
    Returns (tar_url, start_offset, end_offset, filename) or None if not found.
    """
    chunk_index = secondary_chunk_from_id(x, chunk_size=1000)
    data_name = f"data-{chunk_index:04d}"

    data = load_secondary_json_index(session, chunk_index)
    if not data or "files" not in data:
        return None

    filename_key = f"{x}.webp"
    file_dict = data["files"].get(filename_key)
    if not file_dict:
        return None

    offset = file_dict["offset"]
    size = file_dict["size"]
    start_offset = offset
    end_offset = offset + size - 1

    tar_url = f"{SECONDARY_TAR_BASE}/{data_name}.tar"
    return (tar_url, start_offset, end_offset, filename_key)

def download_danbooru_id(x: int) -> BytesIO:
    """
    Finds ID=x in either primary or secondary dataset, issues a partial download,
    and returns the raw file contents as a BytesIO.
    Raises an exception if not found anywhere or if download fails.
    """
    # Load token
    HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
    if not HF_ACCESS_TOKEN:
        if os.path.isfile("env.json"):
            with open("env.json", "r") as f:
                env = json.load(f)
                HF_ACCESS_TOKEN = env["HF_ACCESS_TOKEN"]
        else:
            # Alternatively, read it from your env.json or raise an error
            raise ValueError("HF_ACCESS_TOKEN is not defined in environment.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {HF_ACCESS_TOKEN}"})

    info = find_in_primary(session, x)
    if info:
        tar_url, start_offset, end_offset, filename = info
    else:
        info = find_in_secondary(session, x)
        if not info:
            raise FileNotFoundError(f"ID {x} not found in primary or secondary dataset.")
        tar_url, start_offset, end_offset, filename = info

    # Download the bytes from the tar
    file_bytes = download_range(session, tar_url, start_offset, end_offset)
    return BytesIO(file_bytes)

def read_general(path) -> Union[str, BytesIO]:
    """
    Unified read function:
      - if path.startswith("danbooru://"), parse the ID, attempt partial download,
        return BytesIO of the content.
      - if path.startswith("s3://"), use your existing Ceph client logic.
      - else return path as a string.
    """
    if path.startswith("danbooru://"):
        # Example: path = "danbooru://7502245"
        # parse out the integer ID from the URI
        parts = path.split("://", 1)
        if len(parts) == 2:
            if os.path.exists(parts[1]):
                return parts[1] # did you make mistake here?
            try:
                danbooru_id = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid danbooru ID in path: {path}")
        else:
            raise ValueError(f"Malformed danbooru:// URI: {path}")

        # Download the image data as BytesIO
        return download_danbooru_id(danbooru_id)

    elif path.startswith("s3://"):
        # Your original Ceph logic
        from io import BytesIO
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes

    else:
        # Just return a normal path
        return path


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("./petreloss.conf")
        print("start read image ")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None
