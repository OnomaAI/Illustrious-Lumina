import os
import json
import requests
import logging
import time
from io import BytesIO
from typing import Union, Optional, Tuple, Dict, Any, Protocol

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)
client = None  # For Ceph/petrel

########################
# Example: Helpers
########################

def primary_subfolder_from_id(x: int) -> str:
    """ Returns a string like '0000', '0001' etc. """
    return f"{x % 1000:04d}"

def secondary_chunk_from_id(x: int, chunk_size=1000) -> int:
    """ Returns the chunk index for the fallback dataset. """
    return x % chunk_size

def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f"initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa
        client = Client("./petreloss.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")

def download_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    """
    Downloads [start, end] (inclusive) bytes from `url` using a Range request,
    returns as raw bytes.
    """
    headers = {"Range": f"bytes={start}-{end}"}
    r = session.get(url, headers=headers, stream=True)
    r.raise_for_status()
    return r.content


########################
# Caching / Index Load
########################

def load_json_index(
    session: requests.Session,
    json_url: str,
    cache_path: Optional[str] = None,
) -> Optional[Dict]:
    """
    Download and cache JSON from `json_url`. If `cache_path` is provided,
    store/reuse from the local cache.
    Returns the loaded JSON dict or None if 404.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        # Already cached
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    resp = session.get(json_url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    data = resp.json()
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return data


########################
# Repository Protocol
########################

class Repository(Protocol):
    """
    A Protocol that each repository must implement:
      - find_image: given an ID, try to produce (tar_url, start_offset, end_offset, filename).
        Returns None if not found.
    """
    def find_image(self, session: requests.Session, image_id: int) -> Optional[Tuple[str, int, int, str]]:
        ...


########################
# Primary Repository
########################

class PrimaryRepository:
    """
    Example for a 'primary' dataset that:
      - Stores images in .tar files named "NNNN.tar", where NNNN = image_id % 1000
      - Has JSON indexes named "NNNN.json"
      - JSON maps "7501000.jpg" -> [start_offset, end_offset]
      - We store them in a local cache dir
    """

    def __init__(
        self,
        base_url: str,
        cache_dir: str,
    ):
        """
        :param base_url: e.g. "https://huggingface.co/datasets/AngelBottomless/Danbooru2025-test/resolve/main"
        :param cache_dir: e.g. "./cache_json_primary"
        """
        self.base_url = base_url
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def build_primary_id_map(self, json_index: Dict[str, Any]) -> Dict[int, str]:
        """
        JSON might look like: { "7501000.jpg": [start_offset, end_offset], ... }
        We create a map from int(7501000) -> "7501000.jpg"
        """
        out = {}
        for filename in json_index.keys():
            root, _ = os.path.splitext(filename)
            try:
                num = int(root)
                out[num] = filename
            except ValueError:
                pass
        return out

    def find_image(self, session: requests.Session, image_id: int) -> Optional[Tuple[str, int, int, str]]:
        folder = primary_subfolder_from_id(image_id)
        json_name = f"{folder}.json"
        json_url = f"{self.base_url}/{json_name}"
        cache_path = os.path.join(self.cache_dir, json_name)

        json_index = load_json_index(session, json_url, cache_path)
        if not json_index:
            return None

        id_map = self.build_primary_id_map(json_index)
        filename = id_map.get(image_id)
        if not filename:
            return None

        start_offset, end_offset = json_index[filename]
        tar_url = f"{self.base_url}/{folder}.tar"
        return tar_url, start_offset, end_offset, filename


########################
# Secondary Repository
########################

class SecondaryRepository:
    """
    Example for a 'secondary' dataset that:
      - Has chunk-based storage: each chunk is "data-XXXX.tar"
      - Has a matching "data-XXXX.json" with "files": { "1000.webp": {"offset":..., "size":..., ...}, ... }
      - We store them in a local cache dir
    """
    def __init__(
        self,
        tar_base_url: str,
        json_base_url: str,
        cache_dir: str,
        chunk_size: int = 1000
    ):
        """
        :param tar_base_url: e.g. "https://huggingface.co/datasets/KBlueLeaf/danbooru2023-webp-4Mpixel/resolve/main/images"
        :param json_base_url: e.g. "https://huggingface.co/datasets/deepghs/danbooru2023-webp-4Mpixel_index/resolve/main/images"
        :param cache_dir: e.g. "./cache_json_secondary"
        :param chunk_size: default 1000
        """
        self.tar_base_url = tar_base_url
        self.json_base_url = json_base_url
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        os.makedirs(self.cache_dir, exist_ok=True)

    def find_image(self, session: requests.Session, image_id: int) -> Optional[Tuple[str, int, int, str]]:
        chunk_index = secondary_chunk_from_id(image_id, self.chunk_size)
        data_name = f"data-{chunk_index:04d}"

        json_url = f"{self.json_base_url}/{data_name}.json"
        cache_path = os.path.join(self.cache_dir, f"{data_name}.json")

        data = load_json_index(session, json_url, cache_path)
        if not data or "files" not in data:
            return None

        filename_key = f"{image_id}.webp"
        file_dict = data["files"].get(filename_key)
        if not file_dict:
            return None

        offset = file_dict["offset"]
        size = file_dict["size"]
        start_offset = offset
        end_offset = offset + size - 1

        tar_url = f"{self.tar_base_url}/{data_name}.tar"
        return (tar_url, start_offset, end_offset, filename_key)


########################
# Fallback finder
########################

def find_in_fallbacks(
    session: requests.Session,
    image_id: int,
    repositories: list[Repository],
) -> Optional[Tuple[str, int, int, str]]:
    """
    Given a list of repositories, try them in order until we find the image_id.
    Returns (tar_url, start_offset, end_offset, filename) or None if not found in any.
    """
    for repo in repositories:
        info = repo.find_image(session, image_id)
        if info is not None:
            # Found it!
            return info
    return None


########################
# The final download logic
########################

def download_danbooru_id(x: int, repositories: list[Repository]) -> BytesIO:
    """
    Try to find ID=x in a series of repositories (in order).
    Issues a partial download and returns the raw file contents as a BytesIO.
    Raises FileNotFoundError if not found.
    """
    # Load token
    HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
    if not HF_ACCESS_TOKEN and os.path.isfile("env.json"):
        with open("env.json", "r") as f:
            env = json.load(f)
            HF_ACCESS_TOKEN = env.get("HF_ACCESS_TOKEN")

    if not HF_ACCESS_TOKEN:
        raise ValueError("HF_ACCESS_TOKEN is not defined in environment or env.json.")

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {HF_ACCESS_TOKEN}"})

    info = find_in_fallbacks(session, x, repositories)
    if not info:
        raise FileNotFoundError(f"ID {x} not found in any of the fallback repositories.")

    tar_url, start_offset, end_offset, filename = info

    # Download the bytes from the tar
    file_bytes = download_range(session, tar_url, start_offset, end_offset)
    return BytesIO(file_bytes)


########################
# Unified read function
########################

def read_general(path: str) -> Union[str, BytesIO]:
    """
    Unified read function:
      - if path.startswith("danbooru://"), parse the ID, attempt partial download from
        the fallback repositories, return BytesIO of the content.
      - if path.startswith("s3://"), use your existing Ceph client logic.
      - else return path as a string.
    """
    if path.startswith("danbooru://"):
        # Example: path = "danbooru://7502245"
        # parse out the integer ID from the URI
        parts = path.split("://", 1)
        if len(parts) == 2:
            # If you wanted a local file fallback, you could check os.path.exists() here,
            # but that doesn't make much sense with "danbooru://".
            try:
                danbooru_id = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid danbooru ID in path: {path}")
        else:
            raise ValueError(f"Malformed danbooru:// URI: {path}")

        # Download the image data as BytesIO from fallback repos
        return download_danbooru_id(danbooru_id, repositories)

    elif path.startswith("s3://"):
        # Your original Ceph logic
        init_ceph_client_if_needed()
        from io import BytesIO
        file_bytes = BytesIO(client.get(path))
        return file_bytes

    else:
        # Just return a normal path string if it's not a special scheme
        return path

primary_repo = PrimaryRepository(
    base_url="https://huggingface.co/datasets/AngelBottomless/Danbooru2025-test/resolve/main",
    cache_dir="./cache_json_primary",
)

secondary_repo = SecondaryRepository(
    tar_base_url="https://huggingface.co/datasets/KBlueLeaf/danbooru2023-webp-4Mpixel/resolve/main/images",
    json_base_url="https://huggingface.co/datasets/deepghs/danbooru2023-webp-4Mpixel_index/resolve/main/images",
    cache_dir="./cache_json_secondary",
)

repositories = [primary_repo, secondary_repo]
########################
# Example usage
########################

if __name__ == "__main__":
    # Build a fallback chain of repositories.
    # You can have as many as you want, in the order you want.


    # Example: read from "danbooru://7502245"
    path = "danbooru://7502245"
    content = read_general(path, repositories)
    if isinstance(content, BytesIO):
        # We got image bytes, do something
        img = Image.open(content)
        print(f"Downloaded image ID=7502245, size={img.size}")
    else:
        print("Got a normal path (this won't happen in danbooru://).")
