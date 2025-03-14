import os
import json
import time
import logging
from io import BytesIO
from typing import Union, Optional, Tuple, Dict, Any, Protocol, List

import requests
from PIL import Image

# Disable Pillow’s large image pixel limit.
Image.MAX_IMAGE_PIXELS = None

#####################################################
# Configure Logging with Level Argument
#####################################################
logger = logging.getLogger(__name__)


def configure_logging(level: Union[str, int] = logging.INFO):
    """
    Configures the root logger (and thus 'logger') to a specific logging level.

    :param level: Either a string like 'DEBUG'/'INFO'/'WARNING'
                  or an integer like logging.DEBUG/logging.INFO/etc.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# Global Ceph/petrel client
client = None  # type: ignore

# Cache for JSON data loaded from a repo
loaded_jsons: Dict[str, Any] = {}

#####################################################
# Helpers for Hugging Face Token & HTTP Session
#####################################################


def _get_hf_access_token() -> str:
    """
    Retrieves the Hugging Face access token from the environment or from 'env.json'.
    Raises ValueError if not found.
    """
    hf_access_token = os.environ.get("HF_ACCESS_TOKEN")
    if not hf_access_token and os.path.isfile("env.json"):
        with open("env.json", "r", encoding="utf-8") as f:
            env_data = json.load(f)
            hf_access_token = env_data.get("HF_ACCESS_TOKEN")

    if not hf_access_token:
        raise ValueError("HF_ACCESS_TOKEN is not defined in environment or env.json.")

    return hf_access_token


def get_hf_session() -> requests.Session:
    """
    Creates and returns a requests.Session object with the Hugging Face token in the headers.
    """
    token = _get_hf_access_token()
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    return session


#####################################################
# Ceph/Petrel Client Initialization
#####################################################


def init_ceph_client_if_needed():
    """
    Initializes the global Ceph/petrel `client` if it has not yet been set.
    """
    global client
    if client is None:
        logger.info("Initializing Ceph/petrel client...")
        start_time = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("./petreloss.conf")
        end_time = time.time()
        logger.info(
            f"Initialized Ceph/petrel client in {end_time - start_time:.2f} seconds"
        )


#####################################################
# Reading & Caching JSON
#####################################################


def read_json_from_repo(
    session: requests.Session, repo_addr: str, file_name: str, cache_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Reads JSON from a given repository address and file name, with caching:
      1. If cached in memory (loaded_jsons), returns it.
      2. Otherwise, checks local disk cache (cache_dir).
      3. If not found on disk, downloads and saves it locally.

    :param session: requests.Session
    :param repo_addr: URL base (e.g. "https://github.com/user/repo/tree/main")
    :param file_name: Name of the JSON file
    :param cache_dir: Local directory to store cache
    :return: Parsed JSON object or None
    """
    unique_key = f"{repo_addr}/{file_name}"
    if unique_key in loaded_jsons:
        logger.debug(f"Found in-memory cache for {unique_key}")
        return loaded_jsons[unique_key]

    # Check local disk cache
    cache_file = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_file):
        logger.debug(f"Reading from local cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            result = json.load(f)
            loaded_jsons[unique_key] = result
            return result
    else:
        # Download and cache
        url = f"{repo_addr}/{file_name}"
        logger.debug(f"Downloading JSON from {url}")
        response = session.get(url)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            if response.status_code == 404:
                loaded_jsons[unique_key] = None
                return None
            raise
        data = response.json()
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        loaded_jsons[unique_key] = data
        return data


def load_json_index(
    session: requests.Session,
    json_url: str,
    cache_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Download (if needed) and cache a JSON file from `json_url`.
    If `cache_path` is provided, data is saved/loaded from that path.

    :param session: requests.Session
    :param json_url: Direct URL to the JSON file
    :param cache_path: Local path for caching the JSON
    :return: Parsed JSON (dict) or None if 404
    """
    if cache_path is not None and os.path.isfile(cache_path):
        logger.debug(f"Found cached JSON at {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.debug(f"Requesting JSON index from {json_url}")
    resp = session.get(json_url)
    if resp.status_code == 404:
        logger.warning(f"JSON index not found (404): {json_url}")
        return None
    resp.raise_for_status()

    data = resp.json()
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.debug(f"Saved JSON index to {cache_path}")
    return data


#####################################################
# Downloading Byte Ranges
#####################################################


def download_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    """
    Downloads the inclusive byte range [start, end] from the specified URL via
    an HTTP Range request and returns the raw bytes.

    :param session: A requests.Session with appropriate headers
    :param url: The file URL to download
    :param start: Start byte (inclusive)
    :param end: End byte (inclusive)
    :return: Raw bytes of the specified range
    """
    headers = {"Range": f"bytes={start}-{end}"}
    logger.debug(f"Downloading range {start}-{end} from {url}")
    response = session.get(url, headers=headers, stream=True)
    response.raise_for_status()
    return response.content


#####################################################
# Repository Protocol and Implementations
#####################################################


class BaseRepository(Protocol):
    """
    A Protocol that each repository must implement. Must have a method:
        find_image(session, image_id) -> (tar_url, start_offset, end_offset, filename) or None
    """

    def find_image(
        self, session: requests.Session, image_id: Union[int, str]
    ) -> Optional[Tuple[str, int, int, str]]: ...


def primary_subfolder_from_id(x: int) -> str:
    """
    Given an integer image ID, return a subfolder name based on the ID mod 1000.
    E.g., 7502245 -> '0245'.
    """
    if not isinstance(x, int):
        raise ValueError(f"Primary subfolder requires an integer ID, given: {x}")
    val = x % 1000
    return f"{val:04d}"


def secondary_chunk_from_id(x: int, chunk_size: int = 1000) -> int:
    """
    Compute the chunk index for a 'secondary' dataset given an image ID.
    """
    return x % chunk_size


class PrimaryRepository(BaseRepository):
    """
    Example of a 'primary' dataset repository:
      - .tar files named "NNNN.tar" where NNNN = image_id % 1000
      - Each .tar file has a companion JSON index "NNNN.json"
      - The JSON maps "7501000.jpg" -> [start_offset, end_offset]
    """

    def __init__(self, base_url: str, cache_dir: str, entry: Optional[str]=None):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.entry = entry
        os.makedirs(self.cache_dir, exist_ok=True)

    def _build_primary_id_map(self, json_index: Dict[str, Any]) -> Dict[int, str]:
        """
        From a JSON index like { "7501000.jpg": [start, end], ... },
        create a map of integer ID -> filename (e.g. 7501000 -> "7501000.jpg").
        """
        out = {}
        for filename in json_index.keys():
            root, _ = os.path.splitext(filename)
            try:
                num = int(root)
                out[num] = filename
            except ValueError:
                continue
        return out

    def find_image(
        self, session: requests.Session, image_id: Union[int, str]
    ) -> Optional[Tuple[str, int, int, str]]:
        if isinstance(image_id, str):
            try:
                image_id = int(image_id)
            except ValueError:
                logger.error(f"Invalid image ID: {image_id}")
                return None
        folder = primary_subfolder_from_id(image_id)
        json_name = f"{folder}.json"
        json_url = f"{self.base_url}/{json_name}"
        cache_path = os.path.join(self.cache_dir, json_name)

        logger.debug(f"Looking for image {image_id} in {json_name} (folder: {folder})")
        json_index = load_json_index(session, json_url, cache_path)
        if not json_index:
            logger.debug(f"No JSON index found for folder {folder}")
            return None

        # Build a map integer_id -> filename
        id_map = self._build_primary_id_map(json_index)
        filename = id_map.get(image_id)
        if not filename:
            logger.debug(f"Image ID {image_id} not found in index for folder {folder}")
            return None

        start_offset, end_offset = json_index[filename]
        tar_url = f"{self.base_url}/{folder}.tar"
        logger.info(
            f"Found image {image_id} in {folder}.tar ({start_offset}-{end_offset})"
        )
        return tar_url, start_offset, end_offset, filename


class SecondaryRepository(BaseRepository):
    """
    Example for a 'secondary' dataset that:
      - Uses chunk-based storage (each chunk is named data-XXXX.tar)
      - For each chunk, there's a corresponding data-XXXX.json with a "files" mapping
    """

    def __init__(
        self,
        tar_base_url: str,
        json_base_url: str,
        cache_dir: str,
        chunk_size: int = 1000,
        entry: Optional[str]=None
    ):
        self.tar_base_url = tar_base_url
        self.json_base_url = json_base_url
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.entry = entry
        os.makedirs(self.cache_dir, exist_ok=True)

    def find_image(
        self, session: requests.Session, image_id: Union[int, str]
    ) -> Optional[Tuple[str, int, int, str]]:
        if isinstance(image_id, str):
            try:
                image_id = int(image_id)
            except ValueError:
                logger.error(f"Invalid image ID: {image_id}")
                return None
        chunk_index = secondary_chunk_from_id(image_id, self.chunk_size)
        data_name = f"data-{chunk_index:04d}"

        json_url = f"{self.json_base_url}/{data_name}.json"
        cache_path = os.path.join(self.cache_dir, f"{data_name}.json")

        logger.debug(f"Looking for image {image_id} in chunk {data_name}")
        data = load_json_index(session, json_url, cache_path)
        if not data or "files" not in data:
            logger.debug(f"No file mapping found in {data_name}.json")
            return None

        filename_key = f"{image_id}.webp"
        file_dict = data["files"].get(filename_key)
        if not file_dict:
            logger.debug(f"Image ID {image_id} not found in chunk {data_name}")
            return None

        offset = file_dict["offset"]
        size = file_dict["size"]
        start_offset = offset
        end_offset = offset + size - 1  # inclusive

        tar_url = f"{self.tar_base_url}/{data_name}.tar"
        logger.info(
            f"Found image {image_id} in {data_name}.tar ({start_offset}-{end_offset})"
        )
        return tar_url, start_offset, end_offset, filename_key


class CustomRepository(BaseRepository):
    """
    Repository that relies on a single 'all_indices.json' plus a structure:
        key -> "tar_path#file_name"
        and then a nested mapping for tar_path -> file_name -> [start_offset, end_offset]
    """

    def __init__(self, base_url: str, cache_dir: str, entry: Optional[str]=None):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.entry = entry
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_range_for_key(
        self, session: requests.Session, key: Union[int, str]
    ) -> Optional[Tuple[str, int, int, str]]:
        # all_indices.json: { key: "tar_path#file_name", tar_path: {...} }
        key = str(key)
        key_index = read_json_from_repo(
            session, self.base_url, "internal_map.json", self.cache_dir
        )
        if key_index is None:
            logger.debug(f"No internal_map.json found in custom repo: {self.base_url}")
            return None
        real_key = key_index.get(key)
        if not real_key:
            logger.debug(f"Key {key} not found in custom repo index")
            return None
        repo_index = read_json_from_repo(
            session, self.base_url, "all_indices.json", self.cache_dir
        )
        if repo_index is None:
            logger.debug(f"No all_indices.json found in custom repo: {self.base_url}")
            return None
        tar_path, file_name = real_key.split("#", 1)
        if tar_path not in repo_index:
            logger.debug(f"Key {real_key} not found in custom repo index")
            return None
        tar_info = repo_index.get(tar_path, {}).get(file_name, None)
        if not tar_info or len(tar_info) < 2:
            return None

        start, end = tar_info
        tar_url = f"{self.base_url}/{tar_path}"
        logger.info(
            f"Found key '{key}' in custom repository {tar_path} ({start}-{end})"
        )
        return tar_url, start, end, file_name

    def find_image(
        self, session: requests.Session, image_id: str
    ) -> Optional[Tuple[str, int, int, str]]:
        return self.get_range_for_key(session, image_id)


#####################################################
# Repository Configuration
#####################################################

class RepositoryConfig:
    """
    Manages loading/storing repository configurations from a JSON file,
    and instantiates the corresponding repository objects, including custom 'entry' prefixes.
    """

    def __init__(self, config_path: str):
        """
        :param config_path: Path to the JSON configuration file.
        """
        self.config_path = config_path
        # Lists to hold instantiated repository objects
        self.repositories: List[BaseRepository] = []
        self.custom_repositories: List[CustomRepository] = []

        # Map from entry string -> list of repositories that handle that entry
        self.entry_map: Dict[str, List[BaseRepository]] = {}

    def load(self):
        """
        Reads the config file from disk and populates repositories and entry_map.
        """
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.debug(f"Loading repository configuration from {self.config_path}")
        print(f"Loading repository configuration from {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.from_dict(data)

    def from_dict(self, data: Dict[str, Any]):
        """
        Populates repositories/customs from a dictionary, building self.entry_map as well.

        :param data: A dict corresponding to the structure of `repository.json`.
        """
        # Clear existing repos
        self.repositories.clear()
        self.custom_repositories.clear()
        self.entry_map.clear()

        # Load standard repositories
        repos_config = data.get("repositories", [])
        for repo_dict in repos_config:
            repo_obj = self._create_repository(repo_dict)
            if repo_obj is not None:
                self.repositories.append(repo_obj)
                # If there's an "entry", register in entry_map
                entry_name = repo_dict.get("entry")
                if entry_name:
                    self.entry_map.setdefault(entry_name, []).append(repo_obj)

        # Load custom repositories
        custom_config = data.get("customs", [])
        for custom_dict in custom_config:
            custom_obj = self._create_custom_repository(custom_dict)
            if custom_obj is not None:
                self.custom_repositories.append(custom_obj)
                entry_name = custom_dict.get("entry")
                if entry_name:
                    self.entry_map.setdefault(entry_name, []).append(custom_obj)
        logger.info(
            f"Loaded {len(self.repositories)} standard repositories, "
            f"{len(self.custom_repositories)} custom repositories, "
            f"with {len(self.entry_map)} distinct entries."
        )

    def _create_repository(self, config: Dict[str, Any]) -> Optional[BaseRepository]:
        """
        Internal helper to instantiate a standard repository based on 'type'.
        """
        repo_type = config.get("type")
        entry = config.get("entry", None)  # new field

        if repo_type == "primary":
            base_url = config.get("base_url")
            cache_dir = config.get("cache_dir")
            if base_url and cache_dir:
                return PrimaryRepository(
                    base_url=base_url,
                    cache_dir=cache_dir,
                    entry=entry,  # pass to constructor
                )
            else:
                logger.warning(
                    "Invalid 'primary' repo config; missing base_url or cache_dir."
                )
                return None

        elif repo_type == "secondary":
            tar_base_url = config.get("tar_base_url")
            json_base_url = config.get("json_base_url")
            cache_dir = config.get("cache_dir")
            chunk_size = config.get("chunk_size", 1000)
            if tar_base_url and json_base_url and cache_dir:
                return SecondaryRepository(
                    tar_base_url=tar_base_url,
                    json_base_url=json_base_url,
                    cache_dir=cache_dir,
                    chunk_size=chunk_size,
                    entry=entry,
                )
            else:
                logger.warning(
                    "Invalid 'secondary' repo config; missing tar_base_url/json_base_url/cache_dir."
                )
                return None

        else:
            logger.warning(
                f"Repository type '{repo_type}' is not recognized or not supported."
            )
            return None

    def _create_custom_repository(
        self, config: Dict[str, Any]
    ) -> Optional[CustomRepository]:
        """
        Internal helper to instantiate a custom repository.
        """
        repo_type = config.get("type")
        entry = config.get("entry", None)

        if repo_type == "custom":
            base_url = config.get("base_url")
            cache_dir = config.get("cache_dir")
            if base_url and cache_dir:
                return CustomRepository(
                    base_url=base_url, cache_dir=cache_dir, entry=entry
                )
            else:
                logger.warning(
                    "Invalid 'custom' repo config; missing base_url or cache_dir."
                )
                return None

        else:
            logger.warning(
                f"Custom repository type '{repo_type}' is not recognized or not supported."
            )
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Reconstructs the config dictionary from the current repository objects.
        """
        return {
            "repositories": [self._repo_to_dict(repo) for repo in self.repositories],
            "customs": [
                self._custom_repo_to_dict(crepo) for crepo in self.custom_repositories
            ],
        }

    def _repo_to_dict(self, repo: BaseRepository) -> Dict[str, Any]:
        """
        Rebuilds the config dict for a standard repository from its attributes.
        """
        # We assume each repository has .entry
        if hasattr(repo, "entry"):
            entry_val = getattr(repo, "entry", None)
        else:
            entry_val = None

        if isinstance(repo, PrimaryRepository):
            return {
                "type": "primary",
                "base_url": repo.base_url,
                "cache_dir": repo.cache_dir,
                "entry": entry_val,
            }
        elif isinstance(repo, SecondaryRepository):
            return {
                "type": "secondary",
                "tar_base_url": repo.tar_base_url,
                "json_base_url": repo.json_base_url,
                "cache_dir": repo.cache_dir,
                "chunk_size": repo.chunk_size,
                "entry": entry_val,
            }
        else:
            return {"type": "unknown", "entry": entry_val}

    def _custom_repo_to_dict(self, repo: CustomRepository) -> Dict[str, Any]:
        """
        Rebuilds the config dict for a CustomRepository from its attributes.
        """
        return {
            "type": "custom",
            "base_url": repo.base_url,
            "cache_dir": repo.cache_dir,
            "entry": getattr(repo, "entry", None),
        }

    def save(self, path: Optional[str] = None):
        """
        Saves the current config (based on the instantiated repo objects) back to a JSON file.
        :param path: Optional; if None, uses self.config_path.
        """
        if path is None:
            path = self.config_path

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Repository configuration saved to {path}")

    def get_repositories_for_entry(self, entry: str) -> List[Union[BaseRepository, CustomRepository]]:
        """
        Retrieves the list of repositories (both standard and custom) that are mapped to a given entry prefix.
        """
        return self.entry_map.get(entry, [])

    def search_entry_and_key(self, entry: str, key: str) -> Optional[BytesIO]:
        """
        Returns a RepositoryPool object that can be used to download images for a given entry.
        """
        repositories = self.get_repositories_for_entry(entry)
        if not repositories:
            logger.warning(f"No repositories found for entry: {entry}")
            return None
        base_repos = BaseRepositoryPool(repositories)
        result = base_repos.download_by_id(key)
        if result:
            return result
        return None


#####################################################
class RepositoryPool(Protocol):
    """
    A Protocol for a set of repositories that can be searched for a given image ID.
    """
    ### class to hold download_by_id method
    def download_by_id(self, image_id: int) -> Optional[BytesIO]: ...


class BaseRepositoryPool(RepositoryPool):
    """
    A pool of BaseRepository objects, allowing for a unified download_by_id method.
    """

    def __init__(self, repositories: List[BaseRepository]):
        self.repositories = repositories
    ### class to hold download_by_id method
    def download_by_id(self, image_id: int) -> Optional[BytesIO]:
        session = get_hf_session()
        for repo in self.repositories:
            info = repo.find_image(session, image_id)
            logger.debug(f"Searching for image {image_id} in {repo}, result: {info}")
            if info:
                break
        if not info:
            msg = f"Image ID {image_id} was not found in any repository. (Base)"
            logger.info(msg)
            return None
        tar_url, start_offset, end_offset, _ = info
        file_bytes = download_range(session, tar_url, start_offset, end_offset)
        logger.info(f"Successfully downloaded image {image_id} from {tar_url}")
        return BytesIO(file_bytes)


#####################################################
# Universal Read Function
#####################################################
REPOSITORY_CONFIG: RepositoryConfig = RepositoryConfig(r"repository.json")
REPOSITORY_CONFIG.load()

def read_general(path: str) -> Union[str, BytesIO]:
    """
    A universal read function:
      - If path starts with "danbooru://", parse out the integer ID and download
        from configured repositories. Returns a BytesIO of the file content.
      - If path starts with "s3://", uses Ceph/petrel client to retrieve data.
      - Otherwise, if the path doesn't exist locally, tries custom repositories.
      - If none of the above, returns the path string as-is (assuming it's local or standard).

    :param path: The path or URI to read
    :return: Either a local path string or an in-memory BytesIO
    """
    config = REPOSITORY_CONFIG
    if path.startswith("s3://"):
        init_ceph_client_if_needed()
        logger.debug(f"Downloading from Ceph/petrel: {path}")
        file_data = client.get(path)  # type: ignore
        return BytesIO(file_data)
    if "://" in path:
        parts = path.split("://", 1)
        entry = parts[0]
        result = config.search_entry_and_key(entry, parts[1])
        if result:
            return result
        raise FileNotFoundError(f"Image ID not found in any repository: {path}")
    # If the path isn't local, try custom repositories
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Otherwise, assume it's a normal local path
    logger.debug(f"Returning local path: {path}")
    return path


if __name__ == "__main__":
    # 2) Configure logging at the desired level
    configure_logging("DEBUG")  # or "INFO", "WARNING", etc.

    # 3) Example usage:
    # try:
    #     data = read_general("danbooru://6706939")
    #     if isinstance(data, BytesIO):
    #         img = Image.open(data)
    #         img.show()
    # except FileNotFoundError as e:
    #     logger.error(str(e))
    # try:
    #     data = read_general("danbooru://8884993")
    #     if isinstance(data, BytesIO):
    #         img = Image.open(data)
    #         img.show()
    # except FileNotFoundError as e:
    #     logger.error(str(e))
    #
    try:
        data = read_general("anime://fancaps/8183457")
        if isinstance(data, BytesIO):
            img = Image.open(data)
            img.show()
    except FileNotFoundError as e:
        logger.error(str(e))
    # Other usage examples:
    # data2 = read_general("s3://bucket_name/path/to/object.jpg")
    # data3 = read_general("some/local/path.jpg")
