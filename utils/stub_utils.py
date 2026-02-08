"""
Simple pickle-based caching ("stub") utilities.

Heavy detection and tracking steps can be cached to disk so that subsequent
runs skip the expensive inference and read pre-computed results instead.
"""

import pickle
import os


def read_stub(read_from_stub, stub_path):
    """Attempt to load a cached result from a pickle file.

    Parameters
    ----------
    read_from_stub : bool
        If ``False`` the function returns ``None`` immediately.
    stub_path : str | None
        Path to the pickle file.  ``None`` is treated as "no cache".

    Returns
    -------
    object | None
        The unpickled object, or ``None`` if the cache was not read.
    """
    if not read_from_stub:
        return None
    if stub_path is None or not os.path.exists(stub_path):
        return None

    with open(stub_path, "rb") as f:
        return pickle.load(f)


def save_stub(stub_path, data):
    """Persist an object to a pickle file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    stub_path : str | None
        Destination path.  If ``None`` the call is silently skipped.
    data : object
        Any picklable Python object.
    """
    if stub_path is None:
        return

    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(data, f)
