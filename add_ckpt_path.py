import sys
import os
import os.path as path


def add_path_to_dust3r(ckpt):
    """Ensure the checkpoint folder and repo src folder are discoverable."""
    here_path = os.path.dirname(os.path.abspath(ckpt))
    repo_root = path.dirname(path.abspath(__file__))
    src_path = path.join(repo_root, "src")

    for candidate in (here_path, repo_root, src_path):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)
