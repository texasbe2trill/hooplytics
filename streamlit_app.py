"""Entry shim for Streamlit Community Cloud — points at the package app."""
import pathlib
import sys

# Ensure the live repo source takes priority over any stale installed package.
# Streamlit Cloud mounts the repo at /mount/src/<repo>; inserting its root
# means `import hooplytics` resolves here before site-packages.
_repo_root = str(pathlib.Path(__file__).parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hooplytics.web import app  # noqa: F401  (executes the Streamlit script)
