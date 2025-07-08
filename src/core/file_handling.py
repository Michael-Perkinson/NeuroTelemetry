from typing import Union, List
from pathlib import Path


def list_files(directory: Path, extensions: Union[str, List[str]] = 'csv') -> str:
    """Return a string listing all files with given extension(s) in a directory."""
    if not directory.exists():
        return f"Directory does not exist: {directory}"

    if isinstance(extensions, str):
        extensions = [extensions]

    output = [f"Files in {directory}:"]
    for ext in extensions:
        for file in directory.glob(f'*.{ext}'):
            output.append(str(file.name))

    return "\n".join(output) if len(output) > 1 else f"No *.{', *.'.join(extensions)} files found in {directory}"
