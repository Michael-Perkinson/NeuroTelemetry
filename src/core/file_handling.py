import os
from typing import Tuple
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


def create_folders_for_graphs(data_file_path: Path) -> Tuple[Path, Path, Path, str]:
    """
    Creates and returns output directories for saving plots and images.
    """
    base_folder = data_file_path.with_suffix('')
    file_base = base_folder.name

    html_save_folder = base_folder / "html"
    svg_save_folder = base_folder / "svg"
    full_trace_folder = base_folder / "full_trace"

    html_save_folder.mkdir(parents=True, exist_ok=True)
    svg_save_folder.mkdir(parents=True, exist_ok=True)
    full_trace_folder.mkdir(parents=True, exist_ok=True)

    return html_save_folder, svg_save_folder, full_trace_folder, file_base
