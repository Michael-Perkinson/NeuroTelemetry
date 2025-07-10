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


def create_folders_for_graphs(data_file_path: str) -> Tuple[str, str, str, str]:
    """
    Creates and returns output directories for saving plots and images.

    Returns:
        html_save_folder, svg_save_folder, full_trace_folder, file_base
    """
    base_folder = os.path.splitext(data_file_path)[0]
    file_base = os.path.basename(base_folder)

    html_save_folder = os.path.join(base_folder, "html")
    svg_save_folder = os.path.join(base_folder, "svg")
    full_trace_folder = os.path.join(base_folder, "full_trace")

    os.makedirs(html_save_folder, exist_ok=True)
    os.makedirs(svg_save_folder, exist_ok=True)
    os.makedirs(full_trace_folder, exist_ok=True)

    return html_save_folder, svg_save_folder, full_trace_folder, file_base
