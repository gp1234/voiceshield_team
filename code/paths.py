from pathlib import Path
from typing import List, Union, Optional


CODE_DIR = Path(__file__).parent.absolute()


MODELS_DIR = CODE_DIR / 'models_testing'
NOTEBOOKS_DIR = CODE_DIR / 'notebooks'
ASSETS_DIR = CODE_DIR / 'assets'

def get_all_files_in_code(extension: Optional[str] = None) -> List[Path]:
    """
    Get all files in the code directory and its subdirectories.
    
    Args:
        extension: Optional file extension to filter by (e.g., '.py', '.ipynb')
    
    Returns:
        List of Path objects for all matching files
    """
    if extension:
        return list(CODE_DIR.rglob(f'*{extension}'))
    return list(CODE_DIR.rglob('*'))

def get_files_in_directory(directory: Union[str, Path], extension: Optional[str] = None) -> List[Path]:
    """
    Get all files in a specific directory.
    
    Args:
        directory: Directory name or Path object
        extension: Optional file extension to filter by
    
    Returns:
        List of Path objects for all matching files
    """
    if isinstance(directory, str):
        directory = CODE_DIR / directory
    
    if extension:
        return list(directory.glob(f'*{extension}'))
    return list(directory.glob('*'))

# Example usage:
# Get all Python files in code directory
# python_files = get_all_files_in_code('.py')
# Get all files in models_testing directory
# model_files = get_files_in_directory('models_testing')
