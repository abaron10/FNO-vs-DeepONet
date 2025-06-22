import sys
import os

def setup_library_paths():
    """Add local libraries to Python path"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    neuralop_path = os.path.join(project_root, 'libs', 'neuraloperator')
    
    if neuralop_path not in sys.path:
        sys.path.insert(0, neuralop_path)

# Call this at import
setup_library_paths()