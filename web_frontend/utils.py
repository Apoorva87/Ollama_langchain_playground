import os

def load_css(css_file_path: str) -> str:
    """Load CSS content from a file.
    
    Args:
        css_file_path: Path to the CSS file
        
    Returns:
        CSS content as a string
    """
    try:
        with open(css_file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Warning: CSS file not found at {css_file_path}")
        return ""
    except Exception as e:
        print(f"Error loading CSS file: {e}")
        return "" 