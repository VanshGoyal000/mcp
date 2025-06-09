import os
import json
import logging

logger = logging.getLogger(__name__)

def debug_print(obj, label=None):
    """Print an object in a formatted way for debugging"""
    if label:
        print(f"\n=== DEBUG: {label} ===")
    
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, indent=2, default=str))
    else:
        print(obj)
    
    if label:
        print(f"=== END {label} ===\n")

def examine_file(filepath):
    """Examine a file and print diagnostic information"""
    print(f"\n=== EXAMINING FILE: {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File does not exist: {filepath}")
        return False
    
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
    
    if file_size == 0:
        print("ERROR: File is empty!")
        return False
    
    # Try to determine file type
    import mimetypes
    mime_type = mimetypes.guess_type(filepath)[0]
    print(f"MIME type: {mime_type}")
    
    # Check file extension
    _, ext = os.path.splitext(filepath)
    print(f"File extension: {ext}")
    
    # If it's a video file, try to get more info using ffprobe
    if mime_type and mime_type.startswith('video'):
        try:
            import subprocess
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                   '-of', 'default=noprint_wrappers=1:nokey=1', filepath]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stdout.strip():
                duration = float(result.stdout.strip())
                print(f"Video duration: {duration:.2f} seconds")
            
            # Get resolution
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                   '-show_entries', 'stream=width,height', 
                   '-of', 'csv=s=x:p=0', filepath]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.stdout.strip():
                print(f"Resolution: {result.stdout.strip()}")
        except Exception as e:
            print(f"Failed to get video info: {str(e)}")
    
    print(f"=== END EXAMINATION OF: {filepath} ===\n")
    return True
