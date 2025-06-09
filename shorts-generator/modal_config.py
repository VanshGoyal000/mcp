import modal
import os
import importlib.util
import sys

# Define your Modal App 
app = modal.App("shorts-generator")

# Define volume for persistent storage
volume = modal.Volume.from_name("shorts-generator-vol", create_if_missing=True)

# Define secret for use across functions
def get_secrets():
    try:
        return modal.Secret.from_name("shorts-generator-secrets")
    except:
        # Create secret on the fly if needed
        env_vars = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY", ""),
            "YOUTUBE_API_KEY": os.environ.get("YOUTUBE_API_KEY", ""),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        }
        return modal.Secret.from_dict("shorts-generator-secrets", env_vars)

# Define common image for reuse
def get_base_image():
    return (
        modal.Image.debian_slim()
        .apt_install(["ffmpeg", "libsm6", "libxext6"])
        .pip_install([
            "ffmpeg-python",
            "openai",
            "openai-whisper", 
            "pytube",
            "langchain",
            "python-dotenv",
            "google-generativeai>=0.3.0"
        ])
    )

# Direct access to deployed Modal functions
def get_modal_functions():
    """Access deployed functions using Modal client API"""
    try:
        # Create a Modal client stub for accessing deployed functions
        stub = modal.Stub("shorts-generator")
        
        # Check if function exists before trying to access it
        available_functions = [f for f in dir(stub.app) if not f.startswith('_')]
        print(f"Available functions in Modal: {available_functions}")
        
        # Create direct references to deployed functions
        if "download_youtube_video" in available_functions:
            app.download_youtube_video = stub.app.download_youtube_video
        
        if "transcribe_video_enhanced" in available_functions:
            app.transcribe_video_enhanced = stub.app.transcribe_video_enhanced
            
        if "smart_highlight_selector" in available_functions:
            app.smart_highlight_selector = stub.app.smart_highlight_selector
            
        if "create_smart_clips" in available_functions:
            app.create_smart_clips = stub.app.create_smart_clips
            
        if "generate_caption" in available_functions:
            app.generate_caption = stub.app.generate_caption
            
        if "clip_video" in available_functions:
            app.clip_video = stub.app.clip_video
            
        if "select_highlights" in available_functions:
            app.select_highlights = stub.app.select_highlights
        
        print("Successfully attached functions from deployed Modal app")
        return True
    except Exception as e:
        print(f"Warning: Could not access deployed Modal functions: {str(e)}")
        print("Make sure to run 'python modal_deploy.py' first to deploy the functions.")
        return False

# Try to get Modal functions when this module is imported
get_modal_functions()