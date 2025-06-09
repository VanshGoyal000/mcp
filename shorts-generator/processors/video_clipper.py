import os
import tempfile
from pathlib import Path
import modal
# Update imports to be direct
from modal_config import app, get_base_image, volume, get_secrets

@app.function(
    image=get_base_image(),
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/shorts-generator": volume},
    secrets=[get_secrets()]
)
def clip_video(video_path: str, highlights: list) -> list:
    """
    Clip segments from a video based on highlight timestamps.
    Returns a list of paths to the clipped videos stored in Modal.
    """
    import ffmpeg
    import os
    import tempfile
    
    clip_paths = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Copy file from Modal volume to temporary location
            temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
            
            with open(video_path, "rb") as f:
                video_data = f.read()
                
            with open(temp_video_path, "wb") as f:
                f.write(video_data)
            
            # Process each highlight
            for i, highlight in enumerate(highlights):
                start_time = highlight["start_time"]
                end_time = highlight["end_time"]
                title = highlight["title"].replace(" ", "_")
                
                # Create a sanitized filename
                output_filename = f"clip_{i}_{title[:30]}.mp4"
                output_path = os.path.join(temp_dir, output_filename)
                
                # Use ffmpeg to clip the video
                (
                    ffmpeg
                    .input(temp_video_path, ss=start_time, to=end_time)
                    .output(output_path, codec='copy')
                    .run()
                )
                
                # Store in Modal volume for persistence
                clip_volume_path = f"/shorts-generator/clips/{output_filename}"
                os.makedirs(os.path.dirname(clip_volume_path), exist_ok=True)
                
                with open(output_path, "rb") as f:
                    clip_data = f.read()
                
                with open(clip_volume_path, "wb") as f:
                    f.write(clip_data)
                
                clip_paths.append({
                    "path": clip_volume_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": highlight["title"],
                    "description": highlight["description"]
                })
                
            return clip_paths
            
        except Exception as e:
            raise Exception(f"Video clipping failed: {e}")
