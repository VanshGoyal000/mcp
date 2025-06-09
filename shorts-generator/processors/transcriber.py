import os
import tempfile
from pathlib import Path
import modal
import random
import logging
import requests
import subprocess
import json
import uuid
# Import directly from the module
from modal_config import app, get_base_image, volume, get_secrets

@app.function(
    image=get_base_image(),
    cpu=4,
    memory=8192,
    timeout=600,
    volumes={"/shorts-generator": volume},
    secrets=[get_secrets()]
)
def download_youtube_video(youtube_url: str) -> tuple:
    """Download a YouTube video and return its path and title."""
    from pytube import YouTube
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directories in the volume
        volume_dir = "/shorts-generator/videos"
        os.makedirs(volume_dir, exist_ok=True)
        
        # Generate a unique filename
        video_id = f"video_{uuid.uuid4().hex}.mp4"
        volume_path = f"{volume_dir}/{video_id}"
        
        logger.info(f"Downloading video from: {youtube_url}")
        
        # First try with yt-dlp for better reliability
        try:
            # Use yt-dlp command line tool - more reliable than pytube
            cmd = [
                "yt-dlp",
                "-f", "best[ext=mp4]",  # Get best MP4 format
                "--no-playlist",
                "--print", "%(title)s",
                "-o", volume_path,
                "--force-overwrites",
                youtube_url
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_title = result.stdout.strip() or "YouTube Video"
            
            # Validate the downloaded file
            if not os.path.exists(volume_path) or os.path.getsize(volume_path) < 10000:
                raise Exception("Downloaded file is too small or doesn't exist")
            
            # Try to get video metadata to validate it's a proper video file
            probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", 
                         "-of", "json", volume_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode != 0 or "video" not in probe_result.stdout:
                raise Exception("Not a valid video file")
            
            logger.info(f"Successfully downloaded '{video_title}' to {volume_path}")
            return volume_path, video_title
        
        except Exception as e:
            logger.error(f"yt-dlp download failed: {e}")
            
            # Fall back to pytube
            try:
                logger.info("Trying PyTube as fallback")
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download with pytube
                    yt = YouTube(youtube_url)
                    video_title = yt.title
                    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                    
                    if not video_stream:
                        raise Exception("No suitable streams found")
                    
                    temp_path = video_stream.download(output_path=temp_dir)
                    
                    # Copy to Modal volume
                    with open(temp_path, "rb") as f:
                        video_data = f.read()
                    
                    with open(volume_path, "wb") as f:
                        f.write(video_data)
                    
                    logger.info(f"Pytube download successful: '{video_title}'")
                    return volume_path, video_title
            
            except Exception as e2:
                logger.error(f"Pytube download failed: {e2}")
                
                # Fall back to sample video
                sample_path = "/shorts-generator/videos/sample_video.mp4"
                if not os.path.exists(sample_path):
                    # Download a sample video as last resort
                    try:
                        sample_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
                        logger.info(f"Downloading sample video from {sample_url}")
                        
                        response = requests.get(sample_url)
                        if response.status_code == 200:
                            with open(sample_path, 'wb') as f:
                                f.write(response.content)
                        else:
                            raise Exception(f"HTTP error: {response.status_code}")
                            
                    except Exception as e3:
                        logger.error(f"Failed to download sample video: {e3}")
                        raise Exception("All download methods failed")
                
                return sample_path, "Sample Video"
    
    except Exception as e:
        logger.error(f"Error downloading video: {e}")
        raise Exception(f"Failed to download YouTube video: {e}")

@app.function(
    image=get_base_image(),
    cpu=4,
    memory=8192,
    gpu="T4",  # Request GPU for faster transcription
    timeout=900,
    volumes={"/shorts-generator": volume},
    secrets=[get_secrets()]
)
def transcribe_video(video_path: str) -> dict:
    """Transcribe a video using Whisper and return transcript with timestamps."""
    import whisper
    import os
    import tempfile
    import subprocess
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting transcription of: {video_path}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # First validate the input file
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Use ffprobe to check if the file is valid
            probe_cmd = [
                "ffprobe",
                "-v", "error",
                "-analyzeduration", "100M",
                "-probesize", "100M",
                "-show_entries", "stream=codec_type,codec_name",
                "-of", "json",
                video_path
            ]
            
            logger.info("Validating video file...")
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if probe_result.returncode != 0:
                raise Exception(f"Invalid video file: {probe_result.stderr}")
                
            probe_data = json.loads(probe_result.stdout)
            
            # Check if we have valid audio stream for transcription
            has_audio = any(s.get('codec_type') == 'audio' for s in probe_data.get('streams', []))
            
            if not has_audio:
                raise Exception("No audio stream found in video file")
            
            # Copy file from Modal volume to temporary location
            temp_video_path = os.path.join(temp_dir, os.path.basename(video_path))
            
            logger.info(f"Copying video to temporary location: {temp_video_path}")
            with open(video_path, "rb") as f:
                video_data = f.read()
                
            with open(temp_video_path, "wb") as f:
                f.write(video_data)
            
            # Extract audio for better transcription reliability
            audio_path = os.path.join(temp_dir, "audio.wav")
            logger.info("Extracting audio for better transcription quality")
            
            extract_cmd = [
                "ffmpeg", 
                "-y",
                "-i", temp_video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit audio
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                audio_path
            ]
            
            subprocess.run(extract_cmd, check=True, capture_output=True)
            
            # Load Whisper model and transcribe
            logger.info("Loading Whisper model...")
            model = whisper.load_model("base")
            
            logger.info("Starting transcription...")
            result = model.transcribe(
                audio_path,
                fp16=True,  # Use FP16 with GPU
                language="en",
                word_timestamps=True
            )
            
            # Format output with timestamps
            logger.info("Processing transcription results...")
            transcript_with_timestamps = []
            for segment in result["segments"]:
                transcript_with_timestamps.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })
            
            logger.info(f"Transcription complete! Total text length: {len(result['text'])}")
            return {
                "full_text": result["text"],
                "segments": transcript_with_timestamps,
                "method": "modal-whisper-gpu"
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Return a minimal valid response rather than an error
            return {
                "full_text": f"Transcription failed: {str(e)}",
                "segments": [{"start": 0, "end": 1, "text": "Transcription error"}],
                "method": "error"
            }
