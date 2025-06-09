import modal
import os
import sys
import random
from dotenv import load_dotenv

# Add current directory to the Python path to enable direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Load environment variables first
load_dotenv()

# Create a completely fresh Modal app
app = modal.App("shorts-generator")

# Create volume for persistent storage
volume = modal.Volume.from_name("shorts-generator-vol", create_if_missing=True)

# Define base image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["ffmpeg", "libsm6", "libxext6"])
    .pip_install([
        "ffmpeg-python",
        "openai>=1.0.0",
        "openai-whisper>=20231117", 
        "pytube>=15.0.0",
        "yt-dlp>=2023.3.4",
        "langchain>=0.1.0",
        "python-dotenv>=1.0.0"
    ])
)

def setup_modal_secrets():
    """Set up Modal secrets using environment variables"""
    try:
        # Create environment dictionary
        secret_env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY", ""),
            "YOUTUBE_API_KEY": os.environ.get("YOUTUBE_API_KEY", ""),
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", ""),
        }
        
        # Create the secret with proper API usage
        secret = modal.Secret(secret_env)
        secret.save("shorts-generator-secrets")
        print("Modal secrets configured successfully")
    except Exception as e:
        print(f"Error setting up Modal secrets: {e}")

# Create a setup_directories function
@app.function(image=image, volumes={"/data": volume})
def setup_directories():
    import os
    os.makedirs("/data/videos", exist_ok=True)
    os.makedirs("/data/clips", exist_ok=True)
    print("Directories created in Modal volume")
    return True

# Define the download_youtube_video function at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def download_youtube_video(youtube_url):
    import os
    import uuid
    import yt_dlp
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create output path
        video_id = f"video_{uuid.uuid4().hex}.mp4"
        output_path = f"/data/videos/{video_id}"
        
        # Download with yt-dlp
        logger.info(f"Downloading video from: {youtube_url}")
        ydl_opts = {
            'format': 'mp4',
            'outtmpl': output_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            title = info.get('title', 'Unknown')
            
        logger.info(f"Downloaded: {title}")
        return output_path, title
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return None, str(e)

# Define transcribe_video_enhanced at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",  # Request GPU for faster transcription
    timeout=900,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def transcribe_video_enhanced(video_path_or_url):
    """Enhanced video transcription with better error handling and validation"""
    import os
    import tempfile
    import whisper
    import subprocess
    import json
    import logging
    import requests
    import uuid
    import yt_dlp
    import shutil
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing transcription request for: {video_path_or_url}")
    
    # Function to repair a corrupted video file
    def repair_video(path):
        """Attempts to repair a corrupted video file"""
        logger.info(f"Attempting to repair video file: {path}")
        
        if not os.path.exists(path):
            return False, "File not found"
            
        try:
            # Create temporary directory for repair
            repair_dir = tempfile.mkdtemp(dir="/data/tmp")
            repaired_path = os.path.join(repair_dir, f"repaired_{os.path.basename(path)}")
            
            # Try to repair with ffmpeg by re-encoding
            cmd = [
                "ffmpeg",
                "-y",
                "-err_detect", "ignore_err",
                "-analyzeduration", "100M",
                "-probesize", "100M",
                "-i", path,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-c:a", "aac",
                repaired_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Repair failed: {result.stderr}")
                return False, f"Repair failed: {result.stderr[:100]}..."
            
            # Check if repaired file is valid
            valid, msg = validate_video(repaired_path)
            if valid:
                return True, repaired_path
            else:
                return False, f"Repaired file still invalid: {msg}"
            
        except Exception as e:
            logger.error(f"Error during repair: {str(e)}")
            return False, f"Repair error: {str(e)}"
            
    # Function to validate a video file
    def validate_video(path):
        if not os.path.exists(path):
            return False, "File not found"
            
        # Check file size first
        file_size = os.path.getsize(path)
        if file_size < 10000:  # Less than 10KB
            return False, f"File too small: {file_size} bytes"
            
        # Use ffprobe with increased analyzeduration and probesize
        cmd = [
            "ffprobe",
            "-v", "error",
            "-analyzeduration", "100M",
            "-probesize", "100M",
            "-show_entries", "stream=codec_type,codec_name,width,height,pix_fmt",
            "-of", "json",
            path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                
                # Check if we have valid video streams
                for stream in streams:
                    if stream.get('codec_type') == 'video':
                        # Check if pixel format is specified
                        if not stream.get('pix_fmt'):
                            return False, "Video stream has no pixel format"
                        
                        # Check dimensions
                        if not stream.get('width') or not stream.get('height'):
                            return False, "Video stream has no dimensions"
                            
                        # If we got here, the video stream seems valid
                        return True, "Video validated"
                
                # No valid video stream found
                return False, "No valid video stream found"
            else:
                return False, f"FFprobe error: {result.stderr[:100]}..."
        except Exception as e:
            return False, f"Validation error: {str(e)}"
            
    # Function to extract audio from video
    def extract_audio(video_path, output_dir):
        """Extract audio from video file for transcription"""
        audio_path = os.path.join(output_dir, "audio.wav")
        
        # Try two different approaches for extraction
        try:
            # First attempt: standard extraction
            cmd = [
                "ffmpeg", 
                "-y",
                "-analyzeduration", "100M",
                "-probesize", "100M",
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit audio
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                return True, audio_path
            
            # Second attempt: copy stream directly
            logger.info("First audio extraction failed, trying alternate method")
            cmd = [
                "ffmpeg", 
                "-y",
                "-analyzeduration", "100M",
                "-probesize", "100M",
                "-i", video_path,
                "-vn",
                "-acodec", "copy",
                os.path.join(output_dir, "audio_copy.aac")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Convert the copied audio to WAV
            if result.returncode == 0:
                cmd = [
                    "ffmpeg", 
                    "-y",
                    "-i", os.path.join(output_dir, "audio_copy.aac"),
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    audio_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    return True, audio_path
            
            # If both methods failed, return failure
            return False, f"Audio extraction failed: {result.stderr[:100]}..."
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            return False, f"Audio extraction error: {str(e)}"
    
    # Handle URL or local path differently
    video_path = video_path_or_url
    temp_dir = None
    proc_temp_dir = None
    
    try:
        # Create temp directories
        temp_dir = tempfile.mkdtemp(dir="/data/tmp")
        proc_temp_dir = tempfile.mkdtemp(dir="/data/tmp")
        
        # If it's a URL, download it first
        if isinstance(video_path_or_url, str) and video_path_or_url.startswith(('http://', 'https://')):
            logger.info("Input is a URL, downloading first...")
            try:
                output_path = os.path.join(temp_dir, f"video_{uuid.uuid4().hex}.mp4")
                
                # Try yt-dlp first (better for YouTube)
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': output_path,
                    'quiet': False,
                    'no_warnings': False
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_path_or_url])
                    
                video_path = output_path
                logger.info(f"Downloaded video to {video_path}")
                
            except Exception as e:
                logger.error(f"Error downloading video: {str(e)}")
                raise Exception(f"Failed to download video: {str(e)}")
        
        # Validate the video file
        logger.info(f"Validating video: {video_path}")
        is_valid, message = validate_video(video_path)
        
        # If invalid, try to repair
        if not is_valid:
            logger.warning(f"Invalid video file: {message}. Attempting repair...")
            repair_success, repair_result = repair_video(video_path)
            
            if repair_success:
                logger.info("Video repaired successfully!")
                video_path = repair_result
            else:
                logger.error(f"Video repair failed: {repair_result}")
                raise Exception(f"Invalid video file and repair failed: {repair_result}")
            
        # Create a temporary copy for whisper processing
        working_video_path = os.path.join(proc_temp_dir, os.path.basename(video_path))
        
        logger.info(f"Creating working copy at {working_video_path}")
        shutil.copy2(video_path, working_video_path)
        
        # Extract audio to improve transcription reliability
        logger.info("Extracting audio for better transcription...")
        audio_success, audio_path = extract_audio(working_video_path, proc_temp_dir)
        
        if not audio_success:
            logger.error(f"Failed to extract audio: {audio_path}")
            raise Exception(f"Failed to extract audio: {audio_path}")
            
        logger.info(f"Audio extracted to {audio_path}")
        
        # Load Whisper model with GPU acceleration
        logger.info("Loading Whisper model...")
        try:
            # Try faster tiny model first
            model = whisper.load_model("tiny")
            logger.info("Using tiny Whisper model for initial pass")
        except Exception as e:
            logger.warning(f"Error loading tiny model: {str(e)}, trying base")
            model = whisper.load_model("base")
        
        # Transcribe the audio
        logger.info("Starting transcription...")
        result = model.transcribe(
            audio_path,
            fp16=True,  # Use FP16 for GPU acceleration
            language="en",  # Specify language if known
            word_timestamps=True
        )
        
        # Check if we got a good result
        if not result.get('text') or len(result.get('text', '')) < 10:
            logger.warning("Initial transcription returned little or no text, trying with base model")
            try:
                # Try using base model for better quality
                model = whisper.load_model("base")
                result = model.transcribe(
                    audio_path,
                    fp16=True,
                    language="en",
                    word_timestamps=True
                )
            except Exception as e:
                logger.error(f"Error in second transcription attempt: {str(e)}")
        
        # Format output with timestamps
        transcript_with_timestamps = []
        for segment in result["segments"]:
            transcript_with_timestamps.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        # Clean up temp files
        try:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if proc_temp_dir:
                shutil.rmtree(proc_temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up some temporary files: {str(e)}")
        
        # Return structured result
        transcript_result = {
            "full_text": result["text"],
            "segments": transcript_with_timestamps,
            "method": "modal-whisper-gpu"
        }
        
        logger.info(f"Transcription completed successfully. Text length: {len(result['text'])}")
        return transcript_result
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        # Clean up temp files in case of error
        try:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if proc_temp_dir:
                shutil.rmtree(proc_temp_dir, ignore_errors=True)
        except:
            pass
            
        # Return a minimal valid response rather than fail completely
        return {
            "full_text": f"Transcription failed: {str(e)}",
            "segments": [{"start": 0, "end": 5, "text": "Transcription failed"}],
            "method": "modal-error"
        }

# Add a video validation and repair function at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def validate_repair_video(video_path):
    """Validate a video file and repair/re-download if needed"""
    import os
    import subprocess
    import json
    import logging
    import yt_dlp
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating video: {video_path}")
    
    # Check if file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False, "File not found"
    
    # Check if video is valid using ffprobe with increased analyzeduration and probesize
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-analyzeduration", "100M",
        "-probesize", "100M",
        "-show_entries",
        "stream=codec_type,codec_name,width,height",
        "-of", "json",
        video_path
    ]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get('streams', [])
            
            # Check if we have video streams with dimensions
            valid = any(s.get('codec_type') == 'video' and 
                      s.get('width') is not None and
                      s.get('height') is not None
                      for s in streams)
            
            if valid:
                logger.info("Video file is valid")
                return True, "Video is valid"
            else:
                logger.warning("Video has no valid video streams")
        else:
            logger.warning(f"FFprobe validation failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error validating video: {str(e)}")
    
    logger.warning("Video file is corrupt or invalid, cleaning cache...")
    
    # Try to remove the invalid file
    try:
        os.remove(video_path)
        logger.info(f"Removed invalid file: {video_path}")
    except Exception as e:
        logger.error(f"Failed to remove file: {str(e)}")
    
    return False, "Video is invalid"

# Helper function for generic highlights
def _generate_generic_highlights(num_highlights, duration=60):
    """Generate generic highlights when all else fails"""
    import random
    
    highlights = []
    for i in range(num_highlights):
        start_time = i * 90  # Space out every 90 seconds
        highlights.append({
            "start_time": start_time,
            "end_time": start_time + duration,
            "title": f"Highlight {i+1}",
            "description": f"Auto-selected highlight starting at {int(start_time//60)}:{int(start_time%60):02d}"
        })
    return highlights

# Add a new smart clip generator function at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def create_smart_clips(video_path, transcript_data, min_duration=20, max_duration=60, 
                       target_clips=3):
    """
    Creates variable-length clips based on content relevance rather than fixed duration.
    
    Args:
        video_path: Path to the video file
        transcript_data: Transcript with timestamps
        min_duration: Minimum clip duration in seconds (default: 20)
        max_duration: Maximum clip duration in seconds (default: 60)
        target_clips: Number of clips to generate (default: 3)
        
    Returns:
        List of clip info dictionaries with paths and metadata
    """
    import os
    import subprocess
    import json
    import logging
    import uuid
    import tempfile
    from math import ceil
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating smart variable-length clips from: {video_path}")
    logger.info(f"Parameters: min={min_duration}s, max={max_duration}s, target={target_clips} clips")
    
    def find_natural_segments(transcript_data):
        """Find natural break points in the transcript for better clip boundaries"""
        segments = transcript_data.get("segments", [])
        if not segments:
            logger.warning("No transcript segments found, using time-based segmentation")
            return []
            
        natural_breaks = []
        
        # Find pauses between sentences (typically longer gaps)
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]
            
            # Calculate gap between segments
            gap = next_seg['start'] - current_seg['end']
            
            # Check if segment ends with sentence-ending punctuation
            ends_sentence = current_seg['text'].rstrip().endswith(('.', '!', '?'))
            
            # Consider it a natural break if there's a significant pause or sentence end
            if gap > 0.75 or ends_sentence:
                natural_breaks.append({
                    'time': current_seg['end'],
                    'quality': (5 if ends_sentence else 3) + (min(gap * 2, 5)),  # Score quality of break
                    'text_context': current_seg['text']
                })
        
        # Sort by quality (higher is better)
        natural_breaks.sort(key=lambda x: x['quality'], reverse=True)
        return natural_breaks
        
    def create_clip(input_path, output_dir, start_time, end_time, index):
        """Create a clip using ffmpeg with the specified start and end times"""
        # Create a unique filename 
        output_filename = f"clip_{index}_smart_{int(start_time)}to{int(end_time)}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Build the ffmpeg command
        duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y",
            "-analyzeduration", "100M", "-probesize", "100M",
            "-ss", str(start_time),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "medium", 
            "-c:a", "aac", "-strict", "experimental", "-b:a", "128k",
            output_path
        ]
        
        try:
            logger.info(f"Creating clip {index+1}: {start_time:.1f}s to {end_time:.1f}s (duration: {duration:.1f}s)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to create clip: {result.stderr[:200]}")
                return None
                
            return {
                "path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "title": f"Clip {index+1}: {start_time:.1f}s to {end_time:.1f}s"
            }
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}")
            return None
    
    try:
        # Create output directory if using local path
        output_dir = "/data/clips"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get total video duration
        duration_cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]
        
        result = subprocess.run(duration_cmd, capture_output=True, text=True)
        duration_data = json.loads(result.stdout)
        total_duration = float(duration_data.get('format', {}).get('duration', 0))
        
        if total_duration <= 0:
            logger.error("Could not determine video duration")
            return []
            
        logger.info(f"Video total duration: {total_duration:.2f} seconds")
        
        # Find natural breaks in the content
        natural_breaks = find_natural_segments(transcript_data)
        logger.info(f"Found {len(natural_breaks)} potential natural break points")
        
        clips = []
        
        # Use natural breaks if we have enough of them
        if natural_breaks and len(natural_breaks) >= target_clips - 1:
            # Take top N-1 best breaks to create N clips
            selected_breaks = natural_breaks[:target_clips - 1]
            selected_times = sorted([b['time'] for b in selected_breaks])
            
            # Create start/end pairs for clips
            start_times = [0] + selected_times
            end_times = selected_times + [total_duration]
            
            # Validate and adjust segments
            for i in range(len(start_times)):
                start = start_times[i]
                end = end_times[i]
                duration = end - start
                
                # Skip segments that are too short
                if duration < min_duration:
                    continue
                    
                # Cap segments that are too long
                if duration > max_duration:
                    end = start + max_duration
                    
                # Create the clip
                clip_info = create_clip(video_path, output_dir, start, end, i)
                if clip_info:
                    clips.append(clip_info)
                    
        else:
            # If we don't have good natural breaks, use evenly spaced clips
            # with slight variations for more natural feel
            logger.info("Using evenly spaced clips with variations")
            
            # Determine base clip duration with some randomness
            base_duration = min(max_duration, total_duration / target_clips)
            
            for i in range(target_clips):
                # Add some variation to make it feel more natural
                variation = random.uniform(-3, 3) if base_duration > 25 else 0
                clip_duration = base_duration + variation
                
                # Ensure duration constraints
                clip_duration = max(min_duration, min(clip_duration, max_duration))
                
                # Calculate start and end, ensuring we don't exceed video length
                start_time = i * (total_duration / target_clips)
                end_time = min(start_time + clip_duration, total_duration)
                
                # Create the clip
                clip_info = create_clip(video_path, output_dir, start_time, end_time, i)
                if clip_info:
                    clips.append(clip_info)
        
        logger.info(f"Successfully created {len(clips)} variable-length clips")
        return clips
        
    except Exception as e:
        logger.error(f"Error in smart clip creation: {str(e)}")
        return []

# Add a smart highlight selector at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=600,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def smart_highlight_selector(transcript_data, video_title, num_highlights=3, 
                            min_duration=15, max_duration=60, content_type="interesting"):
    """
    Selects highlights from a transcript based on specific content preferences.
    
    Args:
        transcript_data: Transcript with timestamps
        video_title: Title of the video
        num_highlights: Number of highlights to select
        min_duration: Minimum highlight duration in seconds
        max_duration: Maximum highlight duration in seconds
        content_type: Type of content to look for (funny, interesting, etc.)
        
    Returns:
        List of highlight info dictionaries with timestamps
    """
    import os
    import json
    import openai
    import logging
    import random
    from difflib import SequenceMatcher
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Configure OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Define content type specific instructions
    content_type_instructions = {
        "interesting": "Focus on intellectually engaging and thought-provoking moments that viewers will find fascinating.",
        "funny": "Look for humorous moments, jokes, laughter, or amusing anecdotes that will entertain viewers.",
        "dramatic": "Identify emotionally powerful moments with high tension, conflicts, or impactful revelations.",
        "educational": "Find explanations of concepts, demonstrations, or moments that teach something valuable.",
        "surprising": "Look for unexpected twists, shocking revelations, or moments that defy expectations.",
        "inspiring": "Identify motivational content, success stories, or uplifting moments that inspire action."
    }
    
    # Get specific instructions or use default
    content_instruction = content_type_instructions.get(
        content_type, "Focus on the most engaging moments that will work well as short clips."
    )
    
    logger.info(f"Selecting {content_type} highlights from video: {video_title}")
    
    # Extract transcript segments and full text
    segments = transcript_data.get("segments", [])
    full_text = transcript_data.get("full_text", "")
    
    if not segments or not full_text:
        logger.warning("Empty transcript data, returning generic highlights")
        return _generate_generic_highlights(num_highlights, max_duration)
    
    try:
        # Create a GPT prompt that focuses on the specified content type
        prompt = f"""
        You are an expert video editor specializing in finding {content_type} moments in videos for social media shorts.
        
        {content_instruction}
        
        For the video titled "{video_title}", analyze this transcript and identify {num_highlights} 
        distinct moments that would make great shorts with durations between {min_duration} 
        and {max_duration} seconds.
        
        TRANSCRIPT:
        {full_text[:4000]}  # Limit text to stay within token limit
        
        For each highlight, provide:
        1. A brief description of the {content_type} moment
        2. A catchy title that will grab viewer attention
        3. Specific text from the transcript that matches this moment
        
        Format your response as a JSON array:
        [
          {{
            "title": "Catchy Title Here",
            "description": "Description of the moment",
            "transcript_text": "Exact text from transcript for matching"
          }}
        ]
        
        ONLY include the JSON array in your response, no other text.
        """
        
        # Call the LLM to identify highlights
        logger.info("Calling LLM to identify highlights...")
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Use an appropriate model
            messages=[
                {"role": "system", "content": f"You are an expert video editor specializing in {content_type} content for social media."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Extract JSON (handling the possibility of code blocks or plain JSON)
        import re
        json_match = re.search(r'(\[[\s\S]*\])', content)
        if json_match:
            highlighted_moments = json.loads(json_match.group(1))
        else:
            # Try parsing the whole thing as JSON
            try:
                highlighted_moments = json.loads(content)
            except:
                logger.error("Could not parse LLM response as JSON")
                return _generate_generic_highlights(num_highlights, max_duration)
        
        logger.info(f"Found {len(highlighted_moments)} potential highlighted moments")
        
        # Match each highlighted moment with transcript segments
        highlights = []
        
        for moment in highlighted_moments:
            transcript_text = moment.get("transcript_text", "").lower()
            
            # Find best matching segment
            best_segment = None
            highest_similarity = 0
            
            for segment in segments:
                segment_text = segment["text"].lower()
                
                # Calculate similarity between the moment text and segment text
                similarity = SequenceMatcher(None, transcript_text, segment_text).ratio()
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_segment = segment
            
            if not best_segment:
                # Fall back to random segment if no match found
                best_segment = random.choice(segments)
            
            # Calculate start and end times
            start_time = best_segment["start"]
            
            # Get segments that fit within desired duration
            clip_segments = []
            current_duration = 0
            
            for segment in segments:
                if segment["start"] >= start_time:
                    segment_duration = segment["end"] - segment["start"]
                    
                    if current_duration + segment_duration <= max_duration:
                        clip_segments.append(segment)
                        current_duration += segment_duration
                    else:
                        break
            
            if clip_segments:
                end_time = clip_segments[-1]["end"]
                
                # Ensure minimum duration
                if end_time - start_time < min_duration:
                    end_time = start_time + min_duration
                    
                # Add highlight
                highlights.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": moment.get("title", f"{content_type.capitalize()} Highlight"),
                    "description": moment.get("description", f"A {content_type} moment from {video_title}")
                })
        
        # If we got fewer highlights than requested, pad with generic ones
        if len(highlights) < num_highlights:
            additional_needed = num_highlights - len(highlights)
            highlights.extend(_generate_generic_highlights(additional_needed, max_duration))
        
        return highlights
        
    except Exception as e:
        logger.error(f"Error selecting highlights: {str(e)}")
        return _generate_generic_highlights(num_highlights, max_duration)

# Add select_highlights at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def select_highlights(transcript_data, video_title, num_highlights=3, max_duration=60):
    """
    Basic highlight selection based on transcript data.
    
    This is used as a fallback when smart_highlight_selector is not available.
    """
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Selecting basic highlights from {video_title}")
    
    segments = transcript_data.get("segments", [])
    if not segments:
        logger.warning("No transcript segments found, generating generic highlights")
        return _generate_generic_highlights(num_highlights, max_duration)
    
    # Find segments spaced evenly throughout the video
    total_segments = len(segments)
    step = max(1, total_segments // (num_highlights + 1))
    
    highlights = []
    for i in range(1, min(num_highlights + 1, total_segments)):
        idx = i * step
        if idx >= total_segments:
            break
            
        segment = segments[idx]
        start_time = segment["start"]
        end_time = min(start_time + max_duration, segments[-1]["end"])
        
        highlights.append({
            "start_time": start_time,
            "end_time": end_time,
            "title": f"Highlight {i}",
            "description": f"Segment starting at {int(start_time//60)}:{int(start_time%60):02d}"
        })
    
    logger.info(f"Selected {len(highlights)} basic highlights")
    return highlights

# Add clip_video at the module level
@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def clip_video(video_path, highlights):
    """
    Create video clips based on highlight timestamps.
    
    This is used as a fallback when create_smart_clips is not available.
    """
    import os
    import subprocess
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating clips from: {video_path}")
    
    output_dir = "/data/clips"
    os.makedirs(output_dir, exist_ok=True)
    
    clips = []
    
    for i, highlight in enumerate(highlights):
        start_time = highlight.get("start_time", 0)
        end_time = highlight.get("end_time", start_time + 60)
        duration = end_time - start_time
        
        output_path = os.path.join(output_dir, f"clip_{i}_{int(start_time)}to{int(end_time)}.mp4")
        
        # Build the ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-analyzeduration", "100M", "-probesize", "100M",
            "-ss", str(start_time),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "medium",
            "-c:a", "aac", 
            output_path
        ]
        
        try:
            logger.info(f"Creating clip {i+1}: {start_time:.1f}s to {end_time:.1f}s")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Failed to create clip: {result.stderr[:200]}")
                continue
                
            clips.append({
                "path": output_path,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "title": highlight.get("title", f"Clip {i+1}")
            })
            
        except Exception as e:
            logger.error(f"Error creating clip: {str(e)}")
    
    logger.info(f"Created {len(clips)} clips")
    return clips

# Add generate_caption at the module level
@app.function(
    image=image,
    timeout=300,
    secrets=[modal.Secret.from_name("shorts-generator-secrets")]
)
def generate_caption(clip_info, transcript_data, video_title):
    """
    Generate engaging captions for a video clip.
    """
    import os
    import openai
    import logging
    import json
    import re
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating caption for clip: {clip_info.get('title')}")
    
    # Extract the transcript text for this clip's time range
    start_time = clip_info.get("start_time", 0)
    end_time = clip_info.get("end_time", 0)
    
    clip_text = ""
    segments = transcript_data.get("segments", [])
    
    for segment in segments:
        # Include segments that overlap with the clip time range
        if (segment["start"] <= end_time and segment["end"] >= start_time):
            clip_text += segment["text"] + " "
    
    clip_text = clip_text.strip()
    
    # If no text found in clip range, use generic caption
    if not clip_text:
        logger.warning("No transcript text found for clip range")
        return {
            "title": f"Interesting moment from {video_title}",
            "caption": f"Check out this clip from {video_title}! #shorts",
            "hashtags": "#shorts #viral #trending"
        }
    
    # Generate caption using OpenAI
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        prompt = f"""
        You are a social media expert creating engaging captions for YouTube Shorts.
        
        Video title: "{video_title}"
        
        Transcript of clip:
        {clip_text[:500]}
        
        Create a captivating caption package with:
        1. A catchy title (max 60 characters)
        2. An engaging caption (2-3 sentences max)
        3. 3-5 relevant hashtags
        
        Format as JSON: {{"title": "...", "caption": "...", "hashtags": "..."}}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a social media caption expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON response
        json_match = re.search(r'{.*}', content, re.DOTALL)
        if json_match:
            caption_data = json.loads(json_match.group(0))
        else:
            logger.warning("Could not parse JSON response, using generic caption")
            caption_data = {
                "title": f"Highlight from {video_title}",
                "caption": f"Check out this amazing moment! #shorts",
                "hashtags": "#shorts #trending"
            }
        
        # Ensure all required fields exist
        if "title" not in caption_data:
            caption_data["title"] = f"Highlight from {video_title}"
        if "caption" not in caption_data:
            caption_data["caption"] = f"Check out this amazing moment! #shorts"
        if "hashtags" not in caption_data:
            caption_data["hashtags"] = "#shorts #trending"
            
        logger.info(f"Generated caption for clip: {caption_data['title']}")
        return caption_data
        
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}")
        return {
            "title": f"Highlight from {video_title}",
            "caption": f"Check out this amazing moment! #shorts",
            "hashtags": "#shorts #trending"
        }

# Main execution block
if __name__ == "__main__":
    print("Starting Modal deployment process...")
    
    # First set up secrets
    setup_modal_secrets()
    
    # Deploy the app - this must be done BEFORE any remote functions are called
    print("Deploying Modal app 'shorts-generator'...")
    app.deploy()
    print("Modal app deployed successfully!")
    
    # Let's skip the client-based function call since it's causing issues
    print("Note: We're skipping directory setup as it seems the Modal Client API has changed.")
    print("The directories will be automatically created when functions are called.")
    
    # Add Windows-specific connection handling
    if os.name == 'nt':  # Check if running on Windows
        print("\nNote: On Windows, you may see 'ConnectionResetError' messages in the console.")
        print("These are harmless asyncio socket issues and can be safely ignored.")
        print("Your videos should still process correctly despite these messages.")
    
    print("\n===== SHORTS GENERATOR FOR MCP HACKATHON =====")
    print("YouTube Shorts Generator using Modal for processing & Gradio for UI")
    print("Part of the MCP Hackathon - Track 3: Agentic Demo Showcase")
    
    print("\nTo work around Modal Client API issues, use these options:")
    print("1. Add processing functions directly to this file:")
    
    print("\n2. Test your functions directly:")
    print("""
    # Example test of function (add this to the end of the script)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Change to a valid URL
    print(f"\\nTesting download with: {test_url}")
    try:
        # Direct call without using Client API
        result = download_youtube_video.call(test_url)
        print(f"Success! Video path: {result[0]}, Title: {result[1]}")
    except Exception as e:
        print(f"Test failed: {str(e)}")
    """)
    
    print("\nℹ️ MCP Hackathon Submission:")
    print("- README.md should include tag: \"agent-demo-track\"")
    print("- Include a video demo link in README.md")
    print("- Deadline: June 8 at 11:59 PM UTC")
    print("- Documentation: modelcontextprotocol.io")
    
    print("\n==== TROUBLESHOOTING CORRUPT VIDEOS ====")
    print("If you're experiencing issues with corrupted video files:")
    print("1. Clear the local cache: delete files in %TEMP%/shorts_generator_cache")
    print("2. Use the validate_repair_video function to check video integrity:")
    print("""
    # Example for validating videos:
    video_path = "/data/videos/your_video.mp4"
    is_valid, message = validate_repair_video.call(video_path)
    print(f"Video valid: {is_valid}, Message: {message}")
    """)
    print("3. Add analyzeduration and probesize options to ffmpeg commands:")
    print("   ffmpeg -analyzeduration 100M -probesize 100M -i video.mp4 ...")
    
    print("\nModal deployment complete!")