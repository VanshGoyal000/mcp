import os
import tempfile
import uuid
import shutil
import requests
import logging
import re
import subprocess
import json
import hashlib
import random
import yt_dlp  # Added missing import

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a cache directory for downloaded videos
CACHE_DIR = os.path.join(tempfile.gettempdir(), "shorts_generator_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def validate_video_file(file_path):
    """Validate that a file is a working video file"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return False
            
        # Check file size - tiny files are suspicious
        file_size = os.path.getsize(file_path)
        if file_size < 10000:  # Less than 10KB
            logger.warning(f"File too small: {file_size} bytes")
            return False
        
        # Use ffprobe to check if it's a valid video
        cmd = ['ffprobe', '-v', 'error', 
               '-select_streams', 'v:0',
               '-show_entries', 'stream=codec_type,codec_name,width,height',
               '-of', 'json', file_path]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, timeout=30)
        
        if result.returncode != 0:
            logger.warning(f"FFprobe validation failed: {result.stderr}")
            return False
            
        # Try to parse the output
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        
        # No video streams is a problem
        if not streams:
            logger.warning("No streams found in the video file")
            return False
        
        # Check if we have at least one video stream
        has_video = False
        for stream in streams:
            if stream.get('codec_type') == 'video':
                has_video = True
                
                # Check if we have basic video properties
                if not stream.get('width') or not stream.get('height'):
                    logger.warning("Video stream missing dimensions")
                    return False
        
        if not has_video:
            logger.warning("No video stream found in file")
            return False
            
        # If we reach here, the file is probably valid
        return True
        
    except Exception as e:
        logger.warning(f"Error validating video file: {str(e)}")
        return False

def get_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard URL format
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URL format
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Short URL format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    # If no patterns match, hash the whole URL as fallback
    return hashlib.md5(youtube_url.encode()).hexdigest()[:11]

def download_video_from_url(youtube_url: str) -> tuple:
    """
    Download a YouTube video locally and return its path and title.
    Implements caching to avoid re-downloading.
    """
    # Get video ID for caching
    video_id = get_video_id(youtube_url)
    cache_path = os.path.join(CACHE_DIR, f"{video_id}.mp4")
    
    # Delete corrupted cached file if it exists
    if os.path.exists(cache_path):
        # Validate the file before using it
        if validate_video_file(cache_path):
            logger.info(f"Using cached video for {youtube_url}")
            
            # Try to get the title from cache metadata
            meta_path = os.path.join(CACHE_DIR, f"{video_id}.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                        video_title = metadata.get('title', 'Cached Video')
                        return cache_path, video_title
                except:
                    pass
            
            return cache_path, "Cached Video"
        else:
            logger.warning(f"Found corrupted cache file, removing: {cache_path}")
            try:
                os.remove(cache_path)
                # Also try to remove metadata if it exists
                meta_path = os.path.join(CACHE_DIR, f"{video_id}.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception as e:
                logger.error(f"Error removing corrupted file: {str(e)}")
    
    # Download the video using yt-dlp which is more reliable
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, f"{video_id}.mp4")
            
            logger.info(f"Downloading video from {youtube_url} using yt-dlp")
            
            # Use yt-dlp with increased options for better reliability
            try:
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': output_path,
                    'quiet': False,
                    'no_warnings': False,
                    'ignoreerrors': False,
                    # Add longer timeout
                    'socket_timeout': 30,
                }
                
                # Run yt-dlp
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(youtube_url, download=True)
                    video_title = info_dict.get('title', 'Unknown Title')
                    
                # Verify the downloaded file
                if not validate_video_file(output_path):
                    raise Exception("Downloaded file is invalid or corrupted")
                
                # Copy to cache
                os.makedirs(CACHE_DIR, exist_ok=True)
                shutil.copyfile(output_path, cache_path)
                
                # Save metadata
                with open(os.path.join(CACHE_DIR, f"{video_id}.json"), 'w') as f:
                    json.dump({
                        'title': video_title,
                        'url': youtube_url
                    }, f)
                    
                logger.info(f"Successfully downloaded video to {cache_path}")
                return cache_path, video_title
                
            except Exception as e:
                logger.error(f"yt-dlp error: {str(e)}")
                raise
    
    except Exception as e:
        logger.error(f"Failed to download video: {str(e)}")
        
        # Use sample video as fallback
        sample_path = os.path.join(CACHE_DIR, "sample_video.mp4")
        if not os.path.exists(sample_path) or not validate_video_file(sample_path):
            # Try to download a known good sample video
            try:
                logger.info("Downloading sample video for fallback")
                sample_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
                
                response = requests.get(sample_url, stream=True)
                if response.status_code == 200:
                    with open(sample_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                            
                    logger.info(f"Sample video downloaded to {sample_path}")
                else:
                    logger.error(f"Failed to download sample video: HTTP {response.status_code}")
            except Exception as sample_error:
                logger.error(f"Error downloading sample video: {str(sample_error)}")
        
        if os.path.exists(sample_path) and validate_video_file(sample_path):
            logger.warning(f"Using sample video due to error: {str(e)}")
            return sample_path, "Sample Video"
        else:
            raise Exception(f"Failed to use sample video as fallback")

def download_using_requests(youtube_url, cache_path, video_id):
    """Try to download using requests as a fallback - may not work for all videos"""
    import re
    
    try:
        temp_dir = tempfile.gettempdir()
        output_filename = f"video_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Extract video ID
        video_id = None
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
        if match:
            video_id = match.group(1)
        else:
            raise ValueError("Could not extract video ID")
            
        # Try to get the title
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}")
        title_match = re.search(r'<title>(.*?)</title>', response.text)
        video_title = title_match.group(1).replace(" - YouTube", "") if title_match else "Unknown Title"
        
        # Use direct download URL
        download_url = f"https://images{random.randint(1,33)}-focus-opensocial.googleusercontent.com/gadgets/proxy?container=none&url=https%3A%2F%2Fwww.youtube.com%2Fget_video_info%3Fvideo_id%3D{video_id}"
        
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        
            return output_path, video_title
        else:
            raise Exception(f"Request failed with status {response.status_code}")
    
    except Exception as e:
        logger.error(f"Alternative download method failed: {str(e)}")
        raise
        
def use_sample_video(error_message):
    """Provide a sample video when download fails"""
    temp_dir = tempfile.gettempdir()
    sample_path = os.path.join(temp_dir, "sample_video.mp4")
    
    # Check if we have the sample video, otherwise create a dummy one
    if not os.path.exists(sample_path):
        logger.info("Creating sample video for demo purposes")
        try:
            # Try to download a reliable sample video from a CDN
            sample_url = "https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
            response = requests.get(sample_url, stream=True)
            
            if response.status_code == 200:
                with open(sample_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
            else:
                # If that fails too, create an empty file
                with open(sample_path, 'wb') as f:
                    f.write(b'')
        except:
            # Last resort: create an empty file
            with open(sample_path, 'wb') as f:
                f.write(b'')
    
    logger.warning(f"Using sample video due to error: {error_message}")
    return sample_path, "Sample Video (Download Failed)"

def local_video_to_path(video_file) -> str:
    """
    Save an uploaded video file to a temporary path and return the path.
    """
    temp_dir = tempfile.gettempdir()
    unique_filename = f"{uuid.uuid4()}.mp4"
    file_path = os.path.join(temp_dir, unique_filename)
    
    # Save the uploaded file
    with open(file_path, 'wb') as f:
        f.write(video_file)
    
    return file_path
