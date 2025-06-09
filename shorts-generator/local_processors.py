import os
import tempfile
import json
from pathlib import Path
import logging
import numpy as np
from dotenv import load_dotenv
import shutil
import requests
import subprocess
import random
from utils.transcription import transcribe_audio
from utils.debug import debug_print, examine_file  # Import debug utilities
from utils.video import validate_video_file  # Import the validation function

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_video_locally(video_path: str) -> dict:
    """
    Transcribe a video using available methods.
    Returns transcript with timestamps.
    """
    # Try different methods in order of preference
    if os.environ.get("GOOGLE_API_KEY"):
        # Use Google API if available
        transcript = transcribe_audio(video_path, method='google')
    else:
        # Fall back to local Whisper
        transcript = transcribe_audio(video_path, method='whisper-local')
    
    # Print transcript for debugging
    print("\n=== TRANSCRIPT DATA ===")
    print(f"Full text length: {len(transcript['full_text'])} characters")
    print(f"Number of segments: {len(transcript['segments'])}")
    print("\nFirst 300 characters of transcript:")
    print(transcript['full_text'][:300] + "...")
    print("\nFirst 3 segments:")
    for i, segment in enumerate(transcript['segments'][:3]):
        print(f"Segment {i}: {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
    print("=== END TRANSCRIPT DATA ===\n")
    
    return transcript

def select_highlights_locally(transcript_data: dict, video_title: str, num_highlights: int = 3, segment_length: int = 60) -> list:
    """
    Select the most engaging highlights from a transcript.
    Returns a list of dictionaries with start_time, end_time, and description.
    """
    logger.info(f"Selecting {num_highlights} highlights from transcript")
    
    # Print transcript data for debugging
    print("\n=== TRANSCRIPT DATA ===")
    print(f"Full text length: {len(transcript_data.get('full_text', ''))} characters")
    print(f"Number of segments: {len(transcript_data.get('segments', []))}")
    print("\nFirst 300 characters of transcript:")
    print(transcript_data.get('full_text', '')[:300] + "...")
    print("\nFirst 3 segments:")
    for i, segment in enumerate(transcript_data.get('segments', [])[:3]):
        print(f"Segment {i}: {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
    print("=== END TRANSCRIPT DATA ===\n")
    
    # Verify transcript has content
    if not transcript_data.get("full_text") or len(transcript_data.get("segments", [])) == 0:
        logger.warning("Transcript is empty or missing data, using algorithm-based highlights")
        return _select_highlights_algorithmically(transcript_data, num_highlights, segment_length)
        
    try:
        # Use Google Gemini instead of OpenAI
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Configure genai with API key
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return _select_highlights_algorithmically(transcript_data, num_highlights, segment_length)
        
        genai.configure(api_key=api_key)
        
        # Use Gemini to select highlights
        segments = transcript_data["segments"]
        full_text = transcript_data["full_text"]
        
        # Prepare prompt for Gemini
        prompt = f"""
        You are an expert YouTube video editor. Your task is to identify {num_highlights} engaging moments from this transcript 
        that would work well as {segment_length}-second YouTube Shorts.
        
        Video title: "{video_title}"
        
        Transcript:
        {full_text[:3000]}  # Limit to first 3000 chars to stay within token limits
        
        For each highlight, provide:
        1. Start time (in seconds)
        2. A catchy title for the Short
        3. A brief description of why this moment is engaging
        
        Format your response as a JSON array like this:
        [
          {{"start_time": 120.5, "title": "Unexpected Plot Twist", "description": "The moment when the main character reveals their secret"}}
        ]
        
        Only return the JSON array, no additional text.
        """
        
        # Call Gemini
        model = genai.GenerativeModel('gemini-1.0-pro')
        response = model.generate_content(prompt)
        
        # Parse the response
        content = response.text
        
        # Extract JSON from response
        import re
        import json
        
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            highlights_data = json.loads(json_match.group(0))
        else:
            raise ValueError("Could not extract JSON from response")
        
        # Format highlights with start_time and end_time
        highlights = []
        for i, highlight in enumerate(highlights_data[:num_highlights]):
            start_time = float(highlight.get("start_time", i * 60))
            
            # Ensure start_time is within the video duration
            start_time = min(start_time, _get_last_segment_end(segments) - segment_length)
            
            highlights.append({
                "start_time": start_time,
                "end_time": start_time + segment_length,
                "title": highlight.get("title", f"Highlight {i+1}"),
                "description": highlight.get("description", f"Interesting moment at {int(start_time//60)}:{int(start_time%60):02d}")
            })
        
        return highlights
        
    except Exception as e:
        logger.error(f"Highlight selection failed: {str(e)}")
        # Return basic highlights at equal intervals as a fallback
        return _select_highlights_algorithmically(transcript_data, num_highlights, segment_length)

def _select_highlights_algorithmically(transcript_data: dict, num_highlights: int, segment_length: int) -> list:
    """Simple algorithm to select highlights at equal intervals"""
    segments = transcript_data["segments"]
    
    if not segments:
        # If no segments, create dummy highlights
        return [
            {
                "start_time": i * segment_length,
                "end_time": (i + 1) * segment_length,
                "title": f"Highlight {i+1}",
                "description": f"Auto-selected highlight {i+1}"
            }
            for i in range(num_highlights)
        ]
    
    # Get the total duration of the video
    last_segment_end = _get_last_segment_end(segments)
    
    # Calculate equal intervals
    interval = max(last_segment_end / (num_highlights + 1), segment_length)
    
    # Generate highlights at these intervals
    highlights = []
    for i in range(num_highlights):
        start_time = min((i + 1) * interval, last_segment_end - segment_length)
        if start_time < 0:
            start_time = 0
            
        # Find the closest segment to this time
        closest_segment = min(segments, key=lambda s: abs(s["start"] - start_time))
        
        highlights.append({
            "start_time": closest_segment["start"],
            "end_time": closest_segment["start"] + segment_length,
            "title": f"Highlight {i+1}",
            "description": f"Interesting moment at {int(closest_segment['start']//60)}:{int(closest_segment['start']%60):02d}"
        })
    
    return highlights

def _get_last_segment_end(segments):
    """Get the end time of the last segment"""
    if not segments:
        return 180  # Default to 3 minutes if no segments
    return segments[-1]["end"]

def clip_video_locally(video_path: str, highlights: list, content_type="interesting") -> list:
    """
    Clip segments from a video based on highlight timestamps.
    Returns a list of paths to the clipped videos.
    
    Args:
        video_path: Path to the video file
        highlights: List of highlight dictionaries with start/end times
        content_type: Type of content (funny, interesting, etc.) to optimize clips for
    """
    logger.info(f"Clipping {len(highlights)} segments from video: {video_path} (Type: {content_type})")
    debug_print(highlights, "Highlights to Clip")
    
    # Verify file exists and is valid
    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        debug_print(f"File does not exist: {video_path}")
        return _generate_dummy_clips(num_clips=len(highlights))
    
    # Validate the video file using our utility function
    if not validate_video_file(video_path):
        logger.error(f"Video file validation failed: {video_path}")
        return _generate_dummy_clips(num_clips=len(highlights))
    
    # Examine input file
    examine_file(video_path)
    
    # Make sure file is accessible
    video_path = os.path.abspath(video_path)
    
    clip_infos = []
    clips_dir = os.path.join(tempfile.gettempdir(), "shorts_clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    try:
        for i, highlight in enumerate(highlights):
            start_time = highlight["start_time"]
            end_time = highlight["end_time"]
            title = highlight["title"].replace(" ", "_")[:30]
            
            # Create output filename
            output_filename = f"clip_{i}_{title}.mp4"
            output_path = os.path.join(clips_dir, output_filename)
            
            logger.info(f"Clipping highlight {i+1}: {start_time}s to {end_time}s -> {output_path}")
            
            # Use ffmpeg with re-encoding instead of stream copy
            try:
                # Use re-encoding by default - more reliable than copy
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output files
                    '-i', video_path,
                    '-ss', str(start_time),  # Put -ss before -i for more accurate seeking
                    '-t', str(end_time - start_time),  # Use duration instead of end time
                    '-c:v', 'libx264',  # Re-encode video
                    '-c:a', 'aac',      # Re-encode audio
                    '-strict', 'experimental',
                    '-b:a', '128k',
                    output_path
                ]
                
                debug_print(" ".join(cmd), f"FFmpeg Command for Clip {i+1}")
                
                import subprocess
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    raise Exception(f"FFmpeg error: {result.stderr}")
                
                # Check if file was created and has content
                if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:  # Ensure it's not just a header
                    examine_file(output_path)
                    clip_infos.append({
                        "path": output_path,
                        "start_time": start_time,
                        "end_time": end_time,
                        "title": highlight["title"],
                        "description": highlight.get("description", "")
                    })
                else:
                    logger.warning(f"Failed to create clip {i}: Output file is empty or too small")
                    raise Exception("Output file is too small or empty")
                    
            except Exception as e:
                logger.error(f"Error clipping segment {i}: {str(e)}")
                # Try with different parameters
                try:
                    logger.info(f"Attempting alternate method for clip {i+1}")
                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', video_path,
                        '-ss', str(start_time),
                        '-to', str(end_time),
                        '-vf', 'scale=640:360',  # Force resolution
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',  # Speed up encoding
                        '-crf', '23',            # Quality setting
                        '-c:a', 'aac',
                        output_path
                    ]
                    debug_print(" ".join(cmd), f"FFmpeg Alternate Command for Clip {i+1}")
                    
                    result = subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True
                    )
                    
                    if result.returncode != 0:
                        logger.error(f"FFmpeg alternate method error: {result.stderr}")
                        raise Exception(f"FFmpeg alternate method error: {result.stderr}")
                    
                    if os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
                        examine_file(output_path)
                        clip_infos.append({
                            "path": output_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "title": highlight["title"],
                            "description": highlight.get("description", "")
                        })
                    else:
                        raise Exception("Output file is too small or empty")
                except Exception as e2:
                    logger.error(f"Second attempt at clipping segment {i} failed: {str(e2)}")
    
    except Exception as e:
        logger.error(f"Video clipping failed: {str(e)}")
        debug_print(str(e), "Video Clipping Error")
    
    # If no clips were successfully created, create dummy clips
    if not clip_infos:
        logger.warning("No clips were created - falling back to dummy clips")
        return _generate_dummy_clips(num_clips=len(highlights))
    
    debug_print(clip_infos, "Successfully Created Clips")
    return clip_infos

def _generate_dummy_clips(num_clips=3):
    """Generate dummy clips when clipping fails"""
    logger.info(f"Generating {num_clips} dummy clips for testing")
    clips_dir = os.path.join(tempfile.gettempdir(), "shorts_clips")
    os.makedirs(clips_dir, exist_ok=True)
    
    dummy_clips = []
    for i in range(num_clips):
        # Create dummy file
        dummy_path = os.path.join(clips_dir, f"dummy_clip_{i}.mp4")
        
        # Try to copy a sample video if we can find one
        sample_video = os.path.join(tempfile.gettempdir(), "sample_video.mp4")
        if os.path.exists(sample_video):
            try:
                shutil.copy(sample_video, dummy_path)
                logger.info(f"Copied sample video to {dummy_path}")
            except Exception as e:
                logger.error(f"Failed to copy sample video: {str(e)}")
                # If copy fails, create empty file
                with open(dummy_path, 'wb') as f:
                    f.write(b'DUMMY VIDEO')
        else:
            # Create empty file
            with open(dummy_path, 'wb') as f:
                f.write(b'DUMMY VIDEO')
        
        dummy_clips.append({
            "path": dummy_path,
            "start_time": i * 60,
            "end_time": (i + 1) * 60,
            "title": f"Example Highlight {i+1}",
            "description": "This is a placeholder clip since video clipping failed."
        })
    
    return dummy_clips

def generate_caption_locally(clip_info: dict, transcript_data: dict, video_title: str) -> dict:
    """
    Generate captions and titles for a video clip.
    Uses simpler approach without requiring external API.
    """
    logger.info(f"Generating caption for clip: {clip_info['title']}")
    
    try:
        # Extract transcript segment for this clip
        start_time = clip_info["start_time"]
        end_time = clip_info["end_time"]
        
        clip_transcript = ""
        for segment in transcript_data["segments"]:
            if segment["end"] >= start_time and segment["start"] <= end_time:
                clip_transcript += segment["text"] + " "
        
        clip_transcript = clip_transcript.strip()
        
        # Check if API keys are available
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            # Use OpenAI instead of Google Gemini
            try:
                import openai
                
                client = openai.OpenAI(api_key=api_key)
                
                # Create prompt
                prompt = f"""
                You are an expert at creating engaging YouTube Shorts content.
                
                Original video title: "{video_title}"
                
                Clip description: {clip_info["description"]}
                
                Transcript of the clip:
                {clip_transcript[:500]}
                
                Please create:
                1. A catchy title (60 characters max) with emojis if appropriate
                2. An engaging caption (150 characters max)
                3. 5-7 relevant hashtags
                
                Return your response in this JSON format:
                {{
                  "title": "Your catchy title here",
                  "caption": "Your engaging caption here",
                  "hashtags": "#shorts #youtube #topic1 #topic2"
                }}
                """
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a YouTube content creator expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                # Parse the response
                content = response.choices[0].message.content
                
                # Extract JSON
                import re
                import json
                
                json_match = re.search(r'{[\s\S]*}', content)
                if json_match:
                    try:
                        caption_data = json.loads(json_match.group(0))
                        return {
                            "title": caption_data.get("title", clip_info["title"]),
                            "caption": caption_data.get("caption", clip_transcript[:100] + "..."),
                            "hashtags": caption_data.get("hashtags", "#shorts #youtube #viral")
                        }
                    except:
                        pass
            except Exception as e:
                logger.warning(f"OpenAI caption generation failed: {str(e)}")
        
        # Fallback: create simple captions based on the clip info
        title = clip_info["title"]
        
        # If title is just "Highlight X", make it more engaging
        if title.startswith("Highlight "):
            words = clip_transcript.split()
            if len(words) > 5:
                title = f"✨ {' '.join(words[:5])}..."
        
        # Generate caption from transcript
        if len(clip_transcript) > 10:
            caption = clip_transcript[:100] + "..." if len(clip_transcript) > 100 else clip_transcript
        else:
            caption = clip_info["description"]
            
        # Generate hashtags based on title
        hashtags = "#shorts #youtube #viral"
        
        return {
            "title": title,
            "caption": caption,
            "hashtags": hashtags
        }
        
    except Exception as e:
        logger.error(f"Caption generation failed: {str(e)}")
        return {
            "title": clip_info["title"],
            "caption": clip_info["description"],
            "hashtags": "#shorts #youtube #viral"
        }
