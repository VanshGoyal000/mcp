import os
import tempfile
import logging
import json
import subprocess
import requests
import time
from pathlib import Path
import traceback

# Initialize logging
logger = logging.getLogger(__name__)

def transcribe_audio(video_path, method='modal', retry_count=2):
    """
    Transcribe audio using various available methods with retries.
    
    Args:
        video_path: Path to the video file
        method: Transcription method ('modal', 'whisper-local', 'google')
        retry_count: Number of retries if transcription fails
        
    Returns:
        Dictionary with transcript data
    """
    logger.info(f"Transcribing video using {method}: {video_path}")
    
    # Verify file exists
    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        return generate_dummy_transcript()
    
    # Use a cache so we don't transcribe the same file twice
    cache_dir = os.path.join(tempfile.gettempdir(), "shorts_generator_transcripts")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key based on the file and method
    import hashlib
    file_hash = hashlib.md5(open(video_path, 'rb').read(1024 * 1024)).hexdigest()  # Hash first MB
    cache_key = f"{file_hash}_{method}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # Check cache first
    if os.path.exists(cache_file):
        logger.info(f"Using cached transcript for {video_path}")
        try:
            with open(cache_file, 'r') as f:
                transcript_data = json.load(f)
                
            # Validate the transcript data
            if (transcript_data and 
                isinstance(transcript_data, dict) and
                transcript_data.get("full_text") and 
                len(transcript_data.get("full_text")) > 0 and
                len(transcript_data.get("segments", [])) > 0):
                
                # Print debug info about the cached transcript
                print("\n=== CACHED TRANSCRIPT DATA ===")
                print(f"Full text length: {len(transcript_data['full_text'])} characters")
                print(f"Number of segments: {len(transcript_data['segments'])}")
                print("\nFirst 300 characters of transcript:")
                print(transcript_data['full_text'][:300] + "...")
                print("\nFirst 3 segments:")
                for i, segment in enumerate(transcript_data['segments'][:3]):
                    print(f"Segment {i}: {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}")
                print("=== END TRANSCRIPT DATA ===\n")
                
                return transcript_data
            else:
                logger.warning("Cached transcript is invalid or empty, regenerating")
                try:
                    os.remove(cache_file)
                except:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load cached transcript: {str(e)}")
    
    # If cache validation fails or file doesn't exist, proceed with transcription
    attempts = 0
    last_error = None
    
    # Try with retries
    while attempts <= retry_count:
        try:
            # Always try Modal first regardless of the method parameter
            if attempts == 0:
                print(f"\nAttempt {attempts+1}: Trying Modal transcription first...")
                result = transcribe_with_modal(video_path)
                if result and result.get("full_text") and len(result.get("full_text")) > 0:
                    logger.info("Modal transcription succeeded")
                    # Cache the valid result
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        logger.warning(f"Failed to cache transcript: {e}")
                    return result
                
                logger.warning("Modal transcription failed or returned empty result")
                print("Modal transcription failed, checking specific error...")
                attempts += 1
                continue
            
            # Then try whisper-local
            elif attempts == 1 or method == 'whisper-local':
                print(f"\nAttempt {attempts+1}: Trying local Whisper transcription...")
                result = transcribe_with_whisper_local(video_path)
                if result and result.get("full_text") and len(result.get("full_text")) > 0:
                    logger.info("Local Whisper transcription succeeded")
                    # Cache the valid result
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        logger.warning(f"Failed to cache transcript: {e}")
                    return result
                
                logger.warning("Local Whisper transcription failed or returned empty result")
                attempts += 1
                continue
            
            # Finally try Google if API key is available
            elif attempts == 2 or method == 'google':
                print(f"\nAttempt {attempts+1}: Trying Google API transcription...")
                result = transcribe_with_google_api(video_path)
                if result and result.get("full_text"):
                    logger.info("Google API transcription succeeded")
                    # Cache the valid result
                    try:
                        with open(cache_file, 'w') as f:
                            json.dump(result, f)
                    except Exception as e:
                        logger.warning(f"Failed to cache transcript: {e}")
                    return result
            
                logger.error("Google API transcription failed or returned empty result")
                attempts += 1
                continue
            
            else:
                logger.error(f"All transcription methods failed after {attempts} attempts")
                break
            
        except Exception as e:
            last_error = str(e)
            logger.error(f"Transcription attempt {attempts+1} failed: {str(e)}")
            print(f"Error on attempt {attempts+1}: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            attempts += 1
        
        # Wait briefly before retry
        time.sleep(1)
    
    logger.error(f"Transcription failed after {retry_count+1} attempts: {last_error}")
    return generate_dummy_transcript()

def transcribe_with_modal(video_path):
    """Use Modal for transcription"""
    print("\n=== TRYING MODAL TRANSCRIPTION ===")
    
    try:
        import modal
        
        # Try to get the path to the modal_deploy.py file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        deploy_file = os.path.join(project_dir, "modal_deploy.py")
        
        if os.path.exists(deploy_file):
            print(f"Modal deploy file found at: {deploy_file}")
        else:
            print(f"Modal deploy file not found at: {deploy_file}")
            
        print("Attempting to access Modal app...")
        
        # Create a new Modal client
        app = modal.App("shorts-generator")
        
        print("Available Modal functions:")
        print([f for f in dir(app.functions) if not f.startswith('_')])
        
        # Try the enhanced version first
        if "transcribe_video_enhanced" in dir(app.functions):
            print("Using enhanced Modal transcription function")
            
            # Get absolute path of the video file for Modal
            abs_video_path = os.path.abspath(video_path)
            print(f"Calling Modal with video path: {abs_video_path}")
            
            try:
                # Attempt the call with specific exception handling
                start_time = time.time()
                print("Starting Modal function call...")
                result = app.functions.transcribe_video_enhanced.call(abs_video_path)
                elapsed = time.time() - start_time
                print(f"Modal function call completed in {elapsed:.2f} seconds")
                
                # Validate result
                if not result or not isinstance(result, dict):
                    print(f"Invalid result type: {type(result)}")
                    return None
                    
                if "full_text" not in result:
                    print(f"Missing 'full_text' in result. Keys: {list(result.keys())}")
                    return None
                
                print(f"Modal returned transcript: {len(result.get('full_text'))} characters, {len(result.get('segments', []))} segments")
                return result
                
            except Exception as modal_error:
                logger.error(f"Modal function 'transcribe_video_enhanced' call error: {str(modal_error)}")
                print(f"Modal function error details: {str(modal_error)}")
                print("Full traceback:")
                traceback.print_exc()
                return None
                
        # Fall back to regular version
        elif "transcribe_video" in dir(app.functions):
            print("Using standard Modal transcription function")
            try:
                result = app.functions.transcribe_video.call(video_path)
                print(f"Modal returned transcript: {len(result.get('full_text'))} characters")
                return result
            except Exception as e:
                logger.error(f"Modal function 'transcribe_video' call failed: {str(e)}")
                print(f"Error: {str(e)}")
                return None
                
        else:
            logger.error("No transcription function found in Modal app")
            print("No transcription functions found in the Modal app")
            return None
            
    except Exception as e:
        logger.error(f"Modal setup failed: {str(e)}")
        print(f"Modal setup error: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        print("=== END MODAL TRANSCRIPTION ATTEMPT ===\n")

def transcribe_with_whisper_local(video_path):
    """Use local Whisper model for transcription"""
    print("\n=== TRYING WHISPER LOCAL TRANSCRIPTION ===")
    try:
        import whisper
        
        print(f"Starting Whisper transcription of: {video_path}")
        
        # Try to use a small model first to save memory
        try:
            model = whisper.load_model("tiny")
            result = model.transcribe(
                video_path,
                fp16=False,  # Disable fp16 for CPU compatibility
                language="en"  # Specify English language for better results
            )
        except Exception as small_model_error:
            print(f"Error with tiny model: {str(small_model_error)}. Trying base model...")
            model = whisper.load_model("base")
            result = model.transcribe(video_path, fp16=False)
        
        print(f"Whisper transcription completed, text length: {len(result['text'])}")
        
        # Format output with timestamps
        transcript_with_timestamps = []
        for segment in result["segments"]:
            transcript_with_timestamps.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        # Print sample of transcript
        if len(result["text"]) > 0:
            print(f"First 100 characters: {result['text'][:100]}...")
        else:
            print("WARNING: Empty transcript text")
            
        return {
            "full_text": result["text"],
            "segments": transcript_with_timestamps,
            "method": "whisper-local"
        }
    except ImportError:
        logger.error("Whisper not installed. Please install with 'pip install openai-whisper'")
        print("Error: Whisper not installed")
        return None
    except Exception as e:
        logger.error(f"Whisper transcription failed: {str(e)}")
        print(f"Whisper transcription error: {str(e)}")
        traceback.print_exc()
        return None
    finally:
        print("=== END WHISPER LOCAL TRANSCRIPTION ATTEMPT ===\n")

def transcribe_with_google_api(video_path):
    """Use Google Speech-to-Text API for transcription"""
    from os import environ
    
    if "GOOGLE_API_KEY" not in environ or not environ["GOOGLE_API_KEY"]:
        logger.error("No Google API key available")
        print("Error: No Google API key available in environment variables")
        return None
    
    # Implementation details for Google API would go here
    # ...

    # Placeholder for Google API implementation
    logger.error("Google Speech-to-Text API not implemented")
    return None

def generate_dummy_transcript():
    """Generate a dummy transcript when all transcription methods fail"""
    print("\n=== GENERATING DUMMY TRANSCRIPT DUE TO FAILURES ===")
    dummy_transcript = {
        "full_text": "Transcription failed. This is a placeholder transcript.",
        "segments": [
            {"start": 0, "end": 10, "text": "Transcription failed. Using placeholder text."},
            {"start": 10, "end": 20, "text": "Please try again with a different video."},
            {"start": 20, "end": 30, "text": "Or try a different transcription method."}
        ],
        "method": "fallback"
    }
    return dummy_transcript
