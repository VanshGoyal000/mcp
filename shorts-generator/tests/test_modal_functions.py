#!/usr/bin/env python
"""
Test script for Modal functions in the shorts-generator project.
Run this script to test individual Modal functions or all of them in sequence.
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from pprint import pprint

# Add project root to path for imports
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

# Import Modal with error handling
try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    print("WARNING: Modal package is not installed. Please install it with: pip install modal")
    MODAL_AVAILABLE = False
except Exception as e:
    print(f"WARNING: Error importing Modal: {e}")
    MODAL_AVAILABLE = False

# Try to import Modal configuration
try:
    from modal_config import app, get_secrets
    MODAL_CONFIG_AVAILABLE = True
except ImportError:
    print("WARNING: Could not import modal_config. Make sure modal_config.py exists in the project root.")
    MODAL_CONFIG_AVAILABLE = False
except Exception as e:
    print(f"WARNING: Error importing modal_config: {e}")
    MODAL_CONFIG_AVAILABLE = False

def setup_argparser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description='Test Modal functions for shorts-generator')
    
    parser.add_argument('--function', '-f', type=str, choices=[
        'all',
        'setup_directories',
        'download_youtube_video',
        'transcribe_video_enhanced',
        'select_highlights',
        'smart_highlight_selector',
        'clip_video',
        'create_smart_clips',
        'generate_caption',
        'validate_repair_video'
    ], default='all', help='Function to test')
    
    parser.add_argument('--url', '-u', type=str, 
                      default='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                      help='YouTube URL for testing')
    
    parser.add_argument('--clips', '-c', type=int, default=2,
                      help='Number of clips to generate')
    
    parser.add_argument('--min-duration', type=int, default=15,
                      help='Minimum clip duration in seconds')
    
    parser.add_argument('--max-duration', type=int, default=60,
                      help='Maximum clip duration in seconds')
    
    parser.add_argument('--content-type', type=str, 
                      choices=['interesting', 'funny', 'dramatic', 'educational', 'surprising', 'inspiring'],
                      default='interesting',
                      help='Content type for highlight selection')
                      
    parser.add_argument('--quiet', '-q', action='store_true',
                      help='Reduce output verbosity')
    
    return parser

def print_divider():
    """Print a divider for better output readability"""
    print("\n" + "="*80 + "\n")

def log(message, quiet=False):
    """Print a log message unless quiet mode is enabled"""
    if not quiet:
        print(message)

def check_modal_available():
    """Check if Modal is properly configured and available"""
    if not MODAL_AVAILABLE:
        print("ERROR: Modal package is not available. Please install it with: pip install modal")
        return False
    
    if not MODAL_CONFIG_AVAILABLE:
        print("ERROR: Modal configuration is not available. Check modal_config.py")
        return False
    
    # Check if app has the necessary attributes
    if not hasattr(app, 'setup_directories') and not hasattr(app, 'download_youtube_video'):
        print("ERROR: Modal app is not properly configured with functions.")
        print("Make sure to run 'python modal_deploy.py' first to deploy the functions.")
        print("You may also need to update modal_config.py to use modal.App instead of modal.Stub.")
        return False
        
    return True

def test_setup_directories(quiet=False):
    """Test the setup_directories function"""
    if not check_modal_available():
        return False
        
    log("Testing setup_directories...", quiet)
    try:
        result = app.setup_directories.call()
        log(f"Result: {result}", quiet)
        return True
    except Exception as e:
        print(f"ERROR: Failed to set up directories: {e}")
        return False

def test_download_video(url, quiet=False):
    """Test the download_youtube_video function"""
    if not check_modal_available():
        return None
        
    log(f"Testing download_youtube_video with URL: {url}...", quiet)
    try:
        # First check if the function exists
        if not hasattr(app, 'download_youtube_video'):
            print("ERROR: download_youtube_video function is not available in the Modal app.")
            print("Make sure you've deployed the Modal app with 'python modal_deploy.py'")
            return None
            
        result = app.download_youtube_video.call(url)
        if isinstance(result, tuple):
            video_path, title = result
            log(f"Downloaded video: '{title}'", quiet)
            log(f"Video path: {video_path}", quiet)
            return video_path
        else:
            log(f"Result: {result}", quiet)
            return result.get('path')
    except AttributeError as e:
        print(f"ERROR: Modal function access error: {e}")
        print("Make sure you've correctly deployed the Modal app with 'python modal_deploy.py'")
        return None
    except Exception as e:
        print(f"ERROR: Failed to download video: {e}")
        return None

def test_transcribe_video(video_path, quiet=False):
    """Test the transcribe_video_enhanced function"""
    if not check_modal_available():
        return None
        
    if not video_path:
        print("ERROR: No video path provided for transcription")
        return None
        
    log(f"Testing transcribe_video_enhanced with path: {video_path}...", quiet)
    try:
        result = app.transcribe_video_enhanced.call(video_path)
        log(f"Transcription complete. Text length: {len(result.get('full_text', ''))}", quiet)
        log(f"Number of segments: {len(result.get('segments', []))}", quiet)
        if not quiet:
            print("First 200 chars of transcript:")
            print(result.get('full_text', '')[:200] + "...")
        return result
    except AttributeError as e:
        print(f"ERROR: Modal function access error: {e}")
        print("Make sure you've correctly deployed the Modal app with 'python modal_deploy.py'")
        return None
    except Exception as e:
        print(f"ERROR: Failed to transcribe video: {e}")
        return None

def test_select_highlights(transcript_data, video_title="Test Video", num_clips=2, 
                          max_duration=60, quiet=False):
    """Test the select_highlights function"""
    if not transcript_data:
        print("ERROR: No transcript data provided for highlight selection")
        return None
        
    log(f"Testing select_highlights for video: {video_title}...", quiet)
    try:
        result = app.select_highlights.call(transcript_data, video_title, num_clips, max_duration)
        log(f"Selected {len(result)} highlights", quiet)
        if not quiet:
            for i, highlight in enumerate(result):
                print(f"  Highlight {i+1}: {highlight['start_time']:.1f}s - {highlight['end_time']:.1f}s")
                print(f"    Title: {highlight['title']}")
                print(f"    Description: {highlight['description'][:50]}...")
        return result
    except Exception as e:
        print(f"ERROR: Failed to select highlights: {e}")
        return None

def test_smart_highlight_selector(transcript_data, video_title="Test Video", num_clips=2, 
                                min_duration=15, max_duration=60, content_type="interesting", quiet=False):
    """Test the smart_highlight_selector function"""
    if not transcript_data:
        print("ERROR: No transcript data provided for smart highlight selection")
        return None
        
    log(f"Testing smart_highlight_selector for video: {video_title} (content type: {content_type})...", quiet)
    try:
        result = app.smart_highlight_selector.call(
            transcript_data, video_title, num_clips, min_duration, max_duration, content_type
        )
        log(f"Selected {len(result)} smart highlights", quiet)
        if not quiet:
            for i, highlight in enumerate(result):
                print(f"  Highlight {i+1}: {highlight['start_time']:.1f}s - {highlight['end_time']:.1f}s")
                print(f"    Title: {highlight['title']}")
                print(f"    Description: {highlight['description'][:50]}...")
        return result
    except Exception as e:
        print(f"ERROR: Failed to select smart highlights: {e}")
        return None

def test_clip_video(video_path, highlights, quiet=False):
    """Test the clip_video function"""
    if not video_path or not highlights:
        print("ERROR: Missing video path or highlights for clipping")
        return None
        
    log(f"Testing clip_video with {len(highlights)} highlights...", quiet)
    try:
        result = app.clip_video.call(video_path, highlights)
        log(f"Created {len(result)} clips", quiet)
        if not quiet:
            for i, clip in enumerate(result):
                print(f"  Clip {i+1}: {clip['path']}")
                print(f"    Duration: {clip['end_time'] - clip['start_time']:.1f}s")
        return result
    except Exception as e:
        print(f"ERROR: Failed to clip video: {e}")
        return None

def test_create_smart_clips(video_path, transcript_data, min_duration=15, max_duration=60, 
                           num_clips=2, quiet=False):
    """Test the create_smart_clips function"""
    if not video_path or not transcript_data:
        print("ERROR: Missing video path or transcript data for smart clipping")
        return None
        
    log(f"Testing create_smart_clips with min={min_duration}s, max={max_duration}s, target={num_clips} clips...", quiet)
    try:
        result = app.create_smart_clips.call(video_path, transcript_data, min_duration, max_duration, num_clips)
        log(f"Created {len(result)} smart clips", quiet)
        if not quiet:
            for i, clip in enumerate(result):
                print(f"  Clip {i+1}: {clip['path']}")
                print(f"    Duration: {clip['end_time'] - clip['start_time']:.1f}s")
                print(f"    Title: {clip.get('title', 'Untitled')}")
        return result
    except Exception as e:
        print(f"ERROR: Failed to create smart clips: {e}")
        return None

def test_generate_caption(clip_info, transcript_data, video_title="Test Video", quiet=False):
    """Test the generate_caption function"""
    if not clip_info or not transcript_data:
        print("ERROR: Missing clip info or transcript data for caption generation")
        return None
        
    log(f"Testing generate_caption for clip: {clip_info.get('title', 'Untitled')}...", quiet)
    try:
        result = app.generate_caption.call(clip_info, transcript_data, video_title)
        log("Caption generated successfully", quiet)
        if not quiet:
            print(f"  Title: {result.get('title', 'No title')}")
            print(f"  Caption: {result.get('caption', 'No caption')[:100]}...")
            print(f"  Hashtags: {result.get('hashtags', '#shorts')}")
        return result
    except Exception as e:
        print(f"ERROR: Failed to generate caption: {e}")
        return None

def test_validate_repair_video(video_path, quiet=False):
    """Test the validate_repair_video function"""
    if not video_path:
        print("ERROR: No video path provided for validation")
        return False, "No video path"
        
    log(f"Testing validate_repair_video with path: {video_path}...", quiet)
    try:
        result = app.validate_repair_video.call(video_path)
        log(f"Result: {result}", quiet)
        return result
    except Exception as e:
        print(f"ERROR: Failed to validate/repair video: {e}")
        return False, str(e)

def run_all_tests(args):
    """Run all tests in sequence"""
    print_divider()
    print("RUNNING ALL TESTS")
    print_divider()
    
    # Start with setup
    test_setup_directories(args.quiet)
    print_divider()
    
    # Download test video
    video_path = test_download_video(args.url, args.quiet)
    if not video_path:
        print("ERROR: Failed to download video. Stopping tests.")
        return
    print_divider()
    
    # Validate video
    valid, message = test_validate_repair_video(video_path, args.quiet)
    if not valid:
        print(f"WARNING: Video validation failed: {message}")
    print_divider()
    
    # Transcribe video
    transcript_data = test_transcribe_video(video_path, args.quiet)
    if not transcript_data:
        print("ERROR: Failed to transcribe video. Stopping tests.")
        return
    print_divider()
    
    # Get video title from URL
    video_title = args.url.split("=")[-1]
    
    # Test basic highlight selection
    highlights = test_select_highlights(transcript_data, video_title, args.clips, args.max_duration, args.quiet)
    print_divider()
    
    # Test smart highlight selection
    smart_highlights = test_smart_highlight_selector(
        transcript_data, video_title, args.clips, args.min_duration, args.max_duration, 
        args.content_type, args.quiet
    )
    print_divider()
    
    # Use whichever highlights worked
    selected_highlights = smart_highlights if smart_highlights else highlights
    if not selected_highlights:
        print("ERROR: Failed to generate highlights. Stopping tests.")
        return
    
    # Test basic clip creation
    clips = test_clip_video(video_path, selected_highlights, args.quiet)
    print_divider()
    
    # Test smart clip creation
    smart_clips = test_create_smart_clips(
        video_path, transcript_data, args.min_duration, args.max_duration, args.clips, args.quiet
    )
    print_divider()
    
    # Use whichever clips worked
    selected_clips = smart_clips if smart_clips else clips
    if not selected_clips:
        print("ERROR: Failed to create clips. Stopping tests.")
        return
    
    # Test caption generation for each clip
    for clip in selected_clips:
        test_generate_caption(clip, transcript_data, video_title, args.quiet)
        
    print_divider()
    print("ALL TESTS COMPLETED")

def main():
    """Main function to parse arguments and run tests"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    print(f"Modal Shorts Generator - Function Testing Tool")
    
    # Check Modal is available before proceeding
    if not check_modal_available():
        print("\nERROR: Modal is not properly configured or available.")
        print("Please fix Modal configuration before running tests.")
        print("If you're seeing 'modal.Stub' errors, you may need to update modal_config.py")
        print("to use modal.App instead of modal.Stub.")
        sys.exit(1)
    
    try:
        if args.function == 'all':
            run_all_tests(args)
        elif args.function == 'setup_directories':
            test_setup_directories(args.quiet)
        elif args.function == 'download_youtube_video':
            test_download_video(args.url, args.quiet)
        elif args.function == 'transcribe_video_enhanced':
            video_path = test_download_video(args.url, True)
            test_transcribe_video(video_path, args.quiet)
        elif args.function == 'select_highlights':
            video_path = test_download_video(args.url, True)
            transcript_data = test_transcribe_video(video_path, True)
            test_select_highlights(transcript_data, args.url, args.clips, args.max_duration, args.quiet)
        elif args.function == 'smart_highlight_selector':
            video_path = test_download_video(args.url, True)
            transcript_data = test_transcribe_video(video_path, True)
            test_smart_highlight_selector(
                transcript_data, args.url, args.clips, args.min_duration, 
                args.max_duration, args.content_type, args.quiet
            )
        elif args.function == 'clip_video':
            video_path = test_download_video(args.url, True)
            transcript_data = test_transcribe_video(video_path, True)
            highlights = test_select_highlights(transcript_data, args.url, args.clips, args.max_duration, True)
            test_clip_video(video_path, highlights, args.quiet)
        elif args.function == 'create_smart_clips':
            video_path = test_download_video(args.url, True)
            transcript_data = test_transcribe_video(video_path, True)
            test_create_smart_clips(
                video_path, transcript_data, args.min_duration, args.max_duration, args.clips, args.quiet
            )
        elif args.function == 'generate_caption':
            video_path = test_download_video(args.url, True)
            transcript_data = test_transcribe_video(video_path, True)
            highlights = test_select_highlights(transcript_data, args.url, args.clips, args.max_duration, True)
            clips = test_clip_video(video_path, highlights, True)
            if clips:
                test_generate_caption(clips[0], transcript_data, args.url, args.quiet)
        elif args.function == 'validate_repair_video':
            video_path = test_download_video(args.url, True)
            test_validate_repair_video(video_path, args.quiet)
    except Exception as e:
        print(f"ERROR: Test execution failed: {str(e)}")

if __name__ == "__main__":
    main()
