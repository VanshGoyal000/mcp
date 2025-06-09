import gradio as gr
import tempfile
import os
import uuid
import sys
import logging
from pathlib import Path
from utils.debug import debug_print, examine_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure system path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_app(use_modal=False):
    """Create and configure the Gradio app."""

    with gr.Blocks(title="YouTube Shorts Generator") as app:
        gr.Markdown("# 🎬 YouTube Shorts Generator")
        gr.Markdown("Upload a video or provide a YouTube URL to generate engaging Shorts!")

        with gr.Row():
            with gr.Column():
                # Input methods
                with gr.Tab("YouTube URL"):
                    youtube_url = gr.Textbox(
                        label="YouTube Video URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                    )
                    url_submit_btn = gr.Button("Process YouTube Video")

                with gr.Tab("Upload Video"):
                    video_input = gr.Video(label="Upload Video", interactive=True, format="mp4")
                    file_submit_btn = gr.Button("Process Uploaded Video")

                # Processing options
                num_highlights = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Highlights to Extract",
                )

                video_length = gr.Slider(
                    minimum=15,
                    maximum=90,
                    value=30,
                    step=5,
                    label="Maximum Length of Each Short (seconds)",
                )
                
                # Add content type selector
                content_types = [
                    "interesting", "funny", "dramatic", "educational", 
                    "surprising", "inspiring"
                ]
                
                content_type = gr.Dropdown(
                    choices=content_types,
                    value="interesting",
                    label="Content Type",
                    info="Select what kind of moments to extract"
                )
                
                with gr.Accordion("Content Type Descriptions", open=False):
                    content_descriptions = {
                        "interesting": "Intellectually engaging and thought-provoking moments",
                        "funny": "Humorous moments that will make viewers laugh",
                        "dramatic": "Powerful emotional moments with high tension or impact",
                        "educational": "Moments that teach valuable information clearly",
                        "surprising": "Unexpected twists or shocking revelations",
                        "inspiring": "Motivational or uplifting moments"
                    }
                    
                    gr.Markdown("\n".join([f"- **{k}**: {v}" for k, v in content_descriptions.items()]))
                
                min_length = gr.Slider(
                    minimum=10,
                    maximum=30,
                    value=15,
                    step=5,
                    label="Minimum Length of Each Short (seconds)",
                )

            with gr.Column():
                # Status and progress
                status = gr.Textbox(label="Status", interactive=False)
                progress = gr.Textbox(label="Progress", interactive=False)

                # Results
                results_md = gr.Markdown("## Generated Shorts Will Appear Here")

                results_gallery = gr.Gallery(
                    label="Generated Shorts",
                    elem_id="results_gallery",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )

                with gr.Accordion("Results Details", open=False):
                    title_results = gr.Textbox(label="Generated Titles")
                    caption_results = gr.Textbox(label="Generated Captions")
                    hashtag_results = gr.Textbox(label="Generated Hashtags")
        
        with gr.Accordion("How to use", open=True):
            gr.Markdown("""
            ### How to use this application
            
            1. **Input a video**: Either paste a YouTube URL or upload a video file
            2. **Configure settings**:
               - Number of Highlights: How many short clips to generate
               - Content Type: What kind of moments to look for (funny, educational, etc.)
               - Min/Max Length: The length range for your shorts (variable length based on content)
            3. **Click Process**: Wait while the system:
               - Downloads the video (if YouTube URL)
               - Transcribes the audio
               - Selects the best highlights based on your content type preference
               - Creates variable-length clips optimized for engagement
               - Generates titles and captions
            4. **View Results**: Play the generated clips and copy titles/captions
            
            **Note**: Processing can take a few minutes depending on video length
            """)

        # Functions for processing
        def process_youtube_url(url, num_clips, max_length, content_type, min_length=15):
            results = []

            try:
                # Update status
                yield "Starting local processing...", "", gr.update(), gr.update(), "", "", ""

                # Local processing implementation - bypassing Modal for now
                from utils.video import download_video_from_url

                # Download video
                yield "Downloading YouTube video...", "", gr.update(), gr.update(), "", "", ""
                try:
                    video_path, video_title = download_video_from_url(url)
                    yield f"Video downloaded: {video_title}", f"Content type: {content_type}, Length: {min_length}s to {max_length}s", gr.update(), gr.update(), "", "", ""
                except Exception as e:
                    logger.error(f"Error downloading video: {str(e)}")
                    yield f"Error downloading video: {str(e)}", "", gr.update(), gr.update(), "", "", ""
                    return

                # Generate results using local processing
                from local_processors import (
                    transcribe_video_locally,
                    select_highlights_locally, 
                    clip_video_locally,
                    generate_caption_locally
                )
                
                # Process steps
                yield f"Video downloaded: {video_title}", "Transcribing video...", gr.update(), gr.update(), "", "", ""
                transcript_data = transcribe_video_locally(video_path)
                
                yield f"Video downloaded: {video_title}", f"Selecting {content_type} highlights...", gr.update(), gr.update(), "", "", ""
                highlights = select_highlights_locally(transcript_data, video_title, num_clips, max_length)
                
                # Adjust highlight durations to be variable based on min/max length
                for highlight in highlights:
                    # Ensure minimum duration
                    if highlight["end_time"] - highlight["start_time"] < min_length:
                        highlight["end_time"] = highlight["start_time"] + min_length
                    
                    # Ensure maximum duration
                    if highlight["end_time"] - highlight["start_time"] > max_length:
                        highlight["end_time"] = highlight["start_time"] + max_length
                
                yield f"Video downloaded: {video_title}", f"Creating {content_type} clips ({min_length}s to {max_length}s)...", gr.update(), gr.update(), "", "", ""
                clip_infos = clip_video_locally(video_path, highlights, content_type)
                
                # After clipping videos
                yield f"Video downloaded: {video_title}", f"Video clips created! Generating captions...", gr.update(), gr.update(), "", "", ""

                # Generate captions for each clip
                all_titles = []
                all_captions = []
                all_hashtags = []
                videos_with_metadata = []

                print("\n=== PROCESSING VIDEO CLIPS ===")
                print(f"Number of clips: {len(clip_infos)}")

                for clip_info in clip_infos:
                    clip_path = clip_info["path"]
                    
                    # Verify clip exists and is valid
                    if not os.path.exists(clip_path):
                        print(f"WARNING: Clip doesn't exist: {clip_path}")
                        continue
                        
                    if os.path.getsize(clip_path) < 10000:  # Less than 10KB is suspicious
                        print(f"WARNING: Clip is too small: {clip_path} ({os.path.getsize(clip_path)} bytes)")
                        continue
                    
                    print(f"\nProcessing clip: {clip_path}")
                    print(f"Size: {os.path.getsize(clip_path)} bytes")
                    
                    caption_data = generate_caption_locally(clip_info, transcript_data, video_title)
                    
                    # Add to results
                    print(f"Adding clip to results with title: {caption_data['title']}")
                    videos_with_metadata.append(
                        (clip_path, caption_data["title"])
                    )
                    
                    all_titles.append(caption_data["title"])
                    all_captions.append(caption_data["caption"])
                    all_hashtags.append(caption_data["hashtags"])

                print(f"\nFinal number of valid clips: {len(videos_with_metadata)}")
                print("=== END PROCESSING VIDEO CLIPS ===\n")

                # Display results
                joined_titles = "\n\n".join(all_titles)
                joined_captions = "\n\n".join(all_captions)
                joined_hashtags = "\n\n".join(all_hashtags)
                
                yield (
                    f"Video processed: {video_title}", 
                    f"Generated {len(videos_with_metadata)} shorts!",
                    gr.update(value=videos_with_metadata),
                    gr.update(visible=True),
                    joined_titles,
                    joined_captions,
                    joined_hashtags
                )

            except Exception as e:
                logger.error(f"Error in processing: {str(e)}")
                yield f"Error: {str(e)}", "", gr.update(), gr.update(), "", "", ""

        def process_uploaded_video(video_file, num_clips, max_length, content_type, min_length=15):
            if video_file is None:
                return "No video uploaded", "", gr.update(), gr.update(), "", "", ""

            try:
                # Save the uploaded video to a path
                from utils.video import local_video_to_path
                video_path = local_video_to_path(video_file)
                video_title = f"Uploaded Video"

                # Use the same processing pipeline as YouTube videos
                for result in process_youtube_url(None, num_clips, max_length, content_type, min_length, _video_path=video_path, _video_title=video_title):
                    yield result

            except Exception as e:
                yield f"Error processing uploaded video: {str(e)}", "", gr.update(), gr.update(), "", "", ""

        # Connect the UI elements to the processing functions including new parameters
        url_submit_btn.click(
            process_youtube_url,
            inputs=[youtube_url, num_highlights, video_length, content_type, min_length],
            outputs=[status, progress, results_gallery, results_md, title_results, caption_results, hashtag_results],
        )

        file_submit_btn.click(
            process_uploaded_video,
            inputs=[video_input, num_highlights, video_length, content_type, min_length],
            outputs=[status, progress, results_gallery, results_md, title_results, caption_results, hashtag_results],
        )

    return app
