import modal
import gradio as gr
import os
import tempfile
import sys
from dotenv import load_dotenv

# Configure environment
load_dotenv()

# Define clip content types
CONTENT_TYPES = {
    "interesting": "Identify the most intellectually engaging and thought-provoking moments",
    "funny": "Find the most humorous and entertaining moments that will make viewers laugh",
    "dramatic": "Locate powerful emotional moments with high tension or impact",
    "educational": "Select moments that teach valuable information or explain concepts clearly",
    "surprising": "Find unexpected twists, revelations or shocking moments",
    "inspiring": "Identify motivational or uplifting moments that inspire viewers"
}

def process_video(youtube_url, num_clips=3, min_duration=15, max_duration=60, content_type="interesting"):
    try:
        # Import from modal_deploy
        import modal
        from modal_deploy import app
        
        status_text = f"Processing video: {youtube_url}\n"
        status_text += f"Content type: {content_type}\n"
        status_text += f"Duration range: {min_duration}s to {max_duration}s\n"
        
        # 1. Download video
        status_text += "Downloading video...\n"
        result = app.download_youtube_video.call(youtube_url)  # Changed from .remote() to .call()
        
        video_path = result[0] if isinstance(result, tuple) else result.get("path")
        video_title = result[1] if isinstance(result, tuple) else result.get("title", "Unknown Title")
        
        if not video_path:
            return f"Download failed: {video_title}"  # Video title contains error message
        
        status_text += f"Video downloaded: {video_title}\n"
        
        # 2. Transcribe video
        status_text += "Transcribing video...\n"
        transcript_result = app.transcribe_video_enhanced.call(video_path)  # Changed from .remote() to .call()
        
        # 3. Select highlights based on content type
        status_text += f"Selecting {content_type} highlights...\n"
        
        # Check available functions
        try:
            # Try calling smart highlight selector
            status_text += "Using smart content-based highlight selection...\n"
            highlights = app.smart_highlight_selector.call(  # Changed from .remote() to .call()
                transcript_result, video_title, num_clips, min_duration, max_duration, content_type)
        except Exception as e:
            status_text += f"Using basic highlight selection...\n"
            highlights = app.select_highlights.call(  # Changed from .remote() to .call()
                transcript_result, video_title, num_clips, max_duration)
        
        # 4. Create variable-length clips
        status_text += "Creating variable-length smart clips...\n"
        
        # Try smart clips
        try:
            clips = app.create_smart_clips.call(  # Changed from .remote() to .call()
                video_path, transcript_result, min_duration, max_duration, num_clips)
        except Exception:
            # Fall back to regular clip function
            clips = app.clip_video.call(video_path, highlights)  # Changed from .remote() to .call()
        
        # 5. Generate captions
        status_text += "Generating engaging captions...\n"
        captions = []
        
        for clip_info in clips:
            try:
                caption = app.generate_caption.call(  # Changed from .remote() to .call()
                    clip_info, transcript_result, video_title)
                captions.append(caption)
            except Exception as e:
                status_text += f"Caption generation error: {str(e)}\n"
        
        
        # Generate summary
        status_text += f"Processing complete! Created {len(clips) if isinstance(clips, list) else '?'} clips.\n\n"
        
        # Add clip details
        if isinstance(clips, list):
            status_text += "CLIPS CREATED:\n"
            for i, clip in enumerate(clips):
                duration = clip.get("end_time", 0) - clip.get("start_time", 0)
                status_text += f"Clip {i+1}: {clip.get('title', 'Untitled')} ({duration:.1f}s)\n"
                
                # Add caption info if available
                if i < len(captions):
                    status_text += f"   Title: {captions[i].get('title', 'No title')}\n"
                    status_text += f"   Caption: {captions[i].get('caption', 'No caption')[:50]}...\n"
                    status_text += f"   Hashtags: {captions[i].get('hashtags', '#shorts')}\n"
                
                status_text += "\n"
        
        return status_text
        
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="YouTube Shorts Generator") as demo:
    gr.Markdown("# 🎬 YouTube Shorts Generator")
    gr.Markdown("Generate high-quality, variable-length shorts from long videos automatically!")
    
    with gr.Row():
        with gr.Column():
            youtube_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            
            with gr.Row():
                num_clips = gr.Slider(1, 5, value=2, step=1, label="Number of clips")
                
            with gr.Row():
                min_duration = gr.Slider(10, 30, value=15, step=5, label="Min duration (seconds)")
                max_duration = gr.Slider(30, 90, value=45, step=5, label="Max duration (seconds)")
            
            content_type = gr.Radio(
                list(CONTENT_TYPES.keys()), 
                value="interesting",
                label="Content Type",
                info="Select what kind of moments to extract"
            )
            
            content_description = gr.Markdown(value=f"**Type Info:** {CONTENT_TYPES['interesting']}")
            
            process_btn = gr.Button("Generate Shorts", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(label="Status", lines=15)
    
    # Update content description when type changes
    def update_description(content_type):
        return f"**Type Info:** {CONTENT_TYPES[content_type]}"
    
    content_type.change(update_description, inputs=[content_type], outputs=[content_description])
    
    # Process when button is clicked
    process_btn.click(
        process_video, 
        inputs=[youtube_url, num_clips, min_duration, max_duration, content_type],
        outputs=output
    )

# Run the app
if __name__ == "__main__":
    demo.launch()