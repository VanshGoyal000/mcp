import modal
# Update imports to be direct
from modal_config import app, get_base_image, volume, get_secrets
from prompts.caption_prompt import CAPTION_GENERATION_PROMPT

@app.function(
    image=get_base_image(),
    cpu=2,
    memory=2048,
    timeout=300,
    volumes={"/shorts-generator": volume},
    secrets=[get_secrets()]  # Use secrets list instead of secret parameter
)
def generate_caption(clip_info: dict, transcript_data: dict, video_title: str) -> dict:
    """
    Generate captions, titles, and hashtags for a video clip.
    Returns a dictionary with the generated content.
    """
    import openai
    import os
    
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Extract the transcript segment for this clip
    start_time = clip_info["start_time"]
    end_time = clip_info["end_time"]
    
    clip_transcript = ""
    for segment in transcript_data["segments"]:
        if segment["end"] >= start_time and segment["start"] <= end_time:
            clip_transcript += segment["text"] + " "
    
    # Create a prompt for caption generation
    prompt = CAPTION_GENERATION_PROMPT.format(
        video_title=video_title,
        clip_transcript=clip_transcript,
        clip_description=clip_info["description"]
    )
    
    # Call LLM to generate captions
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at creating engaging YouTube Shorts titles, captions, and hashtags."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=500
    )
    
    try:
        # Parse the response
        content = response.choices[0].message.content
        
        import json
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # Try to parse directly as JSON if no code block
            try:
                result = json.loads(content)
            except:
                # Manual parsing as fallback
                lines = content.split('\n')
                result = {
                    "title": next((line.split(':', 1)[1].strip() for line in lines if line.lower().startswith("title:")), f"Highlight from {video_title}"),
                    "caption": next((line.split(':', 1)[1].strip() for line in lines if line.lower().startswith("caption:")), clip_transcript[:100] + "..."),
                    "hashtags": next((line.split(':', 1)[1].strip() for line in lines if line.lower().startswith("hashtags:")), "#shorts #youtube #viral")
                }
                
        return {
            "title": result.get("title", f"Highlight from {video_title}"),
            "caption": result.get("caption", clip_transcript[:100] + "..."),
            "hashtags": result.get("hashtags", "#shorts #youtube #viral")
        }
        
    except Exception as e:
        print(f"Error parsing caption generation response: {e}")
        return {
            "title": f"Highlight from {video_title}",
            "caption": clip_transcript[:100] + "...",
            "hashtags": "#shorts #youtube #viral"
        }
