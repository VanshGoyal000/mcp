import modal
from modal_config import app, get_base_image, volume, get_secrets
from prompts.highlight_prompt import HIGHLIGHT_SELECTION_PROMPT

@app.function(
    image=get_base_image(),
    cpu=2,
    memory=4096,
    timeout=300,
    secrets=[get_secrets()]  # Use get_secrets function instead of secrets directly
)
def select_highlights(transcript_data: dict, video_title: str, num_highlights: int = 3, segment_length: int = 60) -> list:
    """
    Select the most engaging highlights from a transcript.
    Returns a list of dictionaries with start_time, end_time, and description.
    """
    import openai
    import os
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Process transcript into 5-minute chunks with overlap
    full_text = transcript_data["full_text"]
    segments = transcript_data["segments"]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", "?", "!"]
    )
    
    chunks = text_splitter.split_text(full_text)
    
    # For each chunk, find potential highlights
    all_potential_highlights = []
    
    for i, chunk in enumerate(chunks):
        # Create a prompt for highlight selection
        prompt = HIGHLIGHT_SELECTION_PROMPT.format(
            video_title=video_title,
            transcript_chunk=chunk,
            num_highlights=min(num_highlights, 2),  # Request fewer per chunk
            segment_length=segment_length
        )
        
        # Call LLM to identify highlights
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert video editor who can identify the most engaging parts of a video transcript."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        try:
            # Parse the response to extract highlights
            highlights_text = response.choices[0].message.content
            import json
            import re
            
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', highlights_text)
            if json_match:
                highlights_json = json.loads(json_match.group(1))
            else:
                # Fallback to trying to parse the entire response as JSON
                highlights_json = json.loads(highlights_text)
                
            all_potential_highlights.extend(highlights_json)
        except Exception as e:
            print(f"Error parsing highlights from chunk {i}: {e}")
            continue
    
    # Now match the highlights with actual timestamps from segments
    final_highlights = []
    
    for highlight in all_potential_highlights[:num_highlights]:
        # Find segments that contain this text
        highlight_text = highlight.get("description", "")
        best_match_segment = None
        highest_similarity = 0
        
        for segment in segments:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, segment["text"].lower(), highlight_text.lower()).ratio()
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_segment = segment
        
        if best_match_segment:
            # Calculate start and end times for a segment_length clip
            start_time = best_match_segment["start"]
            end_time = min(start_time + segment_length, best_match_segment["end"])
            
            final_highlights.append({
                "start_time": start_time,
                "end_time": end_time,
                "description": highlight.get("description", ""),
                "title": highlight.get("title", f"Highlight at {int(start_time//60)}:{int(start_time%60):02d}")
            })
    
    return final_highlights
