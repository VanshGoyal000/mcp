def extract_transcript_segment(transcript_data: dict, start_time: float, end_time: float) -> str:
    """
    Extract a segment of transcript between start_time and end_time.
    """
    segments = transcript_data["segments"]
    
    relevant_text = []
    for segment in segments:
        if segment["end"] >= start_time and segment["start"] <= end_time:
            relevant_text.append(segment["text"])
            
    return " ".join(relevant_text)
