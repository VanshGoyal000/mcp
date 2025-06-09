HIGHLIGHT_SELECTION_PROMPT = """
You are an expert YouTube video editor. Your task is to identify the most engaging {num_highlights} moments from this transcript that would work well as variable-length YouTube Shorts ({min_duration}-{max_duration} seconds).

Original video title: "{video_title}"

Content type: {content_type}

Focus on finding {content_type} moments in the transcript.

Transcript chunk:
```
{transcript_chunk}
```

For each highlight, I need:
1. A short description of the key moment
2. A catchy title for the Short

Select moments that have:
- High emotional content
- Surprising revelations
- Key insights or advice
- Humor or entertaining elements
- Clear beginning and end points that work well as a standalone clip

Return your response in this JSON format:
```json
[
  {{
    "title": "Catchy title for the Short",
    "description": "Description of the key moment"
  }}
]
```

Only include the JSON in your response, no additional text.
"""
