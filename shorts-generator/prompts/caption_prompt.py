CAPTION_GENERATION_PROMPT = """
You are an expert at creating engaging YouTube Shorts content. I need you to create a title, caption, and hashtags for a short video clip.

Original video title: "{video_title}"

Clip description: {clip_description}

Transcript of the clip:
```
{clip_transcript}
```

Please create:
1. A catchy title (60 characters max) with emojis if appropriate
2. An engaging caption (150 characters max)
3. 5-7 relevant hashtags

Return your response in this JSON format:
```json
{{
  "title": "Your catchy title here",
  "caption": "Your engaging caption here",
  "hashtags": "#shorts #youtube #topic1 #topic2"
}}
```

Only include the JSON in your response, no additional text.
"""
