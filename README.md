# 🎬 YouTube Shorts Generator — AI-Powered Video Highlights to Engaging Shorts

**Track:** `agent-demo-track`

## 🧠 Overview
Automatically generate engaging YouTube Shorts from longer videos using AI. This tool leverages advanced AI to identify the most compelling moments in any video, creating perfectly timed, content-aware clips optimized for social media engagement.

## 🎬 Demo Video
👉 [Click here to watch the demo]() *(Upload and replace this link)*  
![demo](![alt text](image.png)) *(Create and replace this thumbnail)*

## 🛠️ Features
- ✅ **Content-aware clips**: Generate variable-length clips (15-90 seconds) optimized for engagement
- ✅ **Smart highlight detection**: Automatically find the most engaging moments based on content type
- ✅ **Multiple content types**: Choose between interesting, funny, dramatic, educational, surprising, or inspiring moments
- ✅ **Automatic transcription**: Uses Whisper model to transcribe audio
- ✅ **Caption generation**: Create engaging titles and captions for your shorts
- ✅ **Support for YouTube URLs**: Process videos directly from YouTube

## 🧰 Tech Stack
- Python 3.9+
- Gradio UI Framework
- FFmpeg for video processing
- OpenAI Whisper for transcription
- OpenAI GPT for caption generation
- Modal for cloud processing
- Google Gemini (optional alternative)

## 🏗️ Architecture Diagram
![Architecture Diagram](https:///your-architecture-diagram.png) *(Create and replace this diagram)*

**Architecture Flow:**
1. **Input Processing** - YouTube URL or local video file is processed
2. **Transcription Engine** - Video audio is transcribed using Whisper
3. **Content Analysis** - AI analyzes transcript to identify key moments
4. **Smart Highlight Selection** - The most engaging segments are identified based on content type
5. **Video Processing** - FFmpeg creates optimized video clips
6. **Caption Generation** - Engaging captions and hashtags are generated for each clip
7. **Output Delivery** - Final shorts are delivered to the user

## 🧪 How to Use

### Prerequisites
- Python 3.9+
- FFmpeg installed and in your PATH
- OpenAI API key (for caption generation)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ```

### Running the Application

There are two ways to run the application:

1. **Local Processing (Recommended for beginners)**:
   ```bash
   python run.py
   ```
   This launches a Gradio web interface that processes videos locally.

2. **Modal Processing (More powerful, requires Modal account)**:
   ```bash
   python modal_deploy.py  # First deploy the Modal app
   python run_modal.py     # Then run the Gradio interface using Modal
   ```

## 📊 Stats / Results
- Average processing time: ~3-5 minutes per video
- Supports videos up to 3 hours long
- 6 different content types for targeted short creation
- Clip duration range: 15-90 seconds (customizable)

## 👤 Team
- Vansh Goyal (Lead Developer)
- github.com/VanshGoyal000
- linkedin.com/in/vanshcodeworks

## Usage Tips

1. For the best results, select the appropriate content type that matches your video (funny, educational, etc.)
2. Variable-length clips are more engaging than fixed-length clips
3. Adjust the min/max duration based on your platform requirements
4. Try different numbers of highlights to get the best moments from your video

## Troubleshooting

- If you see "ConnectionResetError" messages on Windows, these are harmless and can be ignored
- If video processing fails, try clearing the cache in `%TEMP%/shorts_generator_cache`
- Check that FFmpeg is properly installed and accessible from your PATH

## 🏆 Submission Tags
- `agent-demo-track`
- `Hackathon 2025`
i.imgur.com