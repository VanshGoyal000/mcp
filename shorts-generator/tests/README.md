# Modal Function Testing

This directory contains scripts for testing Modal functions in the shorts-generator project.

## Test Modal Functions Script

The `test_modal_functions.py` script allows you to test individual Modal functions or run all tests in sequence.

### Prerequisites

1. Ensure Modal is properly set up and configured
2. Make sure you've deployed the Modal app first:
   ```
   python ../modal_deploy.py
   ```

### Usage

Basic usage:

```bash
# Run all tests
python test_modal_functions.py

# Test a specific function
python test_modal_functions.py --function download_youtube_video

# Provide a custom YouTube URL
python test_modal_functions.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# Adjust clip parameters
python test_modal_functions.py --clips 3 --min-duration 20 --max-duration 45

# Change content type
python test_modal_functions.py --content-type funny
```

### Available Functions

- `setup_directories`: Test directory creation in Modal volume
- `download_youtube_video`: Test video downloading from YouTube
- `transcribe_video_enhanced`: Test video transcription
- `select_highlights`: Test basic highlight selection
- `smart_highlight_selector`: Test content-aware highlight selection
- `clip_video`: Test basic video clipping
- `create_smart_clips`: Test smart clip creation
- `generate_caption`: Test caption generation
- `validate_repair_video`: Test video validation and repair

### Example Commands

```bash
# Test downloading a video
python test_modal_functions.py -f download_youtube_video -u "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Test smart highlight selection for educational content
python test_modal_functions.py -f smart_highlight_selector --content-type educational

# Test creating 3 clips between 30-90 seconds
python test_modal_functions.py -f create_smart_clips -c 3 --min-duration 30 --max-duration 90

# Run all tests with minimal output
python test_modal_functions.py --quiet
```

## Troubleshooting

If you encounter issues:

1. Ensure Modal is correctly installed and configured
2. Make sure your Modal app is deployed before running tests
3. Check your API keys in the `.env` file
4. Look for "ERROR:" messages in the output for specific issues
5. Try running individual functions rather than all tests at once
