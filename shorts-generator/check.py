from modal_deploy import app
# Load the app and function
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=cOiH2C6-p1Y"
    with app.run():
        result = download_youtube_video.remote(youtube_url)
        print("Result:", result)
