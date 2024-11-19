import sys
import os

# Add local bin directory to PATH
os.environ["PATH"] += os.pathsep + "/home/appuser/.local/bin"

import subprocess
try:
    import transformers
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import subprocess
import sys


def main(desired_length):
    try:
        # Read the YouTube video ID from file
        with open('texts/video_id.txt', 'r') as file:
            youtube_video = file.read().strip()

        # Validate the video ID format
        video_id = youtube_video.split("=")[-1]
        print(f"Video ID: {video_id}")

        # Fetch the transcript for the video
        print("Fetching transcript...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = " ".join([t['text'] for t in transcript])
        print(f"Transcript Length: {len(result)} characters.")

        # Initialize the summarizer pipeline
        print("Initializing summarizer...")
        summarizer = pipeline('summarization')

        # Calculate how many iterations are needed for large texts
        num_iters = len(result) // 1000 + 1
        summarized_text = []

        # Summarize in chunks
        for i in range(num_iters):
            start, end = i * 1000, (i + 1) * 1000
            out = summarizer(result[start:end], max_length=desired_length, min_length=desired_length//2, do_sample=False)[0]['summary_text']
            summarized_text.append(out)
            print(f"Chunk {i + 1}/{num_iters} summarized.")

        # Combine the summarized text
        final_summary = " ".join(summarized_text)
        print("Final summary generated.")

        # Format the summary in bullet points
        bullet_points = final_summary.split('. ')
        formatted_summary = "\n".join([f"- {point.strip()}" for point in bullet_points if point])

        # Save the formatted summary to a file
        with open('texts/summary.txt', 'w') as file:
            file.write(formatted_summary)
        print("Summary saved to 'pages/texts/summary.txt'.")

    except Exception as e:
        print(f"Error occurred: {e}", file=sys.stderr)
        print("Debugging information:")
        if 'video_id' in locals():
            print(f"Video ID: {video_id}")
        if 'result' in locals():
            print(f"Transcript Length: {len(result)}")
        print(f"Desired Length: {desired_length}")
        sys.exit(1)

if __name__ == "__main__":
    # Get the desired summary length from command-line arguments
    desired_length = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(desired_length)
