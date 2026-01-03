import requests

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX04C6EN/Testing%20speech%20to%20text.mp3"
response = requests.get(url)

if response.status_code == 200:
    with open("downloaded_audio.mp3", "wb") as file:
        file.write(response.content)
    print("File downloaded successfully")

from faster_whisper import WhisperModel

# Use 'tiny' for speed or 'base' for a balance of speed/accuracy
model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("downloaded_audio.mp3", beam_size=5)

print("Transcription:")
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
