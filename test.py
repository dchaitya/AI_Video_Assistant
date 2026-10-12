
from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all




source="https://www.youtube.com/watch?v=RVaWve3a3w8"
language='hindi'

chunks=process_input(source)
transcript=transcribe_all(chunks,lan=language)

print(transcript)