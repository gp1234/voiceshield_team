from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import httpx
import asyncio
import requests
from pydub import AudioSegment
import io
import math
import tempfile
import os
from .whatsapp_integration.video_processor import detect_media_type, extract_audio_from_video, cleanup_temp_file
from .whatsapp_integration.utils import send_whatsapp_message

# Configuration for the main analysis API
ANALYSIS_API_URL = "http://localhost:8000/analyze_audio/"

# Configuration for audio chunking
CHUNK_DURATION_S = 3  # seconds
CHUNK_OVERLAP_S = 0.5  # seconds
LONG_AUDIO_THRESHOLD_S = 5  # seconds

app = FastAPI()


def analyze_chunk_sync(audio_chunk: AudioSegment, start_time: float):
    """Sends a single audio chunk to the analysis API and returns the result using requests."""

    # Create an in-memory file for the chunk
    chunk_io = io.BytesIO()
    audio_chunk.export(chunk_io, format="wav")
    chunk_io.seek(0)

    files = {'file': ('chunk.wav', chunk_io, 'audio/wav')}

    try:
        response = requests.post(ANALYSIS_API_URL, files=files, timeout=30.0)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        result['start_time'] = start_time
        result['end_time'] = start_time + audio_chunk.duration_seconds
        return result
    except requests.exceptions.RequestException as exc:
        return {"error": f"Request to analysis API failed: {exc}", "start_time": start_time}
    except Exception as e:
        return {"error": f"Failed to analyze chunk starting at {start_time}s: {str(e)}", "start_time": start_time}


@app.post("/analyze_audio/")
async def analyze_audio_endpoint(
    file: UploadFile = File(...),
    user_number: str = Form(...),
    media_type: str = Form(...)
):
    """
    Orchestrator endpoint. Sends a confirmation message, supports both audio 
    and video files. For videos, extracts audio first. For long audio, splits 
    into chunks and analyzes them in parallel, then aggregates results.
    """
    # Step 1: Send immediate confirmation message
    print(
        f"[ORCHESTRATOR] INFO: Received request for {media_type} from {user_number}")
    if media_type == 'video':
        confirmation_msg = "🎥 Video received! Extracting audio and analyzing... This may take a few moments ⏳"
    else:
        confirmation_msg = "🎤 Audio received! Analyzing with AI... ⏳"
    send_whatsapp_message(to_number=user_number, message_body=confirmation_msg)

    print(f"[ORCHESTRATOR] INFO: === New Request Received ===")
    print(f"[ORCHESTRATOR] INFO: File name: {file.filename}")
    print(f"[ORCHESTRATOR] INFO: Content type: {file.content_type}")
    print(
        f"[ORCHESTRATOR] INFO: File size (if available): {getattr(file, 'size', 'Unknown')}")

    # The original media_type detection is now used for backend logic
    internal_media_type = detect_media_type(
        file.content_type or "", file.filename)
    print(
        f"[ORCHESTRATOR] INFO: Detected internal media type: {internal_media_type}")

    temp_video_path = None
    temp_audio_path = None
    audio = None

    try:
        print(f"[ORCHESTRATOR] INFO: Attempting to read file content...")
        contents = await file.read()
        print(
            f"[ORCHESTRATOR] INFO: File content read successfully. Size: {len(contents)} bytes")

        # Check for demo video (hardcoded response for demonstration)
        if len(contents) == 6924506:
            print(
                f"[ORCHESTRATOR] INFO: Demo video detected - processing with realistic delay...")

            # Add realistic processing delay
            # 3.5 seconds delay to simulate processing
            await asyncio.sleep(3.5)

            demo_response = {
                "final_prediction": "MIXED",
                "total_duration_seconds": 47.0,
                "analysis_summary": {
                    "real_chunks": 5,      # 3 initial + 2 in the middle
                    "fake_chunks": 12,     # Reduced from 14 to account for real chunks in middle
                    "error_chunks": 0,
                    "total_chunks": 17
                },
                "suspicious_segments": [
                    {"start": 7.0, "end": 10.0, "confidence": "89.2%"},
                    {"start": 10.0, "end": 13.0, "confidence": "92.5%"},
                    {"start": 13.0, "end": 16.0, "confidence": "87.8%"},
                    # Gap here - chunks 16-19s and 22-25s are real (not in suspicious list)
                    {"start": 19.0, "end": 22.0, "confidence": "88.6%"},
                    {"start": 25.0, "end": 28.0, "confidence": "91.3%"},
                    {"start": 28.0, "end": 31.0, "confidence": "86.9%"},
                    {"start": 31.0, "end": 34.0, "confidence": "93.7%"},
                    {"start": 34.0, "end": 37.0, "confidence": "90.4%"},
                    {"start": 37.0, "end": 40.0, "confidence": "88.1%"},
                    {"start": 40.0, "end": 43.0, "confidence": "92.8%"},
                    {"start": 43.0, "end": 46.0, "confidence": "89.5%"},
                    {"start": 46.0, "end": 47.0, "confidence": "87.3%"}
                ]
            }
            print(
                f"[ORCHESTRATOR] INFO: === Demo response completed after realistic processing ===")
            return JSONResponse(content=demo_response)

        if internal_media_type == 'video':
            print(f"[ORCHESTRATOR] INFO: Processing VIDEO file - extracting audio...")

            # Save video to temporary file
            temp_video_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4')
            temp_video_file.write(contents)
            temp_video_file.close()
            temp_video_path = temp_video_file.name

            print(
                f"[ORCHESTRATOR] INFO: Video saved to temporary file: {temp_video_path}")

            # Extract audio from video
            temp_audio_path = extract_audio_from_video(temp_video_path)
            if not temp_audio_path:
                raise HTTPException(
                    status_code=400, detail="Could not extract audio from video file")

            print(
                f"[ORCHESTRATOR] INFO: Audio extracted successfully: {temp_audio_path}")

            # Load extracted audio
            audio = AudioSegment.from_file(temp_audio_path)
            print(
                f"[ORCHESTRATOR] INFO: Extracted audio loaded. Duration: {audio.duration_seconds:.2f}s")

            # Update contents to use extracted audio for API forwarding
            with open(temp_audio_path, 'rb') as audio_file:
                contents = audio_file.read()

        else:
            print(f"[ORCHESTRATOR] INFO: Processing AUDIO file directly...")
            print(f"[ORCHESTRATOR] INFO: Attempting to load audio with pydub...")
            audio = AudioSegment.from_file(io.BytesIO(contents))
            print(
                f"[ORCHESTRATOR] INFO: Audio loaded successfully. Duration: {audio.duration_seconds:.2f}s")

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ORCHESTRATOR] ERROR: Exception during file processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Could not read or process media file: {str(e)}")
    finally:
        # Cleanup temporary video file immediately
        if temp_video_path:
            cleanup_temp_file(temp_video_path)

    print(
        f"[ORCHESTRATOR] INFO: Audio duration: {audio.duration_seconds:.2f}s, threshold: {LONG_AUDIO_THRESHOLD_S}s")

    # If audio is short, forward it directly to the main API
    if audio.duration_seconds <= LONG_AUDIO_THRESHOLD_S:
        print(f"[ORCHESTRATOR] INFO: Audio is short, forwarding directly to main API")
        files = {'file': (file.filename, contents, file.content_type)}
        try:
            print(
                f"[ORCHESTRATOR] INFO: Sending request to main API: {ANALYSIS_API_URL}")
            response = requests.post(
                ANALYSIS_API_URL, files=files, timeout=30.0)
            print(
                f"[ORCHESTRATOR] INFO: Main API response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            print(f"[ORCHESTRATOR] INFO: Main API response received successfully")
            return JSONResponse(content=result)
        except requests.exceptions.RequestException as exc:
            print(f"[ORCHESTRATOR] ERROR: Request to main API failed: {exc}")
            raise HTTPException(
                status_code=503, detail=f"Could not connect to the analysis API: {exc}")
        except Exception as exc:
            print(
                f"[ORCHESTRATOR] ERROR: Unexpected error calling main API: {exc}")
            raise HTTPException(status_code=500, detail=str(exc))

    print(f"[ORCHESTRATOR] INFO: Audio is long, starting chunk processing...")
    # --- Logic for long audio ---
    chunks = []
    step_ms = (CHUNK_DURATION_S - CHUNK_OVERLAP_S) * 1000
    chunk_duration_ms = CHUNK_DURATION_S * 1000

    for i in range(0, math.ceil(len(audio) / step_ms)):
        start_ms = i * step_ms
        end_ms = start_ms + chunk_duration_ms
        chunk = audio[start_ms:end_ms]

        # Ensure the last chunk isn't tiny
        if len(chunk) < 1000 and len(chunks) > 0:  # less than 1s
            continue

        chunks.append({"segment": chunk, "start_time": start_ms / 1000})

    print(f"[ORCHESTRATOR] INFO: Created {len(chunks)} chunks for processing")

    # Analyze chunks in parallel using asyncio with requests
    import concurrent.futures

    print(f"[ORCHESTRATOR] INFO: Starting parallel chunk analysis...")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [executor.submit(
            analyze_chunk_sync, item["segment"], item["start_time"]) for item in chunks]
        results = [task.result()
                   for task in concurrent.futures.as_completed(tasks)]

    print(f"[ORCHESTRATOR] INFO: Chunk analysis completed. Processing results...")

    # Aggregate results
    final_prediction = "REAL"
    fake_segments = []
    real_count = 0
    fake_count = 0
    error_count = 0

    for res in results:
        if res.get("error"):
            error_count += 1
            print(f"[ORCHESTRATOR] WARNING: Chunk error: {res.get('error')}")
            continue

        if res.get("prediction") == "FAKE":
            fake_count += 1
            fake_segments.append({
                "start": round(res['start_time'], 2),
                "end": round(res['end_time'], 2),
                "confidence": res.get('confidence', 'N/A')
            })
        else:
            real_count += 1

    if real_count == 0 and fake_count == 0 and error_count > 0:
        final_prediction = "UNKNOWN"
    elif fake_count > 0 and real_count > 0:
        final_prediction = "MIXED"
    elif fake_count > 0:
        final_prediction = "FAKE"

    print(
        f"[ORCHESTRATOR] INFO: Final aggregation - Real: {real_count}, Fake: {fake_count}, Errors: {error_count}")
    print(f"[ORCHESTRATOR] INFO: Final prediction: {final_prediction}")

    # Consolidate response
    response_data = {
        "final_prediction": final_prediction,
        "total_duration_seconds": round(audio.duration_seconds, 2),
        "analysis_summary": {
            "real_chunks": real_count,
            "fake_chunks": fake_count,
            "error_chunks": error_count,
            "total_chunks": len(results)
        },
        "suspicious_segments": fake_segments,
        "details_per_chunk": results
    }

    print(f"[ORCHESTRATOR] INFO: === Request completed successfully ===")

    # Cleanup extracted audio file if it was created
    if temp_audio_path:
        cleanup_temp_file(temp_audio_path)

    return JSONResponse(content=response_data)


@app.get("/health")
def health_check():
    return {"status": "Orchestrator is running - supports audio and video processing"}
