from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import httpx
import asyncio
import requests
from pydub import AudioSegment
import io
import math

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
async def analyze_audio_endpoint(file: UploadFile = File(...)):
    """
    Orchestrator endpoint. Forwards short audio to the main API directly.
    For long audio, it splits it into chunks, analyzes them in parallel,
    and aggregates the results.
    """

    print(f"[ORCHESTRATOR] INFO: === New Request Received ===")
    print(f"[ORCHESTRATOR] INFO: File name: {file.filename}")
    print(f"[ORCHESTRATOR] INFO: Content type: {file.content_type}")
    print(
        f"[ORCHESTRATOR] INFO: File size (if available): {getattr(file, 'size', 'Unknown')}")

    try:
        print(f"[ORCHESTRATOR] INFO: Attempting to read file content...")
        contents = await file.read()
        print(
            f"[ORCHESTRATOR] INFO: File content read successfully. Size: {len(contents)} bytes")

        print(f"[ORCHESTRATOR] INFO: Attempting to load audio with pydub...")
        audio = AudioSegment.from_file(io.BytesIO(contents))
        print(
            f"[ORCHESTRATOR] INFO: Audio loaded successfully. Duration: {audio.duration_seconds:.2f}s")

    except Exception as e:
        print(f"[ORCHESTRATOR] ERROR: Exception during file processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=400, detail=f"Could not read or process audio file: {str(e)}")

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
    return JSONResponse(content=response_data)


@app.get("/health")
def health_check():
    return {"status": "Orchestrator is running"}
