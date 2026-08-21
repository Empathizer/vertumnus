"""
Local websocket control server. The Tauri/React frontend (and, for testing,
any plain websocket client) talks to this to control the AudioPipeline:
device selection, voice model loading, live effect params, monitor/mute,
presets, and a periodic latency readout.

Run: python -m server.ws_server
Listens on ws://127.0.0.1:8765
"""
import os

# Some combination of torch/faiss/numba on macOS bundles more than one
# OpenMP runtime, which aborts the process at import time ("OMP: Error #15").
# Must be set before those libraries are imported anywhere below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Confirmed empirically: calling into faiss/BLAS from a background thread
# (our audio worker thread, separate from the main/asyncio thread — see
# audio/pipeline.py) segfaults reliably as soon as a *second* thread first
# touches these OpenMP-linked libraries, even with faiss.omp_set_num_threads
# set. Forcing every OpenMP-linked library to never spin up a thread pool at
# all avoids the race. Must be set before numpy/torch/faiss are imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import websockets

import presets as presets_module
import test_audio_files
import training as training_module
import voice_models
from audio.devices import list_input_devices, list_output_devices
from audio.pipeline import AudioPipeline, PipelineError, find_virtual_mic_device
from gpu_device import select_torch_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

HOST = "127.0.0.1"
PORT = 8765
LATENCY_BROADCAST_INTERVAL = 0.25

pipeline = AudioPipeline(select_torch_device())
clients: set = set()
main_loop: asyncio.AbstractEventLoop | None = None


def _on_file_finished() -> None:
    # Called from the file-feeder thread (not the asyncio thread) — hop
    # back onto the event loop to broadcast, and stop the pipeline so
    # "running" state reflects reality.
    if main_loop is None:
        return

    def _handle():
        pipeline.stop()
        asyncio.create_task(broadcast({"type": "status", "running": False}))
        asyncio.create_task(broadcast({"type": "file_playback_finished"}))

    main_loop.call_soon_threadsafe(_handle)


pipeline.on_file_finished = _on_file_finished

training_manager = training_module.TrainingManager()


def _on_training_progress(progress: "training_module.TrainingProgress") -> None:
    # Called from the training thread — hop back onto the event loop.
    if main_loop is None:
        return

    def _handle():
        asyncio.create_task(broadcast({
            "type": "training_progress",
            "stage": progress.stage,
            "message": progress.message,
            "fraction": progress.fraction,
        }))
        if progress.stage == "done":
            asyncio.create_task(broadcast(voice_models_payload()))

    main_loop.call_soon_threadsafe(_handle)


def devices_payload() -> dict:
    return {
        "type": "devices",
        "input": [d.__dict__ for d in list_input_devices()],
        "output": [d.__dict__ for d in list_output_devices()],
        "suggested_virtual_mic": find_virtual_mic_device(),
    }


def voice_models_payload() -> dict:
    return {"type": "voice_models", "models": [m.__dict__ for m in voice_models.list_voice_models()]}


def presets_payload() -> dict:
    return {"type": "presets", "names": presets_module.list_presets()}


def test_audio_payload() -> dict:
    return {"type": "test_audio_files", "files": test_audio_files.list_test_audio_files()}


def error_payload(kind: str, message: str) -> dict:
    return {"type": "error", "kind": kind, "message": message}


async def handle_message(ws, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await ws.send(json.dumps(error_payload("unknown", "Malformed JSON message")))
        return

    msg_type = msg.get("type")
    try:
        if msg_type == "list_devices":
            await ws.send(json.dumps(devices_payload()))

        elif msg_type == "list_voice_models":
            await ws.send(json.dumps(voice_models_payload()))

        elif msg_type == "list_presets":
            await ws.send(json.dumps(presets_payload()))

        elif msg_type == "start":
            input_device = msg["input_device"]
            output_device = msg["output_device"]
            monitor_device = msg.get("monitor_device")
            pipeline.start(input_device, output_device, monitor_device)
            await broadcast({"type": "status", "running": True})

        elif msg_type == "stop":
            pipeline.stop()
            await broadcast({"type": "status", "running": False})

        elif msg_type == "list_test_audio":
            await ws.send(json.dumps(test_audio_payload()))

        elif msg_type == "start_from_file":
            file_path = msg["file_path"]
            output_device = msg["output_device"]
            monitor_device = msg.get("monitor_device")
            pipeline.start_from_file(file_path, output_device, monitor_device)
            await broadcast({"type": "status", "running": True, "file_input": True})

        elif msg_type == "start_training":
            try:
                training_manager.start(
                    source_file_path=msg["source_file_path"],
                    voice_name=msg["voice_name"],
                    epochs=msg.get("epochs", 20),
                    on_progress=_on_training_progress,
                )
            except training_module.TrainingError as e:
                await ws.send(json.dumps(error_payload("model_not_found", str(e))))

        elif msg_type == "cancel_training":
            training_manager.cancel()

        elif msg_type == "load_voice":
            pipeline.load_voice(
                pth_path=msg["pth_path"],
                engine=msg.get("engine", "rvc"),
                index_path=msg.get("index_path"),
                config_path=msg.get("config_path"),
                index_rate=msg.get("index_rate", 0.75),
                f0_up_key=msg.get("f0_up_key", 0),
                f0_method=msg.get("f0_method", "harvest"),
                block_time=msg.get("block_time"),
                crossfade_time=msg.get("crossfade_time"),
                extra_time=msg.get("extra_time"),
                infer_step=msg.get("infer_step", 30),
                t_start=msg.get("t_start", 0.7),
                sampling_method=msg.get("sampling_method", "euler"),
            )
            await broadcast({"type": "voice_loaded", "pth_path": msg["pth_path"]})

        elif msg_type == "unload_voice":
            pipeline.unload_voice()
            await broadcast({"type": "voice_unloaded"})

        elif msg_type == "set_pitch":
            pipeline.set_voice_pitch(msg["f0_up_key"])

        elif msg_type == "update_effects":
            params = {k: v for k, v in msg.items() if k != "type"}
            pipeline.update_effects(**params)

        elif msg_type == "set_monitor":
            if msg.get("enabled"):
                pipeline.enable_monitor(msg["device"])
            else:
                pipeline.disable_monitor()

        elif msg_type == "set_mute":
            pipeline.muted = bool(msg.get("muted", False))

        elif msg_type == "save_preset":
            data = {
                "effects": pipeline.effect_chain.params.__dict__ if pipeline.effect_chain else {},
                "voice_pth": pipeline.voice.pth_path if pipeline.voice else None,
                "f0_up_key": pipeline.converter.f0_up_key if pipeline.converter else 0,
            }
            presets_module.save_preset(msg["name"], data)
            await ws.send(json.dumps(presets_payload()))

        elif msg_type == "load_preset":
            data = presets_module.load_preset(msg["name"])
            if data.get("effects"):
                pipeline.update_effects(**data["effects"])
            await ws.send(json.dumps({"type": "preset_loaded", "name": msg["name"], "data": data}))

        elif msg_type == "delete_preset":
            presets_module.delete_preset(msg["name"])
            await ws.send(json.dumps(presets_payload()))

        else:
            await ws.send(json.dumps(error_payload("unknown", f"Unknown message type: {msg_type}")))

    except PipelineError as e:
        await ws.send(json.dumps(error_payload(e.kind, e.message)))
    except KeyError as e:
        await ws.send(json.dumps(error_payload("unknown", f"Missing field: {e}")))
    except FileNotFoundError as e:
        await ws.send(json.dumps(error_payload("model_not_found", str(e))))
    except Exception as e:
        logger.exception("Unhandled error processing message")
        await ws.send(json.dumps(error_payload("unknown", str(e))))


async def broadcast(payload: dict) -> None:
    if not clients:
        return
    data = json.dumps(payload)
    await asyncio.gather(*(c.send(data) for c in clients), return_exceptions=True)


async def latency_broadcaster() -> None:
    while True:
        await asyncio.sleep(LATENCY_BROADCAST_INTERVAL)
        if clients:
            await broadcast({
                "type": "latency",
                "model_ms": pipeline.latency.model_ms,
                "total_ms": pipeline.latency.total_ms,
                "underruns": pipeline.underrun_count,
            })


async def handler(ws) -> None:
    clients.add(ws)
    logger.info("Client connected (%d total)", len(clients))
    try:
        await ws.send(json.dumps({"type": "hello", "device": str(pipeline.torch_device)}))
        async for raw in ws:
            await handle_message(ws, raw)
    finally:
        clients.discard(ws)
        logger.info("Client disconnected (%d total)", len(clients))


async def main() -> None:
    global main_loop
    main_loop = asyncio.get_running_loop()
    asyncio.create_task(latency_broadcaster())
    async with websockets.serve(handler, HOST, PORT):
        logger.info("Vertumnus backend listening on ws://%s:%d (device=%s)", HOST, PORT, pipeline.torch_device)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
