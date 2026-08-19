// Thin websocket client for the local Python backend (audio/RVC engine).
// Protocol is plain JSON messages, defined in backend/server/ws_server.py.

export interface DeviceInfo {
  index: number;
  name: string;
  max_input_channels: number;
  max_output_channels: number;
  default_samplerate: number;
  host_api: string;
}

export interface VoiceModelInfo {
  name: string;
  engine: "rvc" | "ddsp";
  pth_path: string;
  index_path: string | null;
  config_path: string | null;
}

export interface EffectParams {
  pitch_semitones: number;
  timbre_db: number;
  softness_hz: number;
  sharpness_db: number;
  volume_db: number;
  mix: number;
  noise_gate_db: number;
}

export interface ErrorPayload {
  kind:
    | "model_not_found"
    | "device_not_found"
    | "cuda_unavailable"
    | "cuda_kernel_mismatch"
    | "virtual_mic_not_installed"
    | "unknown";
  message: string;
}

export type TrainingStage = "preprocess" | "f0" | "features" | "train" | "index" | "done" | "error";

export type BackendMessage =
  | { type: "hello"; device: string }
  | { type: "devices"; input: DeviceInfo[]; output: DeviceInfo[]; suggested_virtual_mic: number | null }
  | { type: "voice_models"; models: VoiceModelInfo[] }
  | { type: "presets"; names: string[] }
  | { type: "test_audio_files"; files: string[] }
  | { type: "preset_loaded"; name: string; data: { effects?: Partial<EffectParams>; voice_pth?: string; f0_up_key?: number } }
  | { type: "status"; running: boolean; file_input?: boolean }
  | { type: "file_playback_finished" }
  | { type: "voice_loaded"; pth_path: string }
  | { type: "voice_unloaded" }
  | { type: "latency"; model_ms: number; total_ms: number; underruns: number }
  | { type: "training_progress"; stage: TrainingStage; message: string; fraction: number }
  | ({ type: "error" } & ErrorPayload);

const WS_URL = "ws://127.0.0.1:8765";

export class BackendClient {
  private ws: WebSocket | null = null;
  private listeners = new Set<(msg: BackendMessage) => void>();
  private reconnectTimer: number | null = null;

  connect(): void {
    this.ws = new WebSocket(WS_URL);
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as BackendMessage;
      this.listeners.forEach((l) => l(msg));
    };
    this.ws.onclose = () => {
      this.reconnectTimer = window.setTimeout(() => this.connect(), 1500);
    };
    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }

  onMessage(listener: (msg: BackendMessage) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  send(payload: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(payload));
    }
  }
}
