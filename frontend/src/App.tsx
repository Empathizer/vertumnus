import { useEffect, useMemo, useRef, useState } from "react";
import { open as openFileDialog } from "@tauri-apps/plugin-dialog";
import {
  BackendClient,
  DeviceInfo,
  EffectParams,
  ErrorPayload,
  TrainingStage,
  VoiceModelInfo,
} from "./backend";

// Picked file paths come from the native OS dialog — Windows paths use "\",
// macOS/Linux use "/". Split on both so the displayed filename is correct
// regardless of platform.
function basename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] ?? path;
}

const DEFAULT_EFFECTS: EffectParams = {
  pitch_semitones: 0,
  timbre_db: 0,
  softness_hz: 20000,
  sharpness_db: 0,
  volume_db: 0,
  mix: 1,
  noise_gate_db: -50,
};

const ERROR_HELP: Record<ErrorPayload["kind"], string> = {
  model_not_found: "Check the .pth (and optional .index) path under backend/models/.",
  device_not_found: "Re-check the input/output device dropdowns — a device may have been unplugged.",
  cuda_unavailable: "Run check_gpu.py on the Windows machine and confirm the cu128 torch build is installed.",
  cuda_kernel_mismatch: "This is the sm_120 kernel mismatch. Switch to the nightly cu128 torch line in setup.bat.",
  virtual_mic_not_installed: "Install VB-Cable (Windows) or BlackHole (macOS), then reopen the output device list.",
  unknown: "See the backend terminal log for the full traceback.",
};

export default function App() {
  const client = useMemo(() => new BackendClient(), []);
  const [connected, setConnected] = useState(false);
  const [device, setDevice] = useState<string>("");
  const [inputDevices, setInputDevices] = useState<DeviceInfo[]>([]);
  const [outputDevices, setOutputDevices] = useState<DeviceInfo[]>([]);
  const [suggestedVirtualMic, setSuggestedVirtualMic] = useState<number | null>(null);
  const [inputDevice, setInputDevice] = useState<number | "">("");
  const [outputDevice, setOutputDevice] = useState<number | "">("");
  const [monitorDevice, setMonitorDevice] = useState<number | "">("");
  const [monitorEnabled, setMonitorEnabled] = useState(false);
  const [muted, setMuted] = useState(false);
  const [running, setRunning] = useState(false);

  const [voiceModels, setVoiceModels] = useState<VoiceModelInfo[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [pitchKey, setPitchKey] = useState(0);
  const [voiceLoaded, setVoiceLoaded] = useState<string | null>(null);

  const [effects, setEffects] = useState<EffectParams>(DEFAULT_EFFECTS);
  const [presets, setPresets] = useState<string[]>([]);
  const [presetName, setPresetName] = useState("");

  const [latency, setLatency] = useState({ model_ms: 0, total_ms: 0, underruns: 0 });
  const [error, setError] = useState<ErrorPayload | null>(null);

  const [testAudioFiles, setTestAudioFiles] = useState<string[]>([]);
  const [selectedTestFile, setSelectedTestFile] = useState<string>("");
  const [usingFileInput, setUsingFileInput] = useState(false);

  const [trainSourceFile, setTrainSourceFile] = useState<string>("");
  const [trainVoiceName, setTrainVoiceName] = useState<string>("");
  const [trainEpochs, setTrainEpochs] = useState<number>(20);
  const [isTraining, setIsTraining] = useState(false);
  const [trainStage, setTrainStage] = useState<TrainingStage | null>(null);
  const [trainMessage, setTrainMessage] = useState<string>("");
  const [trainFraction, setTrainFraction] = useState<number>(0);

  const effectsDebounce = useRef<number | null>(null);

  useEffect(() => {
    client.connect();
    const unsubscribe = client.onMessage((msg) => {
      switch (msg.type) {
        case "hello":
          setConnected(true);
          setDevice(msg.device);
          client.send({ type: "list_devices" });
          client.send({ type: "list_voice_models" });
          client.send({ type: "list_presets" });
          client.send({ type: "list_test_audio" });
          break;
        case "devices": {
          setInputDevices(msg.input);
          setOutputDevices(msg.output);
          setSuggestedVirtualMic(msg.suggested_virtual_mic);
          // Device indices can shift on hot-plug (headphones in/out, etc.) —
          // drop a selection that no longer refers to a real device instead
          // of silently pointing at whatever now has that stale index.
          const validIn = new Set(msg.input.map((d) => d.index));
          const validOut = new Set(msg.output.map((d) => d.index));
          setInputDevice((prev) => (prev !== "" && !validIn.has(prev) ? "" : prev));
          setOutputDevice((prev) => {
            if (prev !== "" && validOut.has(prev)) return prev;
            return msg.suggested_virtual_mic !== null ? msg.suggested_virtual_mic : "";
          });
          setMonitorDevice((prev) => (prev !== "" && !validOut.has(prev) ? "" : prev));
          break;
        }
        case "voice_models":
          setVoiceModels(msg.models);
          break;
        case "presets":
          setPresets(msg.names);
          break;
        case "preset_loaded":
          if (msg.data.effects) setEffects((e) => ({ ...e, ...msg.data.effects }));
          if (typeof msg.data.f0_up_key === "number") setPitchKey(msg.data.f0_up_key);
          break;
        case "status":
          setRunning(msg.running);
          setUsingFileInput(msg.running ? !!msg.file_input : false);
          break;
        case "file_playback_finished":
          break;
        case "test_audio_files":
          setTestAudioFiles(msg.files);
          break;
        case "voice_loaded":
          setVoiceLoaded(msg.pth_path);
          setError(null);
          break;
        case "voice_unloaded":
          setVoiceLoaded(null);
          break;
        case "latency":
          setLatency(msg);
          break;
        case "training_progress":
          setTrainStage(msg.stage);
          setTrainMessage(msg.message);
          setTrainFraction(msg.fraction);
          setIsTraining(msg.stage !== "done" && msg.stage !== "error");
          break;
        case "error":
          setError(msg);
          setIsTraining(false);
          break;
      }
    });
    return () => {
      unsubscribe();
      client.disconnect();
    };
  }, [client]);

  function pushEffects(next: EffectParams) {
    setEffects(next);
    if (effectsDebounce.current) window.clearTimeout(effectsDebounce.current);
    effectsDebounce.current = window.setTimeout(() => {
      client.send({ type: "update_effects", ...next });
    }, 30);
  }

  function handleStart() {
    if (inputDevice === "" || outputDevice === "") return;
    client.send({
      type: "start",
      input_device: inputDevice,
      output_device: outputDevice,
      monitor_device: monitorEnabled && monitorDevice !== "" ? monitorDevice : null,
    });
  }

  function handleStop() {
    client.send({ type: "stop" });
  }

  async function handleBrowseFile() {
    const picked = await openFileDialog({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aiff"] }],
    });
    if (typeof picked === "string") {
      setSelectedTestFile(picked);
    }
  }

  function handleStartFromFile() {
    if (selectedTestFile === "" || outputDevice === "") return;
    client.send({
      type: "start_from_file",
      file_path: selectedTestFile,
      output_device: outputDevice,
      monitor_device: monitorEnabled && monitorDevice !== "" ? monitorDevice : null,
    });
  }

  async function handleBrowseTrainingFile() {
    const picked = await openFileDialog({
      multiple: false,
      filters: [{ name: "Audio", extensions: ["wav", "mp3", "m4a", "flac", "ogg", "aiff"] }],
    });
    if (typeof picked === "string") {
      setTrainSourceFile(picked);
      if (trainVoiceName === "") {
        const base = basename(picked);
        setTrainVoiceName(base.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]/g, "_"));
      }
    }
  }

  function handleStartTraining() {
    if (trainSourceFile === "" || trainVoiceName.trim() === "" || isTraining) return;
    setIsTraining(true);
    setTrainStage(null);
    setTrainMessage("Starting...");
    setTrainFraction(0);
    client.send({
      type: "start_training",
      source_file_path: trainSourceFile,
      voice_name: trainVoiceName.trim(),
      epochs: trainEpochs,
    });
  }

  function handleLoadVoice(pth: string) {
    setSelectedVoice(pth);
    if (pth === "") {
      // "No voice (DSP passthrough only)" — must actually tell the backend
      // to unload, otherwise a previously-loaded model keeps running even
      // though the UI shows nothing selected.
      client.send({ type: "unload_voice" });
      return;
    }
    const model = voiceModels.find((m) => m.pth_path === pth);
    if (!model) return;
    client.send({
      type: "load_voice",
      pth_path: model.pth_path,
      index_path: model.index_path,
      f0_up_key: pitchKey,
    });
  }

  function handlePitchChange(value: number) {
    setPitchKey(value);
    client.send({ type: "set_pitch", f0_up_key: value });
  }

  function handleMonitorToggle(enabled: boolean) {
    setMonitorEnabled(enabled);
    if (enabled && monitorDevice !== "") {
      client.send({ type: "set_monitor", enabled: true, device: monitorDevice });
    } else {
      client.send({ type: "set_monitor", enabled: false });
    }
  }

  function handleMuteToggle() {
    const next = !muted;
    setMuted(next);
    client.send({ type: "set_mute", muted: next });
  }

  function handleSavePreset() {
    if (!presetName.trim()) return;
    client.send({ type: "save_preset", name: presetName.trim() });
    setPresetName("");
  }

  function handleLoadPreset(name: string) {
    client.send({ type: "load_preset", name });
  }

  return (
    <div className="app">
      <header>
        <h1>Vertumnus</h1>
        <span className={`status-dot ${connected ? "ok" : "bad"}`} />
        <span className="status-text">
          {connected ? `backend connected (${device})` : "backend disconnected"}
        </span>
      </header>

      {error && (
        <div className="banner error">
          <strong>{error.kind.replace(/_/g, " ")}:</strong> {error.message}
          <div className="banner-help">{ERROR_HELP[error.kind]}</div>
        </div>
      )}

      <section>
        <h2>Devices</h2>
        <div className="row">
          <button onClick={() => client.send({ type: "list_devices" })}>
            Refresh devices
          </button>
        </div>
        <p className="hint">
          Plugged in or unplugged headphones? Click Refresh, then re-select input/output below.
        </p>
        <div className="row">
          <label>
            Input (mic)
            <select value={inputDevice} onChange={(e) => setInputDevice(Number(e.target.value))}>
              <option value="">Select...</option>
              {inputDevices.map((d) => (
                <option key={d.index} value={d.index}>
                  {d.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Output (virtual mic)
            <select value={outputDevice} onChange={(e) => setOutputDevice(Number(e.target.value))}>
              <option value="">Select...</option>
              {outputDevices.map((d) => (
                <option key={d.index} value={d.index}>
                  {d.name}
                  {d.index === suggestedVirtualMic ? " (detected)" : ""}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="row">
          <label className="checkbox">
            <input
              type="checkbox"
              checked={monitorEnabled}
              onChange={(e) => handleMonitorToggle(e.target.checked)}
            />
            Monitor (hear myself)
          </label>
          <select
            value={monitorDevice}
            disabled={!monitorEnabled}
            onChange={(e) => setMonitorDevice(Number(e.target.value))}
          >
            <option value="">Select monitor device...</option>
            {outputDevices.map((d) => (
              <option key={d.index} value={d.index}>
                {d.name}
              </option>
            ))}
          </select>
        </div>
        <div className="row">
          {!running ? (
            <button onClick={handleStart} disabled={inputDevice === "" || outputDevice === ""}>
              Start
            </button>
          ) : (
            <button onClick={handleStop} className="danger">
              Stop
            </button>
          )}
          <button onClick={handleMuteToggle} className={muted ? "danger" : ""}>
            {muted ? "Unmute" : "Mute"}
          </button>
        </div>
        <hr />
        <p className="hint">
          Or test with a pre-recorded file instead of the live mic (no background noise/feedback to deal with):
        </p>
        <div className="row">
          <select value={selectedTestFile} onChange={(e) => setSelectedTestFile(e.target.value)}>
            <option value="">Select a test audio file...</option>
            {selectedTestFile !== "" && !testAudioFiles.includes(selectedTestFile) && (
              <option value={selectedTestFile}>{basename(selectedTestFile)} (browsed)</option>
            )}
            {testAudioFiles.map((f) => (
              <option key={f} value={f}>
                {basename(f)}
              </option>
            ))}
          </select>
          <button onClick={handleBrowseFile}>Browse...</button>
          <button onClick={() => client.send({ type: "list_test_audio" })}>Refresh</button>
          <button
            onClick={handleStartFromFile}
            disabled={selectedTestFile === "" || outputDevice === "" || (running && !usingFileInput)}
          >
            Start from file
          </button>
        </div>
        <p className="hint">
          Click Browse... to pick any audio file from anywhere on your Mac, or drop .wav files into{" "}
          <code>backend/test_audio/</code> and hit Refresh. Output/Monitor device selection above still applies.
        </p>
        {usingFileInput && running && <p className="hint ok-text">Playing from file...</p>}
      </section>

      <section>
        <h2>Voice model</h2>
        <div className="row">
          <select value={selectedVoice} onChange={(e) => handleLoadVoice(e.target.value)}>
            <option value="">No voice (DSP passthrough only)</option>
            {voiceModels.map((m) => (
              <option key={m.pth_path} value={m.pth_path}>
                {m.name}
                {m.index_path ? " (+index)" : ""}
              </option>
            ))}
          </select>
          <button onClick={() => client.send({ type: "list_voice_models" })}>Refresh</button>
        </div>
        <p className="hint">
          Drop .pth (and optional same-named .index) files into <code>backend/models/</code>, then Refresh.
        </p>
        {voiceLoaded && <p className="hint ok-text">Loaded: {basename(voiceLoaded)}</p>}
        <label>
          Pitch key ({pitchKey > 0 ? "+" : ""}
          {pitchKey} semitones)
          <input
            type="range"
            min={-24}
            max={24}
            value={pitchKey}
            onChange={(e) => handlePitchChange(Number(e.target.value))}
          />
        </label>
      </section>

      <section>
        <h2>Train a new voice</h2>
        <div className="row">
          <button onClick={handleBrowseTrainingFile} disabled={isTraining}>
            Browse...
          </button>
          <span className="hint">
            {trainSourceFile ? basename(trainSourceFile) : "No source recording selected"}
          </span>
        </div>
        <div className="row">
          <label>
            Voice name
            <input
              type="text"
              value={trainVoiceName}
              onChange={(e) => setTrainVoiceName(e.target.value)}
              disabled={isTraining}
              placeholder="e.g. myvoice"
            />
          </label>
          <label>
            Epochs
            <input
              type="number"
              min={1}
              max={1000}
              value={trainEpochs}
              onChange={(e) => setTrainEpochs(Number(e.target.value))}
              disabled={isTraining}
            />
          </label>
        </div>
        <div className="row">
          <button
            onClick={handleStartTraining}
            disabled={isTraining || trainSourceFile === "" || trainVoiceName.trim() === ""}
          >
            {isTraining ? "Training..." : "Start Training"}
          </button>
          {isTraining && (
            <button className="danger" onClick={() => client.send({ type: "cancel_training" })}>
              Cancel
            </button>
          )}
        </div>
        <p className="hint">
          Pick a recording of the voice you want to clone (needs your rights to that voice — your own recording, or
          one you're licensed to use). More epochs = better quality but longer training; this Mac trains at roughly
          ~2-3 min/epoch on a short recording — much faster on the Windows RTX box. The trained model appears
          automatically in the Voice model dropdown above when done.
        </p>
        {(isTraining || trainStage) && (
          <div className="training-progress">
            <div className="training-progress-bar">
              <div
                className="training-progress-fill"
                style={{ width: `${Math.round(trainFraction * 100)}%` }}
              />
            </div>
            <p className={`hint ${trainStage === "error" ? "error-text" : trainStage === "done" ? "ok-text" : ""}`}>
              {trainMessage}
            </p>
          </div>
        )}
      </section>

      <section>
        <h2>Effects</h2>
        <SliderRow
          label="Pitch"
          value={effects.pitch_semitones}
          min={-12}
          max={12}
          step={0.1}
          unit="st"
          onChange={(v) => pushEffects({ ...effects, pitch_semitones: v })}
        />
        <SliderRow
          label="Timbre"
          value={effects.timbre_db}
          min={-12}
          max={12}
          step={0.1}
          unit="dB"
          onChange={(v) => pushEffects({ ...effects, timbre_db: v })}
        />
        <SliderRow
          label="Softness"
          value={effects.softness_hz}
          min={500}
          max={20000}
          step={50}
          unit="Hz"
          onChange={(v) => pushEffects({ ...effects, softness_hz: v })}
        />
        <SliderRow
          label="Sharpness"
          value={effects.sharpness_db}
          min={0}
          max={12}
          step={0.1}
          unit="dB"
          onChange={(v) => pushEffects({ ...effects, sharpness_db: v })}
        />
        <SliderRow
          label="Output Volume"
          value={effects.volume_db}
          min={-24}
          max={24}
          step={0.1}
          unit="dB"
          onChange={(v) => pushEffects({ ...effects, volume_db: v })}
        />
        <p className="hint">
          Applies after voice conversion too (works with or without a model loaded) — use this to fine-tune final loudness on top of automatic gain matching.
        </p>
        <SliderRow
          label="Mix (dry/wet)"
          value={effects.mix}
          min={0}
          max={1}
          step={0.01}
          unit=""
          onChange={(v) => pushEffects({ ...effects, mix: v })}
        />
        <SliderRow
          label="Noise Gate"
          value={effects.noise_gate_db}
          min={-60}
          max={-10}
          step={0.5}
          unit="dB"
          onChange={(v) => pushEffects({ ...effects, noise_gate_db: v })}
        />
        <p className="hint">
          Raise (toward -10dB) to cut more background noise during pauses; lower (toward -60dB) if it's clipping quiet speech.
        </p>
      </section>

      <section>
        <h2>Presets</h2>
        <div className="row">
          <input
            placeholder="preset name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
          />
          <button onClick={handleSavePreset}>Save</button>
        </div>
        <div className="row wrap">
          {presets.map((name) => (
            <button key={name} onClick={() => handleLoadPreset(name)}>
              {name}
            </button>
          ))}
        </div>
      </section>

      <section>
        <h2>Latency</h2>
        <div className="latency-readout">
          <span>model: {latency.model_ms.toFixed(1)} ms</span>
          <span>total: {latency.total_ms.toFixed(1)} ms</span>
          <span>underruns: {latency.underruns}</span>
        </div>
      </section>
    </div>
  );
}

function SliderRow(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="slider-row">
      <span>
        {props.label}: {props.value.toFixed(2)}
        {props.unit}
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}
