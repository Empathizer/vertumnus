"""Audio device enumeration helpers, built on sounddevice/PortAudio."""
from dataclasses import dataclass

import sounddevice as sd


@dataclass
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    host_api: str


def list_devices() -> list[DeviceInfo]:
    hostapis = sd.query_hostapis()
    devices = []
    for idx, d in enumerate(sd.query_devices()):
        devices.append(
            DeviceInfo(
                index=idx,
                name=d["name"],
                max_input_channels=d["max_input_channels"],
                max_output_channels=d["max_output_channels"],
                default_samplerate=d["default_samplerate"],
                host_api=hostapis[d["hostapi"]]["name"],
            )
        )
    return devices


def list_input_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_input_channels > 0]


def list_output_devices() -> list[DeviceInfo]:
    return [d for d in list_devices() if d.max_output_channels > 0]


def print_devices() -> None:
    print("\n=== Input devices ===")
    for d in list_input_devices():
        print(f"  [{d.index}] {d.name}  (in={d.max_input_channels}, "
              f"sr={d.default_samplerate:.0f}, api={d.host_api})")

    print("\n=== Output devices ===")
    for d in list_output_devices():
        print(f"  [{d.index}] {d.name}  (out={d.max_output_channels}, "
              f"sr={d.default_samplerate:.0f}, api={d.host_api})")
    print()


def find_device_by_name(fragment: str, want_input: bool) -> DeviceInfo | None:
    fragment_lower = fragment.lower()
    candidates = list_input_devices() if want_input else list_output_devices()
    for d in candidates:
        if fragment_lower in d.name.lower():
            return d
    return None
