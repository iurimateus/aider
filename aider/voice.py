import math
import os
import queue
import tempfile
import threading
import time
import warnings

from prompt_toolkit.application import get_app
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.eventloop.inputhook import (
    InputHookContext,
)

from aider.llm import litellm

from .dump import dump  # noqa: F401

warnings.filterwarnings(
    "ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)
warnings.filterwarnings("ignore", category=SyntaxWarning)


from pydub import AudioSegment  # noqa
from pydub.exceptions import CouldntDecodeError, CouldntEncodeError  # noqa

try:
    import soundfile as sf
except (OSError, ModuleNotFoundError):
    sf = None


class SoundDeviceError(Exception):
    pass


class Voice:
    max_rms = 0
    min_rms = 1e5
    pct = 0

    threshold = 0.15

    def __init__(self, audio_format="wav", device_name=None, model="whisper-1"):
        if sf is None:
            raise SoundDeviceError
        self.model = model
        try:
            import sounddevice as sd

            self.sd = sd

            devices = sd.query_devices()

            if device_name:
                # Find the device with matching name
                device_id = None
                for i, device in enumerate(devices):
                    if device_name in device["name"]:
                        device_id = i
                        break
                if device_id is None:
                    available_inputs = [d["name"] for d in devices if d["max_input_channels"] > 0]
                    raise ValueError(
                        f"Device '{device_name}' not found. Available input devices:"
                        f" {available_inputs}"
                    )

                print(f"Using input device: {device_name} (ID: {device_id})")

                self.device_id = device_id
            else:
                self.device_id = None

        except (OSError, ModuleNotFoundError):
            raise SoundDeviceError
        if audio_format not in ["wav", "mp3", "webm"]:
            raise ValueError(f"Unsupported audio format: {audio_format}")
        self.audio_format = audio_format

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        import numpy as np

        rms = np.sqrt(np.mean(indata**2))
        self.max_rms = max(self.max_rms, rms)
        self.min_rms = min(self.min_rms, rms)

        rng = self.max_rms - self.min_rms
        if rng > 0.001:
            self.pct = (rms - self.min_rms) / rng
        else:
            self.pct = 0.5

        self.q.put(indata.copy())

    def get_prompt(self):
        num = 10
        if math.isnan(self.pct) or self.pct < self.threshold:
            cnt = 0
        else:
            cnt = int(self.pct * 10)

        bar = "░" * cnt + "█" * (num - cnt)
        bar = bar[:num]

        dur = time.time() - self.start_time
        return f"Recording, press ENTER when done... {dur:.1f}sec {bar}"

    def record_and_transcribe(self, history=None, language=None):
        try:
            return self.raw_record_and_transcribe(history, language)
        except KeyboardInterrupt:
            return
        except SoundDeviceError as e:
            print(f"Error: {e}")
            print("Please ensure you have a working audio input device connected and try again.")
            return

    def raw_record_and_transcribe(self, history, language):
        self.q = queue.Queue()
        temp_wav = tempfile.mktemp(suffix=".wav")

        try:
            sample_rate = int(self.sd.query_devices(self.device_id, "input")["default_samplerate"])
        except (TypeError, ValueError):
            sample_rate = 16000  # fallback to 16kHz if unable to query device
        except self.sd.PortAudioError:
            raise SoundDeviceError("No audio input device detected.")

        self.start_time = time.time()
        last_active_time = self.start_time
        stop_event = threading.Event()  # Thread-safe event for stopping

        def run_prompt():
            try:
                # Create input hook that checks our stop condition
                def inputhook(context: InputHookContext):
                    if stop_event.is_set():
                        get_app().loop.call_soon_threadsafe(get_app().exit, EOFError())
                        return
                    time.sleep(0.2)

                # Run the prompt with our input hook
                prompt(
                    self.get_prompt,
                    refresh_interval=0.1,
                    inputhook=inputhook,
                    # in_thread=True,
                )
                stop_event.set()

            except (KeyboardInterrupt, EOFError):
                stop_event.set()

        input_thread = threading.Thread(target=run_prompt, daemon=True)
        input_thread.start()

        try:
            timeout = 5  # seconds
            with self.sd.InputStream(
                samplerate=sample_rate, channels=1, callback=self.callback, device=self.device_id
            ):
                while not stop_event.is_set():
                    current_time = time.time()
                    if self.pct >= self.threshold + 0.02:
                        last_active_time = current_time

                    if current_time - last_active_time >= timeout:
                        break

                    # Shorter wait period allows checking activity more frequently
                    stop_event.wait(0.2)

        except self.sd.PortAudioError as err:
            raise SoundDeviceError(f"Error accessing audio input device: {err}")
        finally:
            stop_event.set()
            input_thread.join(timeout=1)

        with sf.SoundFile(temp_wav, mode="x", samplerate=sample_rate, channels=1) as file:
            while not self.q.empty():
                file.write(self.q.get())

        use_audio_format = self.audio_format

        # Check file size and offer to convert to mp3 if too large
        file_size = os.path.getsize(temp_wav)
        if file_size > 24.9 * 1024 * 1024 and self.audio_format == "wav":
            print(f"\nWarning: {temp_wav} is too large, switching to mp3 format.")
            use_audio_format = "mp3"

        filename = temp_wav
        if use_audio_format != "wav":
            try:
                new_filename = tempfile.mktemp(suffix=f".{use_audio_format}")
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(new_filename, format=use_audio_format)
                os.remove(temp_wav)
                filename = new_filename
            except (CouldntDecodeError, CouldntEncodeError) as e:
                print(f"Error converting audio: {e}")
            except (OSError, FileNotFoundError) as e:
                print(f"File system error during conversion: {e}")
            except Exception as e:
                print(f"Unexpected error during audio conversion: {e}")

        with open(filename, "rb") as fh:
            try:
                transcript = litellm.transcription(
                    model=self.model,
                    file=fh,
                    prompt=history,
                    language=language,
                    custom_llm_provider="openai",
                )
            except Exception as err:
                print(f"Unable to transcribe {filename}: {err}")
                return

        if filename != temp_wav:
            os.remove(filename)

        text = transcript.text
        return text


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    print(Voice().record_and_transcribe())
