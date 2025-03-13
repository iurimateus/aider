import math
import os
import queue
import shutil
import tempfile
import threading
import wave
from io import BytesIO
import time
import warnings
from queue import Queue
from threading import Event, Thread

import numpy as np
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

class VADVoice:
    def __init__(
        self, io, audio_format="wav", device_name=None, language=None, model="whisper-1"
    ):
        self.model = model
        self.io = io
        self.enabled = False
        self.audio_format = audio_format
        self.device_name = device_name
        self.language = language
        self.stop_event = Event()
        self.audio_queue = Queue()
        self.recording_thread: Thread | None = None

        try:
            from silero_vad import load_silero_vad, read_audio
        except ImportError as e:
            raise ImportError("Silero VAD dependencies missing") from e

        try:
            import sounddevice as sd
            self.sd = sd
            self.device_id = self._get_device_id(device_name)
        except (OSError, ModuleNotFoundError):
            raise SoundDeviceError

        self.vad_model = load_silero_vad()
        self.sample_rate = 16000  # Silero requires 16kHz
        self.chunk_size = 512  # Samples per VAD check

    def _get_device_id(self, device_name):
        if not device_name:
            return None

        devices = self.sd.query_devices()
        for i, device in enumerate(devices):
            if device_name in device["name"]:
                return i
        return None

    def toggle(self):
        if self.enabled:
            self.stop()
        else:
            self.start()

    def start(self):
        self.vad_model.reset_states()  # Reset VAD states
        self.enabled = True
        self.stop_event.clear()
        self.recording_thread = Thread(target=self._run_vad, daemon=True)
        self.recording_thread.start()

        # # Warmup with silent audio
        # silent_chunk = np.zeros(self.chunk_size, dtype=np.float32)
        # self.audio_queue.put(silent_chunk)

    def stop(self):
        self.enabled = False
        if self.stop_event:
            self.stop_event.set()
        if self.recording_thread:
            self.recording_thread.join(timeout=1)
            self.recording_thread = None
        if self.vad_model:
            self.vad_model.reset_states()

    def _run_vad(self):
        last_speech_time = time.time()
        silence_timeout = 3.0  # Seconds of silence before transcribing
        listening_timeout = 3 * silence_timeout
        cb_called = False

        def callback(indata, _frames, _time, _status):
            if self.io.prompt_active:  # Add this check first
                return

            import torch

            nonlocal last_speech_time
            nonlocal cb_called
            audio = indata[:, 0].astype("float32")
            audio_tensor = torch.from_numpy(audio)
            speech_prob = self.vad_model(audio_tensor, self.sample_rate).item()

            if speech_prob >= 0.55:
                now = time.time()
                if not cb_called or last_speech_time - now > listening_timeout:
                    cb_called = True
                    self.io.tool_output("Listening...")

                last_speech_time = now
                self.audio_queue.put(audio)

        with self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            device=self.device_id,
            blocksize=self.chunk_size
        ):
            while not self.stop_event.is_set():
                # Check for silence timeout every 0.1s
                current_time = time.time()
                if (
                    current_time - last_speech_time >= silence_timeout
                    and self.audio_queue.qsize() > 0
                ):
                    self._transcribe_audio()
                    last_speech_time = time.time()  # Prevent immediate retrigger
                self.stop_event.wait(0.2)

    def _transcribe_audio(self):
        # Completely drain the queue of all available chunks
        audio_chunks = []
        while not self.audio_queue.empty():
            chunk = self.audio_queue.get_nowait()
            audio_chunks.append(chunk)

        if not audio_chunks:
            return

        # Prepend 0.1 seconds of silence to avoid abrupt start
        silence_samples = int(0.1 * self.sample_rate)
        silent_chunk = np.zeros(silence_samples, dtype=np.float32)
        audio_chunks.insert(0, silent_chunk)

        self.io.tool_output("Transcribing...")

        try:
            # Convert float32 samples to int16
            int16_chunks = [
                (np.ascontiguousarray(np.clip(chunk, -1.0, 1.0)) * 32767)
                .astype("int16")
                .tobytes()
                for chunk in audio_chunks
            ]
            raw_audio = b"".join(int16_chunks)
            # Convert PCM buffers to WAV format
            with BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(raw_audio)

                wav_buffer.seek(0)

                transcript = litellm.transcription(
                    model=self.model,
                    file=wav_buffer,
                    language=self.language,
                    custom_llm_provider="openai",
                    prompt="Developer talking to an AI coding assistant about its codebase.",
                )  # type: ignore

            if text := transcript.text.strip():
                # Append to placeholder instead of overwriting
                if self.io.placeholder:
                    self.io.placeholder += " " + text
                else:
                    self.io.placeholder = text
                self.io.interrupt_input()

        except Exception as e:
            # if self.verbose:
            #     from aider.dump import dump
            #
            #     dump(f"Transcription error: {e}")
            self.io.tool_error(f"Transcription error: {str(e)}")
        finally:
            pass  # get_nowait() already marks tasks done


class Voice:
    max_rms = 0
    min_rms = 1e5
    pct = 0

    threshold = 0.15

    def __init__(self, audio_format="wav", device_name=None, model="whisper-1"):
        if sf is None:
            raise SoundDeviceError
        self.model = model
        self.vad_enabled = False
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
                        app = get_app()
                        app.loop.call_soon_threadsafe(app.exit, EOFError())  # type: ignore
                        return
                    time.sleep(0.1)

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
            silence_timeout = 3.5  # seconds
            with self.sd.InputStream(
                samplerate=sample_rate, channels=1, callback=self.callback, device=self.device_id
            ):
                while not stop_event.is_set():
                    current_time = time.time()
                    if self.pct >= self.threshold + 0.02:
                        last_active_time = current_time

                    if current_time - last_active_time >= silence_timeout:
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
