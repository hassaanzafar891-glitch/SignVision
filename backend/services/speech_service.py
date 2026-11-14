
import os
import hashlib
import threading

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tts_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

_mixer_initialized = False
_mixer_lock = threading.Lock()


def _cache_path(label: str) -> str:
    key = hashlib.md5(label.encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{key}.mp3")


def _play_mp3(path: str):
    from pygame import mixer
    global _mixer_initialized
    with _mixer_lock:
        if not _mixer_initialized:
            mixer.init()
            _mixer_initialized = True
        if mixer.music.get_busy():
            mixer.music.stop()
        mixer.music.load(path)
        mixer.music.play()
    import time
    while mixer.music.get_busy():
        time.sleep(0.05)


def preload(label: str):
    def _cache():
        try:
            import asyncio
            asyncio.run(audio_path(label))
        except Exception as e:
            print(f"[speech] preload ERROR: {e}")
    threading.Thread(target=_cache, daemon=True).start()


def play(label: str, mode: str = "offline"):
    def _speak():
        try:
            import asyncio
            cached = asyncio.run(audio_path(label))
            _play_mp3(cached)
        except Exception as e:
            print(f"[speech] ERROR: {e}")
    threading.Thread(target=_speak, daemon=False).start()


async def audio_path(label: str) -> str:
    """Generate audio for browser playback, without opening a server audio device."""
    import edge_tts
    import tempfile
    cached = _cache_path(label)
    if not os.path.isfile(cached):
        # Publish only complete audio; concurrent requests use separate files.
        handle, temporary = tempfile.mkstemp(suffix=".mp3", dir=_CACHE_DIR)
        os.close(handle)
        try:
            await edge_tts.Communicate(label, voice="ur-PK-UzmaNeural").save(temporary)
            os.replace(temporary, cached)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return cached
