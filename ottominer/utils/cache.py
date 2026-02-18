import hashlib
from pathlib import Path
from functools import wraps
import json
import pickle
import logging
import time
from typing import Union, Optional, Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".ottominer" / "cache"


def cache_result(cache_dir: Path = Path(".cache"), ttl_hours: int = 24 * 7):
    """Cache extraction results with TTL support."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, file_path: Union[str, Path], *args, **kwargs):
            file_path = Path(file_path)

            stat = file_path.stat()
            key_data = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            cache_file = cache_dir / f"{cache_key}.pkl"

            cache_dir.mkdir(parents=True, exist_ok=True)

            if cache_file.exists():
                try:
                    with cache_file.open("rb") as f:
                        cached = pickle.load(f)
                    if isinstance(cached, dict) and "timestamp" in cached:
                        age = time.time() - cached["timestamp"]
                        if age < ttl_hours * 3600:
                            logger.debug(f"Cache hit for {file_path}")
                            return cached.get("result")
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")

            result = func(self, file_path, *args, **kwargs)

            try:
                with cache_file.open("wb") as f:
                    pickle.dump({"result": result, "timestamp": time.time()}, f)
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

            return result

        return wrapper

    return decorator


class ExtractionCache:
    """Cache for PDF extraction results."""

    def __init__(self, cache_dir: Path = None, ttl_hours: int = 24 * 7):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.extractions_dir = self.cache_dir / "extractions"
        self.extractions_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600

    def _get_key(self, file_path: Path) -> str:
        stat = file_path.stat()
        key_data = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get(self, file_path: Path) -> Optional[str]:
        """Get cached extraction result if valid."""
        key = self._get_key(file_path)
        cache_file = self.extractions_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))

            if time.time() - data.get("timestamp", 0) > self.ttl_seconds:
                cache_file.unlink()
                logger.debug(f"Cache expired for {file_path}")
                return None

            logger.info(f"Cache hit for {file_path}")
            return data.get("text")

        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def set(self, file_path: Path, text: str, metadata: Dict = None) -> None:
        """Store extraction result in cache."""
        key = self._get_key(file_path)
        cache_file = self.extractions_dir / f"{key}.json"

        try:
            data = {
                "file_path": str(file_path),
                "text": text,
                "timestamp": time.time(),
                "metadata": metadata or {},
            }
            cache_file.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            logger.debug(f"Cached extraction for {file_path}")
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def invalidate(self, file_path: Path) -> bool:
        """Remove cached result for a file."""
        key = self._get_key(file_path)
        cache_file = self.extractions_dir / f"{key}.json"

        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cached extractions."""
        count = 0
        for f in self.extractions_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached extractions")
        return count

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        files = list(self.extractions_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "ttl_hours": self.ttl_seconds / 3600,
        }
