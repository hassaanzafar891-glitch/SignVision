"""Anonymous retrieval and read-only validation of the actual reviewed DVC object."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

from tracking.data_validation import TABLES, validate_training_database
from tracking.evaluation_gate import DATASET_MD5, GateError, require
from tracking.import_baselines import sha256


SIZE = 6864896
SHA256 = "e6d6c59c7a0c7a207388d290c7ddae7f05ba1aa1507df8b5d293ac4ec13c33a9"
REMOTE = "https://dagshub.com/hassaanzafar891-glitch/SignVision.dvc"
POINTER = Path(__file__).resolve().parents[1] / "data/db/main_dataset.db.dvc"


def verify_pointer(pointer=POINTER):
    # Intentionally accepts only the reviewed single-object pointer format.
    # New datasets need an explicit policy review, never an implicit re-pin.
    text = Path(pointer).read_text(encoding="utf-8-sig")
    match = re.fullmatch(r"outs:\s*\n- md5: ([0-9a-f]{32})\s*\n  size: ([0-9]+)\s*\n  hash: md5\s*\n  path: main_dataset\.db\s*", text)
    require(match is not None, "Unsupported or malformed DVC dataset pointer")
    require(match[1] == DATASET_MD5 and int(match[2]) == SIZE, "DVC dataset differs from reviewed identity")


def validate(path, *, pointer=POINTER):
    verify_pointer(pointer)
    path = Path(path)
    require(path.is_file() and path.stat().st_size == SIZE, "Actual DVC dataset missing or wrong size")
    require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-shm", "-journal")),
            "Dataset has SQLite sidecar changes")
    raw = path.read_bytes()
    require(hashlib.md5(raw).hexdigest() == DATASET_MD5 and hashlib.sha256(raw).hexdigest() == SHA256,
            "Actual dataset MD5/SHA-256 mismatch")
    tables = [validate_training_database(path, table) for table in TABLES]
    require(sha256(path) == SHA256, "Dataset changed during validation")
    return {"scope": "actual-dvc-dataset", "source": REMOTE, "dvc_md5": DATASET_MD5,
            "sha256": SHA256, "bytes": SIZE, "tables": tables, "validated": True}


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        require(newurl.startswith("https://"), "Refusing non-HTTPS dataset redirect")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def fetch(destination, *, pointer=POINTER):
    verify_pointer(pointer)
    destination = Path(destination)
    require(not destination.exists(), "Refusing to overwrite dataset destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # No auth/password/cookie handlers, local DVC config, netrc or environment proxy credentials.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), HTTPSRedirect())
    suffix = DATASET_MD5[:2] + "/" + DATASET_MD5[2:]
    response = None
    for layout in ("files/md5/", ""):
        try:
            response = opener.open(REMOTE + "/" + layout + suffix, timeout=30)
            break
        except urllib.error.HTTPError as error:
            error.close()
            if error.code == 404 and layout:
                continue
            raise GateError(f"Anonymous actual DVC dataset unavailable (HTTP {error.code}); required validation BLOCKED. "
                            "Provide approved credential-free access; PR secrets and fixture substitution are forbidden.") from None
        except urllib.error.URLError:
            raise GateError("Anonymous actual DVC dataset retrieval failed; required validation BLOCKED") from None
    require(response is not None, "Actual DVC object unavailable")
    created = False
    try:
        with response, destination.open("xb") as stream:
            created = True
            total = 0
            while block := response.read(min(65536, SIZE + 1 - total)):
                total += len(block)
                require(total <= SIZE, "Downloaded dataset exceeds reviewed size")
                stream.write(block)
        return validate(destination, pointer=pointer)
    except BaseException:
        if created:
            destination.unlink()  # Only this new download; originals are never opened for writing.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fetch-to", type=Path)
    source.add_argument("--dataset", type=Path, help="Read-only verification of an existing actual dataset")
    args = parser.parse_args()
    try:
        print(json.dumps(fetch(args.fetch_to) if args.fetch_to else validate(args.dataset), indent=2))
    except (GateError, OSError, ValueError) as error:
        parser.exit(1, f"Actual dataset validation failed: {error}\n")


if __name__ == "__main__":
    main()
