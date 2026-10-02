"""Build a source handoff without local secrets, resident data or dependencies."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'CURE-handoff.zip'
EXCLUDED_DIRS = {'.venv', 'venv', 'node_modules', '__pycache__', 'core_store',
                 '.pytest_cache', '.electron-smoke', 'dist', '.git', '.codex', '.agents'}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (not any(part in EXCLUDED_DIRS for part in relative.parts)
            and not any(part.startswith('pytest-cache-files-') for part in relative.parts)
            and not path.is_symlink()
            and (not path.name.startswith('.env') or path.name == '.env.example'))


def main() -> None:
    files = [ROOT / name for name in ('README.md', 'CURE.cmd', 'INSTALL.cmd', '.gitignore')]
    for directory in ('backend', 'front', 'tools'):
        files.extend(path for path in (ROOT / directory).rglob('*') if path.is_file()
                     and include(path) and path.suffix.lower() not in {'.pyc', '.pyo', '.log', '.zip'})
    manifest = {}
    with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr('CURE/' + relative, data)
            manifest[relative] = hashlib.sha256(data).hexdigest()
        archive.writestr('CURE/MANIFEST.sha256.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(OUTPUT) as archive:
        assert archive.testzip() is None
        for relative, digest in manifest.items():
            assert hashlib.sha256(archive.read('CURE/' + relative)).hexdigest() == digest
    print(f'{OUTPUT.name}: {len(files)} files, {OUTPUT.stat().st_size:,} bytes; integrity verified')


if __name__ == '__main__':
    main()
