from pathlib import Path
import hashlib 
proj_dir = Path(__file__).parent.parent
paths = [proj_dir / 'include/luisa/runtime/rhi', proj_dir / 'include/luisa/backends']
dst_dir = proj_dir / 'include/luisa/runtime/rhi/backend_version.inl'

hash_str = ""
for folder in paths:
    for file_path in folder.iterdir():
        if file_path.is_file():
            f = open(file_path, 'rb')
            bytes = f.read()
            f.close()
            hash_str += hashlib.sha256(bytes).hexdigest()

final_hash = hashlib.sha256(hash_str.encode('ascii')).hexdigest()
f = open(dst_dir, 'w')
f.write(f'constexpr const char luisa_version_symbol[] = "{final_hash}";')
f.close()