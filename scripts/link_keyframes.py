import os
import glob
from pathlib import Path
import shutil

def link_keyframes():
    base_dir = Path("data")
    batch_dir = base_dir / "batch_01"
    target_dir = base_dir / "keyframes"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all L* keyframe directories
    # They are like data/batch_01/Keyframes_L21/keyframes/L21_V001/
    search_pattern = str(batch_dir / "Keyframes_L*" / "keyframes" / "*")
    video_dirs = glob.glob(search_pattern)
    
    print(f"Found {len(video_dirs)} video keyframe directories.")
    
    success_count = 0
    fallback_count = 0
    
    for vdir in video_dirs:
        vdir_path = Path(vdir)
        if not vdir_path.is_dir():
            continue
            
        vid = vdir_path.name
        link_path = target_dir / vid
        
        if link_path.exists():
            continue
            
        try:
            # Create a directory junction / symlink
            os.symlink(vdir_path.resolve(), link_path.resolve(), target_is_directory=True)
            success_count += 1
        except OSError as e:
            # If symlink fails (often due to privileges on Windows), use copy as a fallback or junction
            try:
                import _winapi
                _winapi.CreateJunction(str(vdir_path.resolve()), str(link_path.resolve()))
                success_count += 1
            except Exception as ex:
                print(f"Symlink and Junction failed for {vid}. Falling back to copy...")
                shutil.copytree(vdir_path, link_path)
                fallback_count += 1
                
    print(f"Successfully linked {success_count} directories.")
    if fallback_count > 0:
        print(f"Copied {fallback_count} directories as fallback.")
        
if __name__ == "__main__":
    link_keyframes()
