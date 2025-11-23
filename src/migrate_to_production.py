import os
import shutil
import sys
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
# Assuming script is run from 'src/'
BASE_DIR = Path(__file__).parent
PACKAGE_DIR = BASE_DIR / "Autonomous_Reasoning_System"
REFACTOR_DIR = PACKAGE_DIR / "refactor"
DATA_DIR = PACKAGE_DIR / "data"

# Folders to DELETE (Legacy)
LEGACY_DIRS = [
    "control", "cognition", "planning", "llm", 
    "memory", "rag", "io", "tools", "infrastructure", 
    "tests", "__pycache__"
]

# Files to DELETE (Legacy)
LEGACY_FILES = [
    "consolidated_app.py", "main.py", "init_runtime.py", 
    "interface.py" # Old interface if present in root
]

def create_backup():
    """Zips the current package before we destroy it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_legacy_{timestamp}"
    print(f"📦 Creating backup: {backup_name}.zip ...")
    shutil.make_archive(backup_name, 'zip', PACKAGE_DIR)
    print("✅ Backup complete.")

def purge_legacy():
    """Deletes old folders and files."""
    print("🔥 Purging legacy code...")
    
    # Delete Directories
    for folder in LEGACY_DIRS:
        target = PACKAGE_DIR / folder
        if target.exists() and target.is_dir():
            try:
                shutil.rmtree(target)
                print(f"   Deleted folder: {folder}/")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {folder}: {e}")

    # Delete Files
    for file in LEGACY_FILES:
        target = PACKAGE_DIR / file
        if target.exists() and target.is_file():
            try:
                target.unlink()
                print(f"   Deleted file: {file}")
            except Exception as e:
                print(f"   ⚠️ Failed to delete {file}: {e}")

def promote_refactor():
    """Moves files from refactor/ to package root."""
    print("🚀 Promoting refactored code...")
    
    if not REFACTOR_DIR.exists():
        print("❌ Error: 'refactor' directory not found!")
        sys.exit(1)

    # Move everything from refactor/ to Autonomous_Reasoning_System/
    for item in REFACTOR_DIR.iterdir():
        if item.name == "__pycache__":
            continue
            
        target = PACKAGE_DIR / item.name
        
        if target.exists():
            print(f"   ⚠️ Overwriting existing: {item.name}")
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        
        shutil.move(str(item), str(target))
        print(f"   Moved: {item.name}")

    # Remove empty refactor dir
    try:
        REFACTOR_DIR.rmdir()
        print("   Cleaned up refactor/ directory.")
    except Exception:
        print("   (Note: refactor/ dir not empty, kept it just in case)")

def main():
    print(f"--- TYRONE MIGRATION TOOL ---")
    print(f"Target Package: {PACKAGE_DIR}")
    
    if input("Are you sure you want to replace the codebase? (y/n): ").lower() != 'y':
        print("Aborted.")
        return

    create_backup()
    purge_legacy()
    promote_refactor()
    
    print("\n✨ Migration Successful!")
    print("You can now run the app using:")
    print("python -m Autonomous_Reasoning_System.interface")

if __name__ == "__main__":
    main()