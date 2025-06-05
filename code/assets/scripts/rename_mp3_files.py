#!/usr/bin/env python3
"""
Script to rename files by adding a custom prefix before the extension.
Usage: python rename_files.py [prefix] [directory_path]
Examples:
    python rename_files.py _fake
    python rename_files.py _real /path/to/directory
    python rename_files.py _processed .
If no directory is provided, uses current directory.
If no prefix is provided, uses '_fake' as default.
"""

import os
import sys
from pathlib import Path

def rename_mp3_files(prefix="_fake", directory_path="."):
    """
    Rename all files in the given directory by adding a custom prefix before the extension.
    
    Args:
        prefix (str): The prefix to add before the extension (e.g., '_fake', '_real', '_processed')
        directory_path (str): Path to the directory containing files to rename
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ Error: Directory '{directory_path}' does not exist.")
        return
    
    if not directory.is_dir():
        print(f"❌ Error: '{directory_path}' is not a directory.")
        return
    
    # Find all files (excluding directories and hidden files)
    all_files = [f for f in directory.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    if not all_files:
        print(f"ℹ️  No files found in '{directory_path}'")
        return
    
    print(f"📁 Found {len(all_files)} file(s) in '{directory_path}'")
    print(f"🏷️  Adding prefix: '{prefix}'")
    print("🔄 Starting rename process...\n")
    
    renamed_count = 0
    skipped_count = 0
    
    for file_item in all_files:
        # Get the filename without extension
        filename_without_ext = file_item.stem
        
        # Skip files that already have the prefix in the name
        if prefix in filename_without_ext:
            print(f"⏭️  Skipping: {file_item.name} (already contains '{prefix}')")
            skipped_count += 1
            continue
        
        # Create new filename - preserve the original extension
        original_extension = file_item.suffix
        new_filename = f"{filename_without_ext}{prefix}{original_extension}"
        new_path = file_item.parent / new_filename
        
        # Check if target file already exists
        if new_path.exists():
            print(f"⚠️  Skipping: {file_item.name} (target file '{new_filename}' already exists)")
            skipped_count += 1
            continue
        
        try:
            # Rename the file
            file_item.rename(new_path)
            print(f"✅ Renamed: {file_item.name} → {new_filename}")
            renamed_count += 1
        except Exception as e:
            print(f"❌ Error renaming {file_item.name}: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"   ✅ Files renamed: {renamed_count}")
    print(f"   ⏭️  Files skipped: {skipped_count}")
    print("🏁 Rename process complete!")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) > 3:
        print("Usage: python rename_files.py [prefix] [directory_path]")
        print("Examples:")
        print("  python rename_files.py _fake")
        print("  python rename_files.py _real /path/to/directory")
        print("  python rename_files.py _processed .")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - use defaults
        prefix = "_fake"
        directory_path = "."
    elif len(sys.argv) == 2:
        # One argument - could be prefix or directory
        arg = sys.argv[1]
        if arg.startswith("_") or arg.startswith("-"):
            # Looks like a prefix
            prefix = arg
            directory_path = "."
        else:
            # Looks like a directory path
            prefix = "_fake"
            directory_path = arg
    else:
        # Two arguments - prefix and directory
        prefix = sys.argv[1]
        directory_path = sys.argv[2]
    
    print("🎵 File Renamer")
    print(f"🏷️  Prefix to add: '{prefix}'")
    print(f"📁 Target directory: {os.path.abspath(directory_path)}\n")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Operation cancelled.")
        return
    
    rename_mp3_files(prefix, directory_path)

if __name__ == "__main__":
    main()
