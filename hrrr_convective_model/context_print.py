import os
from pathlib import Path
from datetime import datetime

# Directory you want to explore
BASE_DIR = Path('/home/ubuntu2/diff-pro/diff-o3-pro/hrrr_convective_model')

# Output context file
OUTPUT_FILE = BASE_DIR / 'project_context_dump.txt'

# File extensions to include content
INCLUDE_EXTENSIONS = ['.py', '.yaml', '.yml', '.json', '.sh', '.md']

# Directories to skip
SKIP_DIRS = {'__pycache__', '.git', 'data/raw', 'data/zarr', 'dask-worker-space'}

# Files to skip
SKIP_FILES = {'*.pyc', '*.pt', '*.grib2', '*.idx'}

def should_skip_dir(dirname):
    return dirname in SKIP_DIRS

def should_skip_file(filename):
    return any(filename.endswith(ext) for ext in ['.pyc', '.pt', '.grib2', '.idx'])

def write_file_structure_and_contents(base_dir, output_file):
    with open(output_file, 'w') as out:
        out.write(f"HRRR Convective Model - Project Context\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write("=" * 80 + "\n\n")
        
        out.write("Directory Structure:\n")
        out.write("=" * 80 + "\n")

        for root, dirs, files in os.walk(base_dir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            
            level = root.replace(str(base_dir), '').count(os.sep)
            indent = ' ' * 4 * level
            out.write(f'{indent}{os.path.basename(root)}/\n')
            subindent = ' ' * 4 * (level + 1)
            
            # Sort files for consistent output
            for file in sorted(files):
                if not should_skip_file(file):
                    out.write(f'{subindent}{file}\n')

        out.write("\n\nFile Contents:\n")
        out.write("=" * 80 + "\n\n")

        # Collect and sort files for consistent output
        all_files = []
        for root, dirs, files in os.walk(base_dir):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            
            for file in files:
                if Path(file).suffix in INCLUDE_EXTENSIONS and not should_skip_file(file):
                    filepath = Path(root) / file
                    rel_path = filepath.relative_to(base_dir)
                    all_files.append((rel_path, filepath))
        
        # Sort by relative path for consistent ordering
        all_files.sort(key=lambda x: str(x[0]))
        
        # Write file contents
        for rel_path, filepath in all_files:
            out.write(f'File: {rel_path}\n')
            out.write("-" * 80 + "\n")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                out.write(content + "\n")
            except Exception as e:
                out.write(f"Failed to read file: {e}\n")
            out.write("\n" + "=" * 80 + "\n\n")
        
        # Summary statistics
        out.write("\n\nSummary:\n")
        out.write("=" * 80 + "\n")
        out.write(f"Total files included: {len(all_files)}\n")
        out.write(f"File types: {', '.join(sorted(set(Path(f[0]).suffix for f in all_files)))}\n")

# Run the script
write_file_structure_and_contents(BASE_DIR, OUTPUT_FILE)

print(f"Context file written to {OUTPUT_FILE}")
print(f"Total files processed: {len([f for f in Path(BASE_DIR).rglob('*') if f.is_file() and f.suffix in INCLUDE_EXTENSIONS and not should_skip_file(f.name)])}")
