import os
import io

def consolidate_python_files(output_filename='consolidated_app.py', start_path='.'):
    """
    Traverses subdirectories from the starting path, reads all Python files (.py),
    and combines their contents into a single output file.

    Each file's content is prefixed with a comment indicating its path and name.

    Args:
        output_filename (str): The name of the file to write the consolidated
                               content to (e.g., 'consolidated_app.py').
        start_path (str): The directory to begin the traversal from.
    """
    script_name = os.path.basename(__file__)
    print(f"Starting consolidation from: {os.path.abspath(start_path)}")
    print(f"Writing output to: {output_filename}\n")

    files_collected_count = 0

    # Using 'io.open' for better handling of different encodings (reading in default system encoding)
    try:
        with io.open(output_filename, 'w', encoding='utf-8') as outfile:
            # os.walk yields (dirpath, dirnames, filenames)
            for root, dirs, files in os.walk(start_path):

                # Skip __pycache__ directories and other unwanted folders
                # Modifying 'dirs' in-place will prevent os.walk from descending into them
                if '__pycache__' in dirs:
                    dirs.remove('__pycache__')
                
                # Filter for .py files
                python_files = [f for f in files if f.endswith('.py')]

                for filename in python_files:
                    full_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(full_path, start_path)

                    # Exclude the collector script itself and the output file
                    if filename == script_name or filename == output_filename:
                        print(f"Skipping collector/output file: {relative_path}")
                        continue

                    try:
                        # Read the content of the source file
                        with io.open(full_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()

                        # Write the file header (path and filename)
                        header = f"\n# =========================================================================\n"
                        header += f"# FILE START: {relative_path}\n"
                        header += f"# =========================================================================\n\n"
                        outfile.write(header)

                        # Write the file content
                        outfile.write(content)
                        outfile.write('\n\n') # Add extra separation between files

                        files_collected_count += 1
                        print(f"  -> Added: {relative_path}")

                    except IOError as e:
                        print(f"Error reading file {relative_path}: {e}")
                    except UnicodeDecodeError as e:
                        print(f"Encoding error in file {relative_path}: {e}")


        print(f"\nConsolidation complete! {files_collected_count} files successfully combined.")

    except IOError as e:
        print(f"\nFATAL ERROR: Could not write to output file {output_filename}. {e}")


if __name__ == '__main__':
    # Run from the current directory ('.') and output to 'consolidated_app.py'
    consolidate_python_files()