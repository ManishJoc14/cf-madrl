import zipfile
import os


def pack_project():
    output_filename = "traffic_cf_madrl_colab.zip"
    # Exclude these directories
    excludes = {
        ".venv",
        "venv"
        "__pycache__",
        ".git",
        ".vscode",
        "pi"
    }

    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in excludes]

            for file in files:
                if file == output_filename:
                    continue
                if file.endswith(".pyc"):
                    continue

                file_path = os.path.join(root, file)
                # Keep the folder structure simple
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)

    print(f"Created {output_filename} successfully!")


if __name__ == "__main__":
    pack_project()
