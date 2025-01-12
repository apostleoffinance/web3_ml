#!/bin/bash

# Project Zip Creation Script

# Set the project directory and output zip filename
PROJECT_DIR="$(pwd)"
ZIP_FILENAME="project_bundle.zip"

# List of files to include in the zip
FILES_TO_ZIP=(
    "Dockerfile"
    "requirements.txt"
    "predict.py"
    "logistic_regression_model.bin"
)

# Verify all required files exist
for file in "${FILES_TO_ZIP[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found!"
        exit 1
    fi
done

# Create zip file without compression
zip -0 "$ZIP_FILENAME" "${FILES_TO_ZIP[@]}"

# Verify the zip file contents
echo "Created zip file: $ZIP_FILENAME"
unzip -l "$ZIP_FILENAME"

# Optional: Print out the zip file size
echo "Zip file size:"
du -h "$ZIP_FILENAME"