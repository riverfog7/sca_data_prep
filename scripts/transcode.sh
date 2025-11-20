#!/usr/bin/env bash

VIDEO_DIR=${1:-"./video_downloads"}
OUTPUT_DIR=${2:-"./audio_outputs"}
IN_EXT=${3:-"mkv"}
OUT_EXT=${4:-"flac"}

SAMPLE_RATE=16000
BIT_DEPTH=16

if [ ! -x "$(command -v ffmpeg)" ]; then
  echo "ffmpeg is not installed. Please install ffmpeg to use this script."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
if [ ! -d "${VIDEO_DIR}" ]; then
  echo "Video directory ${VIDEO_DIR} does not exist. Exiting."
  exit 1
fi

for video_file in "${VIDEO_DIR}"/*."${IN_EXT}"; do
  if [ ! -f "${video_file}" ]; then
    continue
  fi

  base_name=$(basename "${video_file}" ."${IN_EXT}")
  output_file="${OUTPUT_DIR}/${base_name}.${OUT_EXT}"

  echo "Transcoding ${video_file} to ${output_file}..."
  ffmpeg -i "${video_file}" -vn -ar "${SAMPLE_RATE}" -ac 2 -sample_fmt s"${BIT_DEPTH}" "${output_file}"

  if [ $? -ne 0 ]; then
    echo "Error transcoding ${video_file}. Skipping."
    continue
  fi

  echo "Successfully transcoded to ${output_file}."
done
