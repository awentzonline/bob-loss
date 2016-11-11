#!/usr/bin/env bash

echo "Extracting frames from season $1"
season_path=./frames/season-$1
for input_full_filename in video/season-$1/*; do
  input_filename=$(basename ${input_full_filename})
  output_path="${season_path}/${input_filename%.*}"
  mkdir -p "$output_path"
  echo "$output_path/%05d.jpg"
  ffmpeg -i "$input_full_filename" -r 1 "$output_path/%05d.jpg"
done
