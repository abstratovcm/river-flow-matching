#!/bin/bash
cd videos

# Count the total number of MOV files
total=$(ls *.MOV | wc -l)
current=1

for f in *.MOV; do
  remaining=$((total - current))
  echo "Processing: $f ($remaining videos left)"
  output="${f%.*}.mp4"
  # Run ffmpeg quietly; remove redirection if you need ffmpeg's output for debugging
  ffmpeg -y -i "$f" -c:v libx264 -c:a aac "$output" > /dev/null 2>&1
  current=$((current + 1))
done

echo "All videos converted."
