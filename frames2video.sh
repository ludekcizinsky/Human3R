#!/bin/bash

# Exit on error
set -e

seq_name="boxing"
FRAMES_FOLDER="/scratch/izar/cizinsky/zurihack/human3r/$seq_name/color_smpl"
OUTPUT_VIDEO="/scratch/izar/cizinsky/zurihack/human3r/$seq_name/${seq_name}_video.mp4"
FPS="15"

# Create parent directory for output if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_VIDEO")"

# Build video:
# - Read frames as a sequence
# - From each frame, crop 3 equal-width columns:
#     col1: x=0
#     col2: x=iw/4
#     col4: x=3*iw/4
# - Stack them horizontally: [col1 | col2 | col4]
#
# Note: Uses split=3 to fan out the stream, then crop each.
# If your frame width isn't divisible by 4, ffmpeg will still handle it,
# but if you ever hit odd-dimension issues with yuv420p, uncomment the
# 'even_w' line to force even widths.

ffmpeg -framerate "$FPS" \
  -pattern_type glob -i "${FRAMES_FOLDER}/*.png" \
  -filter_complex "\
    [0:v]split=3[a][b][c]; \
    [a]crop=w=iw/4:h=ih:x=0:y=0[a1]; \
    [b]crop=w=iw/4:h=ih:x=iw/4:y=0[b1]; \
    [c]crop=w=iw/4:h=ih:x=3*iw/4:y=0[c1]; \
    [a1][b1][c1]hstack=3[outv]" \
  -map "[outv]" \
  -c:v libx264 -pix_fmt yuv420p -preset fast -crf 20 \
  "$OUTPUT_VIDEO"

echo "✅ Video created successfully at: $OUTPUT_VIDEO"
