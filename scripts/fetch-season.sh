#!/usr/bin/env bash

season_urls=(
  ""
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi69x_7JbQvSMprLRK_KSVLu"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi69x_7JbQvSMprLRK_KSVLu"
)
URL=${season_urls[$1]}

echo "Downloading season $1 $URL"
youtube-dl --yes-playlist -f worstvideo -o "./season$1/%(playlist_index)s.%(ext)s" $URL
