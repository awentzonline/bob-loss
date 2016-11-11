#!/usr/bin/env bash

season_urls=(
  "not a season"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi69x_7JbQvSMprLRK_KSVLu"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi5VAEOviVE6svrUW2axISf6"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi7zDD6O36FKkEHse-JCdVvh"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi56KMlB1P_wK9pJZcMpEFQd"
  "https://www.youtube.com/playlist?list=PLAEQD0ULngi6bAFRfcqgpKP4T4SnoxoAz"
)
URL=${season_urls[$1]}


echo "Downloading season $1"
youtube-dl --yes-playlist -f worstvideo \
  -o "./video/season-$1/%(playlist_index)s.%(ext)s" \
  $URL
