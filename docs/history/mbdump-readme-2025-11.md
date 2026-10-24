# Record Scan

## What do the files mean

### Python

`dev/test_cuda.py`: Dev file to see if device supports Cuda
`dev/count_files.py`: Dev file to count the number of images in our giant image repo
`dev/count_existing_files.py`: Dev file that loads existing_files.json and checks how many there are

`src/fetch-covers/slim_file.py`: Prod file which, given full data for all musicbrainz releases (~250gb) slims it down into just data we care about, e.g. album id, artist id(s)
`src/fetch-covers/fetch_cover_art_urls.py`: Prod file which, given list of album IDs, fetches associated cover art IDs if it exists
`src/fetch-covers/parallel_fetch_cover_art_urls.py`: Prod file which does same as above but parallelized
`src/fetch-covers/fetch_cover_art.py`: Prod file which, given list of album IDs with associated cover art ID, downloads the 500px cover art
`src/fetch-covers/move_images_to_subdirectories.py`: Prod file which moves images to subdirectories based on first 2 characters of album ID.

### JSON

`src/fetch-covers/slim_releases.json`: Prod file which is full releases data (~250gb) slimmed down to just fields we care about
`src/fetch-covers/slim_releases_with_covers.json`: Prod file which is above + cover art ID for albums which have it
`src/fetch-covers/release_slim_artists.json`: Prod file which maps artist IDs to their names + aliases + disambiguations
`src/fetch-covers/failed_cover_id_fetches.json`: List of album IDs for which we failed to fetch cover IDs
`src/fetch-covers/failed_cover_fetches.json`: List of album IDs for which we failed to fetch cover images
`src/fetch-covers/existing_files.json`: List of album IDs for which we've managed to download images

`dev/line1.json`: Dev file which shows what data is contained in 1 entry in the 250gb releases data file
`dev/line1_ideal.json`: Dev file which shows what slimmed down data we want
