# denoise_audio
denoising a .wav audio file using fft.

some codes were adopted from: https://realpython.com/python-scipy-fft/

## usage
### 1. install dependencies listsed in the requirements.txt
### 2. run the script
```
python3 audio_filter path_to_root_directory name_of_the_file_without_extension
```
**the input audio file must be a .wav file.

#### convert .mp4 file to .wav (ubuntu using ffmpeg)
```
ffmpeg -i your.MP4 -ac 2 -f wav your.wav

# if there are multiple video files to convert in a directory
for FILE in *; do ffmpeg -i $FILE -ac 2 -f wav ${FILE//.MP4/.wav}; done
```
