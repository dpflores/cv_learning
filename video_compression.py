import os

input_file= 'output.avi'
output_file= 'output_compressed.mp4'
crf = 28 # higher crf reduces the file size 

os.system(f'ffmpeg -i {input_file} -vcodec libx264 -crf {crf} {output_file}')
