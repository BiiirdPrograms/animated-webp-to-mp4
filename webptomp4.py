import argparse
import numpy as np
import os
import time

from PIL import Image
from moviepy.video.io import ImageSequenceClip
from multiprocessing import Pool, TimeoutError, cpu_count

DEFAULT_TIMEOUT = 30
MS_IN_SECOND = 1000

MAX_THREADS = 16 # pooling issues crop up above 16 threads
THREADS = int(np.min((cpu_count(), MAX_THREADS)))

def is_image_partial_mode(im):
  # Determine whether image is storing all frames fully or some partially
  partial = False

  for i in range(im.n_frames):
    im.seek(i)
    if im.tile:
      tile = im.tile[0]
      update_region = tile[1]
      update_region_dimensions = update_region[2:]
      if update_region_dimensions != im.size:
        return True
  
  return partial

def process_image_chunk(path, frame_range, size, is_partial):
  img = Image.open(path)

  indexed_image_array = []

  for i in frame_range:
    img.seek(i)

    new_frame = Image.new('RGBA', size)

    new_frame.paste(img, (0, 0), img.convert('RGBA'))

    frame_time = img.info['duration']

    # Storing raw image data for postprocessing partial mode sources
    if is_partial:
      indexed_image_array.append((new_frame, i, frame_time))
    else:
      indexed_image_array.append((np.array(new_frame), i, frame_time))

  return indexed_image_array

def process_image(path, timeout):
  im = Image.open(path)
  is_partial = is_image_partial_mode(im)
  size = im.size
  total_frames = im.n_frames
  im.close()

  pool_size = np.ceil(total_frames / THREADS)
  threads = int(np.ceil(total_frames / pool_size))

  with Pool(processes=threads) as pool:
    frame_ranges = []
    results = []
    for j in range(threads):
      frame_ranges.append(range(int(pool_size * j), int(pool_size * (j + 1) if j < threads - 1 else total_frames)))
      
    results = [pool.apply_async(process_image_chunk, (path, frame_range, size, is_partial)) for frame_range in frame_ranges]
    try:
      indexed_image_arrays = [res.get(timeout=timeout) for res in results]
    except TimeoutError:
      print('Processing timed out after %d seconds. Try a higher --timeout value' % timeout)
      

  indexed_image_array = [ii for iia in indexed_image_arrays for ii in iia]
  indexed_image_array_sorted = sorted(indexed_image_array, key=lambda x: x[1])
  images = []
  frame_times = []
  for img, _, frame_time in indexed_image_array_sorted:
    images.append(img)
    frame_times.append(frame_time)

  # Frame durations given in ms
  fps = MS_IN_SECOND / (sum(frame_times) / len(frame_times))

  # Partial mode source files don't store each image completely so we build them up progressively
  if is_partial:
    pasting_array = []
    for i in range(len(images)):
      new_frame = Image.new('RGBA', size)
      if i > 0:
        new_frame.paste(pasting_array[i - 1])
      new_frame.paste(images[i])
      pasting_array.append(new_frame)
    
    images = [np.array(im) for im in pasting_array]

  return (images, fps)

def webp_mp4(filename, outfile, timeout):
  filename = '/'.join(filename.split('\\'))

  if (outfile is None):
    outfile = '.'.join(filename.split('.')[0:-1])

    index = ''

    while os.path.exists('%s/%s.mp4' % (os.getcwd(), outfile + str(index))):
      if index == '':
        index = 1
      else:
        index += 1

    outfile += str(index) + '.mp4'
  else:
    outfile = '/'.join(outfile.split('\\'))

  file_path = '{}/{}'.format(os.getcwd(), filename)
  if not os.path.exists(file_path):
    print('Can\'t find file %s' % (file_path))
    return

  outdir = '{}/{}'.format(os.getcwd(), '/'.join(outfile.split('/')[0:-1]))
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  t = round(time.time(), 2)

  (images, fps) = process_image(filename, timeout)

  print('Image processed in %0.2f seconds. Detected %0.2f fps' % (round(time.time(), 2) - t, fps))

  clip = ImageSequenceClip.ImageSequenceClip(images, fps=fps)
  clip.write_videofile(outfile, threads=THREADS)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='Animated WebP to mp4 converter',
                                   description='Convert animated webp files into ubiquitous h.264 mp4 video files')

  parser.add_argument('filename')
  parser.add_argument('-o', '--outfile',
                      help='Output file name. Default is the source file name with a .mp4 extension')
  parser.add_argument('-t', '--timeout',
                      help='Timeout in seconds for processing images per thread. Default is 30.',
                      default=DEFAULT_TIMEOUT,
                      type=int)

  args = parser.parse_args()

  webp_mp4(args.filename, args.outfile, args.timeout)
