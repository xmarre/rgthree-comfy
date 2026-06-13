import ctypes
import gc
import json
import math
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from nodes import PreviewImage

try:
  from comfy.cli_args import args
except Exception:  # pragma: no cover - compatibility with older ComfyUI layouts.
  args = None

from .constants import get_category, get_name


def _env_int(name, default):
  try:
    return int(os.environ.get(name, default))
  except (TypeError, ValueError):
    return default


def _env_bool(name, default=False):
  value = os.environ.get(name)
  if value is None:
    return default
  return value.strip().lower() not in ("0", "false", "no", "off", "")


def _debug(message):
  if _env_bool("RGTHREE_IMAGE_COMPARER_DEBUG", False):
    print(f"[rgthree][ImageComparer] {message}", flush=True)


def _malloc_trim():
  if not _env_bool("RGTHREE_IMAGE_COMPARER_MALLOC_TRIM", True):
    return
  try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
  except Exception:
    pass


def _metadata(prompt=None, extra_pnginfo=None):
  if args is not None and getattr(args, "disable_metadata", False):
    return None

  metadata = PngInfo()
  if prompt is not None:
    metadata.add_text("prompt", json.dumps(prompt))
  if extra_pnginfo is not None:
    for key in extra_pnginfo:
      metadata.add_text(key, json.dumps(extra_pnginfo[key]))
  return metadata


def _image_tensor_to_uint8(image, chunk_pixels):
  """Convert one HWC IMAGE tensor frame to uint8 with bounded float temporaries."""
  image_np = image.detach().cpu().numpy()
  height, width = int(image_np.shape[0]), int(image_np.shape[1])
  rows = max(1, min(height, chunk_pixels // max(1, width)))

  result = np.empty(image_np.shape, dtype=np.uint8)
  scratch = None
  for y0 in range(0, height, rows):
    y1 = min(height, y0 + rows)
    source = image_np[y0:y1]
    if scratch is None or scratch.shape != source.shape:
      scratch = np.empty(source.shape, dtype=np.float32)
    np.multiply(source, 255.0, out=scratch, casting="unsafe")
    np.clip(scratch, 0.0, 255.0, out=scratch)
    result[y0:y1] = scratch
  return result


def _resize_for_preview(img, max_preview_pixels):
  # Default is 0: preserve full-resolution comparer previews and all visual detail.
  # Set RGTHREE_IMAGE_COMPARER_MAX_PREVIEW_PIXELS explicitly to enable downsizing.
  if max_preview_pixels <= 0:
    return img

  width, height = img.size
  pixels = width * height
  if pixels <= max_preview_pixels:
    return img

  scale = math.sqrt(max_preview_pixels / pixels)
  new_width = max(1, int(round(width * scale)))
  new_height = max(1, int(round(height * scale)))
  return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


class RgthreeImageComparer(PreviewImage):
  """A node that compares two images in the UI."""

  NAME = get_name('Image Comparer')
  CATEGORY = get_category()
  FUNCTION = "compare_images"
  DESCRIPTION = "Compares two images with a hover slider, or click from properties."

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {},
      "optional": {
        "image_a": ("IMAGE",),
        "image_b": ("IMAGE",),
      },
      "hidden": {
        "prompt": "PROMPT",
        "extra_pnginfo": "EXTRA_PNGINFO"
      },
    }

  def _save_images_lowmem(self, images, filename_prefix="rgthree.compare.", prompt=None, extra_pnginfo=None):
    """Save every comparer preview sequentially with bounded conversion temporaries.

    This preserves rgthree Image Comparer's normal batch behavior by default: every
    image in image_a and every image in image_b is saved and returned to the UI.

    Environment controls:
      RGTHREE_IMAGE_COMPARER_MAX_IMAGES_PER_INPUT: default 0, <=0 saves all frames.
      RGTHREE_IMAGE_COMPARER_MAX_PREVIEW_PIXELS: default 0, <=0 preserves full resolution.
      RGTHREE_IMAGE_COMPARER_CHUNK_PIXELS: default 262144 conversion temp size.
      RGTHREE_IMAGE_COMPARER_MALLOC_TRIM: default 1 on Linux/glibc.
      RGTHREE_IMAGE_COMPARER_DEBUG: default 0, prints save progress.
    """
    filename_prefix += self.prefix_append

    max_images = _env_int("RGTHREE_IMAGE_COMPARER_MAX_IMAGES_PER_INPUT", 0)
    max_preview_pixels = _env_int("RGTHREE_IMAGE_COMPARER_MAX_PREVIEW_PIXELS", 0)
    chunk_pixels = max(4096, _env_int("RGTHREE_IMAGE_COMPARER_CHUNK_PIXELS", 262144))

    if max_images > 0:
      images_to_save = images[:max_images]
    else:
      images_to_save = images

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
      filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

    results = []
    total = len(images_to_save)
    for batch_number, image in enumerate(images_to_save):
      _debug(f"saving {filename_prefix} batch={batch_number + 1}/{total} shape={tuple(image.shape)} "
             f"dtype={image.dtype} device={image.device}")
      arr = _image_tensor_to_uint8(image, chunk_pixels)
      img = Image.fromarray(arr)
      original_size = img.size
      img = _resize_for_preview(img, max_preview_pixels)

      filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
      file = f"{filename_with_batch_num}_{counter:05}_.png"
      img.save(os.path.join(full_output_folder, file), pnginfo=_metadata(prompt, extra_pnginfo),
               compress_level=self.compress_level)
      _debug(f"saved {filename_prefix} batch={batch_number + 1}/{total} original_size={original_size} "
             f"preview_size={img.size} file={file}")
      results.append({
        "filename": file,
        "subfolder": subfolder,
        "type": self.type,
      })
      counter += 1

      del img, arr
      gc.collect()
      _malloc_trim()

    return { "ui": { "images": results } }

  def compare_images(self,
                     image_a=None,
                     image_b=None,
                     filename_prefix="rgthree.compare.",
                     prompt=None,
                     extra_pnginfo=None):

    result = { "ui": { "a_images":[], "b_images": [] } }
    if _env_bool("RGTHREE_IMAGE_COMPARER_DISABLED", False):
      _debug("disabled by RGTHREE_IMAGE_COMPARER_DISABLED=1")
      return result

    if image_a is not None and len(image_a) > 0:
      result['ui']['a_images'] = self._save_images_lowmem(
        image_a, f"{filename_prefix}a.", prompt, extra_pnginfo)['ui']['images']

    if image_b is not None and len(image_b) > 0:
      result['ui']['b_images'] = self._save_images_lowmem(
        image_b, f"{filename_prefix}b.", prompt, extra_pnginfo)['ui']['images']

    gc.collect()
    _malloc_trim()
    return result
