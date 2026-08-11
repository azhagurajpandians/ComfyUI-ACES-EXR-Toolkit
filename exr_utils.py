import os
import numpy as np

def load_exr(filepath):
    """
    Load an EXR file into a numpy float32 array.
    Attempts to use OpenEXR (Python library), then tifffile as fallback.

    Note: cv2.imread is intentionally skipped for EXR files because the
    OpenCV build shipped with ComfyUI has its EXR codec disabled
    (OPENCV_IO_ENABLE_OPENEXR is not set), which produces noisy warnings
    and always falls back anyway.

    Returns array of shape (H, W, 3) or (H, W, 4).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    errors = []

    # 2. Try OpenEXR
    try:
        import OpenEXR
        import Imath
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        channels = header['channels'].keys()
        chan_data = {}
        
        if 'R' in channels and 'G' in channels and 'B' in channels:
            for c in ['R', 'G', 'B']:
                chan_str = exr_file.channel(c, pt)
                chan_data[c] = np.frombuffer(chan_str, dtype=np.float32).reshape(size[1], size[0])
            
            if 'A' in channels:
                chan_str = exr_file.channel('A', pt)
                chan_data['A'] = np.frombuffer(chan_str, dtype=np.float32).reshape(size[1], size[0])
                arr = np.stack([chan_data['R'], chan_data['G'], chan_data['B'], chan_data['A']], axis=-1)
            else:
                arr = np.stack([chan_data['R'], chan_data['G'], chan_data['B']], axis=-1)
            return arr
    except Exception as e:
        errors.append(f"OpenEXR: {e}")

    # 3. Try tifffile (can sometimes handle simple EXRs, mostly TIFFs)
    try:
        import tifffile
        arr = tifffile.imread(filepath)
        return arr.astype(np.float32)
    except Exception as e:
        errors.append(f"tifffile: {e}")

    raise RuntimeError(f"Could not load EXR file {filepath}. Errors: " + " | ".join(errors))
