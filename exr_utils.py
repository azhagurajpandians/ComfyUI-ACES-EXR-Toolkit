import os
import numpy as np

def load_exr(filepath):
    """
    Load an EXR file into a numpy float32 array.
    Attempts to use imageio (with freeimage), then OpenEXR, then tifffile.
    Returns array of shape (H, W, 3) or (H, W, 4).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # 1. Try imageio with freeimage plugin
    try:
        import imageio.v3 as iio
        # using the freeimage format usually gives proper 32-bit floats
        arr = iio.imread(filepath, plugin="FI")
        if arr is not None and arr.dtype in (np.float32, np.float16, np.float64):
            return arr.astype(np.float32)
        elif arr is not None:
             # Just in case it reads it as something else but didn't throw
             return arr.astype(np.float32)
    except Exception as e:
        pass

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
        pass

    # 3. Try tifffile (can sometimes handle simple EXRs, mostly TIFFs)
    try:
        import tifffile
        arr = tifffile.imread(filepath)
        return arr.astype(np.float32)
    except Exception as e:
        pass

    # 4. Try standard imageio (might return 8-bit or 16-bit int if no plugins)
    try:
        import imageio.v3 as iio
        arr = iio.imread(filepath)
        # Normalize to 0-1 if it's integer type
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        return arr.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Could not load EXR file {filepath}. Please install imageio[freeimage] or OpenEXR.")
