import os
from pathlib import Path

import numpy as np
import torch
import uuid

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import folder_paths
except Exception:
    folder_paths = None


CATEGORY = "image/ACES + EXR"
DEFAULT_OCIO_CONFIG = "ocio://studio-config-v1.0.0_aces-v1.3_ocio-v2.1"

OCIO_SPACE_ALIASES = {
    "sRGB": ["sRGB", "Output - sRGB", "Utility - sRGB - Texture"],
    "Linear sRGB": ["Linear sRGB", "Utility - Linear - sRGB"],
    "ACEScg": ["ACEScg", "ACES - ACEScg"],
    "ACES2065-1": ["ACES2065-1", "ACES - ACES2065-1"],
}

COMPRESSIONS = {
    "ZIP  (lossless, recommended)":  "ZIP_COMPRESSION",
    "ZIPS (lossless, scanline)":     "ZIPS_COMPRESSION",
    "PIZ  (lossless, wavelet)":      "PIZ_COMPRESSION",
    "PXR24 (lossy, 24-bit)":         "PXR24_COMPRESSION",
    "RLE  (lossless, run-length)":   "RLE_COMPRESSION",
    "B44  (lossy, fixed ratio)":     "B44_COMPRESSION",
    "DWAA (lossy, DCT, fast)":       "DWAA_COMPRESSION",
    "None (uncompressed)":           "NO_COMPRESSION",
}
COMPRESSION_KEYS = list(COMPRESSIONS.keys())
BIT_DEPTHS = ["16f (half-float)", "32f (float)"]

OCIO_SPACES = [
    "sRGB",
    "Linear sRGB",
    "ACEScg",
    "ACES2065-1",
    "Utility - sRGB - Texture",
    "Utility - Linear - sRGB",
    "ACES - ACEScg",
    "ACES - ACES2065-1",
    "Camera - ARRI - LogC3 EI800 Wide Gamut",
    "Camera - ARRI - LogC4 Wide Gamut 4",
    "Camera - Sony - S-Log3 S-Gamut3.Cine",
    "Camera - RED - Log3G10 REDWideGamutRGB",
]

SRGB_TO_XYZ_D65 = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

XYZ_D65_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ_D65).astype(np.float32)

ACESCG_TO_XYZ_D60 = np.array(
    [
        [0.6624541811, 0.1340042065, 0.1561876870],
        [0.2722287168, 0.6740817658, 0.0536895174],
        [-0.0055746495, 0.0040607335, 1.0103391003],
    ],
    dtype=np.float32,
)

XYZ_D60_TO_ACESCG = np.linalg.inv(ACESCG_TO_XYZ_D60).astype(np.float32)

BRADFORD_D65_TO_D60 = np.array(
    [
        [1.01303493, 0.00610531, -0.01497100],
        [0.00769823, 0.99816579, -0.00503203],
        [-0.00284131, 0.00468516, 0.92450614],
    ],
    dtype=np.float32,
)

BRADFORD_D60_TO_D65 = np.linalg.inv(BRADFORD_D65_TO_D60).astype(np.float32)


def _input_dir() -> str:
    if folder_paths is None:
        return os.getcwd()
    return folder_paths.get_input_directory()


def _output_dir() -> str:
    if folder_paths is None:
        return os.getcwd()
    return folder_paths.get_output_directory()


def _temp_dir() -> str:
    if folder_paths is None:
        return os.getcwd()
    return folder_paths.get_temp_directory()


def _get_all_allowed_dirs() -> list:
    """Return all permitted directories: input, output, temp, plus all directories
    configured in ComfyUI (e.g., via extra_model_paths.yaml)."""
    dirs = []
    for d in [_input_dir(), _output_dir(), _temp_dir()]:
        if d and d not in dirs:
            dirs.append(d)
    if folder_paths is not None and hasattr(folder_paths, "folder_names_and_paths"):
        for name, entry in folder_paths.folder_names_and_paths.items():
            if isinstance(entry, (list, tuple)) and len(entry) > 0:
                paths = entry[0]
                if isinstance(paths, (list, tuple, set)):
                    for p in paths:
                        if p and p not in dirs:
                            dirs.append(p)
                elif isinstance(paths, (str, Path)) and paths not in dirs:
                    dirs.append(str(paths))
    return dirs


def _assert_path_allowed(path: Path, allowed_dirs: list, action: str = "access") -> None:
    """Raise PermissionError if *path* is not contained inside one of *allowed_dirs*.

    Uses os.path.realpath to resolve symlinks / '..' components before comparing,
    so traversal attacks such as '../../../etc/passwd' are blocked.
    """
    real = os.path.realpath(str(path))
    for allowed in allowed_dirs:
        real_allowed = os.path.realpath(str(allowed))
        try:
            if os.path.commonpath([real, real_allowed]) == real_allowed:
                return  # contained — OK
        except ValueError:
            # commonpath raises ValueError on Windows when paths have different drives
            pass
    allowed_strs = "\n  - " + "\n  - ".join(str(d) for d in allowed_dirs)
    raise PermissionError(
        f"Security: refusing to {action} '{path}'.\n"
        f"Path must be inside one of the allowed directories:{allowed_strs}\n"
        f"To allow additional directories (e.g. render folders or external drives), add them to extra_model_paths.yaml in ComfyUI."
    )


def _assert_exr_read_allowed(path: Path) -> None:
    """Confines reads to ComfyUI allowed directories (input/output/temp/configured extra paths).
    For external render drives, allows valid .exr or .hdr files while blocking non-EXR system file reads.
    """
    allowed = _get_all_allowed_dirs()
    real = os.path.realpath(str(path))
    for allowed_dir in allowed:
        real_allowed = os.path.realpath(str(allowed_dir))
        try:
            if os.path.commonpath([real, real_allowed]) == real_allowed:
                return  # contained in ComfyUI allowed dirs — OK
        except ValueError:
            pass

    # For external drives: strictly require .exr or .hdr file extension
    ext = path.suffix.lower()
    if ext in (".exr", ".hdr") and path.exists():
        return  # valid EXR file — OK

    _assert_path_allowed(path, allowed, action="read")


def _list_input_exrs():
    root = Path(_input_dir())
    if not root.exists():
        return ["<input folder not found>"]
    files = [str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.suffix.lower() == ".exr"]
    return files or ["<select EXR or paste full path below>"]


def _resolve_input_path(filename: str, exr_path: str = "") -> Path:
    selected = (filename or "").strip()
    typed = (exr_path or "").strip().strip('"')
    value = typed or selected
    if not value or value.startswith("<"):
        raise FileNotFoundError(
            "No EXR file selected. Put a .exr file in ComfyUI/input and restart/refresh ComfyUI, "
            "or paste the full .exr path into exr_path."
        )
    path = Path(value)
    if not path.is_absolute():
        path = Path(_input_dir()) / path
    if path.is_dir():
        raise IsADirectoryError(
            f"Expected an .exr file but got a folder: {path}. "
            "Select a specific .exr file or paste the full file path into exr_path."
        )
    if path.suffix.lower() != ".exr":
        raise ValueError(f"Expected an .exr file, got: {path}")
    return path


def _image_to_tensor(image: np.ndarray, alpha_mode: str):
    image = np.asarray(image, dtype=np.float32)
    # Ensure shape is [B, H, W, C]
    if image.ndim == 2:
        image = image[None, :, :, None]
    elif image.ndim == 3:
        image = image[None, ...]
        
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    B, H, W, C = image.shape
    alpha = None
    if C >= 4:
        alpha = np.clip(image[..., 3].copy(), 0.0, 1.0)
        rgb = image[..., :3].copy()
        if alpha_mode == "unpremultiply":
            denom = np.maximum(alpha[..., None], 1.0e-6)
            rgb = np.where(alpha[..., None] > 1.0e-6, rgb / denom, rgb)
        elif alpha_mode == "premultiply":
            rgb = rgb * alpha[..., None]
        elif alpha_mode.startswith("composite"):
            if alpha_mode == "composite black":
                background = np.zeros_like(rgb)
            elif alpha_mode == "composite white":
                background = np.ones_like(rgb)
            elif alpha_mode == "composite gray":
                background = np.full_like(rgb, 0.5)
            else:
                yy, xx = np.indices((H, W))
                checker = (((xx // 64) + (yy // 64)) % 2).astype(np.float32)
                checker = checker * 0.35 + 0.35
                background = np.repeat(checker[None, :, :, None], 3, axis=3)
                background = np.repeat(background, B, axis=0)
            rgb = rgb * alpha[..., None] + background * (1.0 - alpha[..., None])
    else:
        rgb = image[..., :3].copy()

    mask = np.zeros((B, H, W), dtype=np.float32) if alpha is None else 1.0 - np.clip(alpha, 0.0, 1.0)
    return torch.from_numpy(rgb), torch.from_numpy(mask)


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[None, ...]
    return np.asarray(arr, dtype=np.float32)


def _read_exr(path: Path) -> np.ndarray:
    errors = []
    try:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2 returned no image")
        if img.ndim == 3 and img.shape[-1] >= 3:
            order = [2, 1, 0] + list(range(3, img.shape[-1]))
            img = img[:, :, order]
        return img
    except Exception as exc:
        errors.append(f"opencv: {exc}")

    try:
        import OpenEXR
        import Imath
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        exr_file = OpenEXR.InputFile(str(path))
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
    except Exception as exc:
        errors.append(f"OpenEXR: {exc}")

    try:
        import tifffile
        img = tifffile.imread(str(path))
        return img.astype(np.float32)
    except Exception as exc:
        errors.append(f"tifffile: {exc}")

    raise RuntimeError(
        "Could not read EXR. Backend errors: " + " | ".join(errors)
    )

def _detect_exr_sequence(path_str):
    import re
    path = Path(path_str)
    if not path.is_file():
        return [], 0, 0
    
    match = re.search(r'([._-]?)(0*\d+)(\.exr)$', path.name, re.IGNORECASE)
    if not match:
        return [path], 1, 1
        
    prefix = path.name[:match.start(2)]
    suffix = path.name[match.end(2):]
    
    dir_path = path.parent
    seq_files = []
    
    for f in dir_path.glob(f"*{suffix}"):
        if f.name.startswith(prefix) and f.name.endswith(suffix):
            num_str = f.name[len(prefix):-len(suffix)]
            if num_str.isdigit():
                seq_files.append((int(num_str), f))
                
    if not seq_files:
        return [path], 1, 1
        
    seq_files.sort(key=lambda x: x[0])
    paths = [f[1] for f in seq_files]
    return paths, seq_files[0][0], seq_files[-1][0]

def _load_exr_sequence(path_str, frame_mode, start_frame, end_frame, missing_policy):
    paths, seq_first, seq_last = _detect_exr_sequence(path_str)
    if not paths:
        raise FileNotFoundError(f"No sequence found at: {path_str}")
        
    import re
    def get_num(p):
        m = re.search(r'([._-]?)(0*\d+)(\.exr)$', p.name, re.IGNORECASE)
        return int(m.group(2)) if m else 0
        
    path_map = {get_num(p): p for p in paths}
    
    if frame_mode == "single":
        f_start = f_end = start_frame
    elif frame_mode == "range":
        f_start, f_end = start_frame, end_frame
    else: # all
        f_start, f_end = seq_first, seq_last
        
    frames = []
    last_good = None
    for f in range(f_start, f_end + 1):
        if f in path_map:
            img = _read_exr(path_map[f])
            frames.append(img)
            last_good = img
        else:
            if missing_policy == "error":
                raise FileNotFoundError(f"Missing frame {f} in sequence.")
            elif missing_policy == "black":
                if last_good is not None:
                    frames.append(np.zeros_like(last_good))
                else:
                    frames.append(None) # resolve later
            elif missing_policy == "hold":
                if last_good is not None:
                    frames.append(last_good)
                else:
                    frames.append(None) # resolve later
                    
    # resolve missing leading frames
    first_good = next((x for x in frames if x is not None), None)
    if first_good is None:
        raise RuntimeError("Entire sequence is missing or invalid.")
    
    for i in range(len(frames)):
        if frames[i] is None:
            frames[i] = np.zeros_like(first_good) if missing_policy == "black" else first_good
            
    seq_arr = np.stack(frames, axis=0) # [B, H, W, C]
    return seq_arr, len(frames), f_start, f_end


def _read_exr_all_layers(path: Path):
    """Reads all channels from an EXR file and groups them into layers.
    Returns: (dict_of_layer_arrays, list_of_layer_names)
    """
    try:
        import OpenEXR
        import Imath
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header['dataWindow']
        W, H = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        channels = list(header['channels'].keys())
        chan_data = {}
        for c in channels:
            chan_str = exr_file.channel(c, pt)
            chan_data[c] = np.frombuffer(chan_str, dtype=np.float32).reshape(H, W)

        # Group channels by layer prefix
        groups = {}
        for c in channels:
            if '.' in c:
                prefix, sub = c.split('.', 1)
            else:
                prefix, sub = 'beauty', c
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][sub] = chan_data[c]

        layers = {}
        for layer_name, ch_dict in groups.items():
            if all(k in ch_dict for k in ['R', 'G', 'B']):
                ch_list = [ch_dict['R'], ch_dict['G'], ch_dict['B']]
                if 'A' in ch_dict:
                    ch_list.append(ch_dict['A'])
                layers[layer_name] = np.stack(ch_list, axis=-1)
            elif all(k in ch_dict for k in ['X', 'Y', 'Z']):
                layers[layer_name] = np.stack([ch_dict['X'], ch_dict['Y'], ch_dict['Z']], axis=-1)
            elif 'Z' in ch_dict:
                layers[layer_name] = ch_dict['Z'][:, :, None]
            elif 'A' in ch_dict and len(ch_dict) == 1:
                layers[layer_name] = ch_dict['A'][:, :, None]
            else:
                sorted_keys = sorted(ch_dict.keys())
                layers[layer_name] = np.stack([ch_dict[k] for k in sorted_keys], axis=-1)

        return layers, list(layers.keys())
    except Exception:
        arr = _read_exr(path)
        return {"beauty": arr}, ["beauty"]


def _read_exr_header_info(path: Path):
    """Returns (meta_dict, width, height, channel_list, compression_name)"""
    try:
        import OpenEXR
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header['dataWindow']
        W, H = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        channels = list(header['channels'].keys())
        comp = str(header.get('compression', 'unknown'))

        meta = {}
        for k, v in header.items():
            if k not in ['channels', 'dataWindow', 'displayWindow']:
                meta[str(k)] = str(v)
        return meta, W, H, channels, comp
    except Exception:
        arr = _read_exr(path)
        H, W = arr.shape[:2]
        C = arr.shape[2] if arr.ndim == 3 else 1
        chans = ["R", "G", "B", "A"][:C]
        return {"backend": "opencv"}, W, H, chans, "Unknown"


def _write_exr_openexr_multilayer(channels_dict: dict, path: str, use_half: bool, compression_key: str):
    import OpenEXR
    import Imath
    
    comp_name = COMPRESSIONS.get(compression_key, "ZIP_COMPRESSION")
    imath_comp = getattr(Imath.Compression, comp_name)
    ptype_write = Imath.PixelType(Imath.PixelType.HALF if use_half else Imath.PixelType.FLOAT)
    dtype_write = np.float16 if use_half else np.float32

    first_arr = next(iter(channels_dict.values()))
    H, W = first_arr.shape[:2]

    header = OpenEXR.Header(W, H)
    header["compression"] = Imath.Compression(imath_comp)
    header["channels"] = {ch: Imath.Channel(ptype_write) for ch in channels_dict.keys()}

    out = OpenEXR.OutputFile(path, header)
    pixels = {}
    for ch, arr in channels_dict.items():
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        pixels[ch] = arr.astype(dtype_write).tobytes()
    out.writePixels(pixels)
    out.close()


def _resize_pass_to_target(pass_arr: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """Resize pass_arr [pH, pW, C] or [pH, pW] to match target [target_H, target_W, C]."""
    pH, pW = pass_arr.shape[:2]
    if pH == target_H and pW == target_W:
        return pass_arr
    try:
        import cv2
        return cv2.resize(pass_arr, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
    except Exception:
        p_tensor = torch.from_numpy(pass_arr)
        if p_tensor.ndim == 2:
            p_tensor = p_tensor[None, None, :, :]
        elif p_tensor.ndim == 3:
            p_tensor = p_tensor.permute(2, 0, 1)[None, :, :, :]
        res = torch.nn.functional.interpolate(p_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False)
        res_np = res[0].permute(1, 2, 0).cpu().numpy()
        return res_np if pass_arr.ndim == 3 else res_np[:, :, 0]


def _write_exr(
    path: Path,
    image: np.ndarray,
    bit_depth: str = "32f (float)",
    compression: str = "None (uncompressed)",
    extra_passes: dict = None
):
    errors = []
    path.parent.mkdir(parents=True, exist_ok=True)
    use_half = "16f" in bit_depth

    # Standard beauty channels
    H, W = image.shape[:2]
    C = image.shape[2] if image.ndim == 3 else 1
    channels_dict = {}

    if C == 4:
        channels_dict["R"] = image[:, :, 0]
        channels_dict["G"] = image[:, :, 1]
        channels_dict["B"] = image[:, :, 2]
        channels_dict["A"] = image[:, :, 3]
    elif C == 3:
        channels_dict["R"] = image[:, :, 0]
        channels_dict["G"] = image[:, :, 1]
        channels_dict["B"] = image[:, :, 2]
    else:
        channels_dict["R"] = image[:, :, 0] if image.ndim == 3 else image
        channels_dict["G"] = channels_dict["R"]
        channels_dict["B"] = channels_dict["R"]

    if extra_passes:
        for layer_name, pass_arr in extra_passes.items():
            if pass_arr is None:
                continue
            if pass_arr.ndim == 2:
                pass_arr = pass_arr[:, :, None]

            # Automatically resize pass array to match beauty pass (H, W) if dimensions differ
            pH, pW = pass_arr.shape[:2]
            if pH != H or pW != W:
                pass_arr = _resize_pass_to_target(pass_arr, H, W)
                if pass_arr.ndim == 2:
                    pass_arr = pass_arr[:, :, None]

            p_C = pass_arr.shape[2]
            if layer_name in ["depth", "Z"]:
                # Depth pass: store strictly as 1-channel depth.Z
                if p_C >= 3:
                    luma = 0.2126 * pass_arr[:, :, 0] + 0.7152 * pass_arr[:, :, 1] + 0.0722 * pass_arr[:, :, 2]
                    channels_dict["depth.Z"] = luma.astype(np.float32)
                else:
                    channels_dict["depth.Z"] = pass_arr[:, :, 0]
            elif layer_name == "mask":
                # Mask pass: store strictly as 1-channel mask.A
                channels_dict["mask.A"] = pass_arr[:, :, 0]
            elif layer_name == "normal":
                # Normal pass: store as normal.X, normal.Y, normal.Z
                sub_chans = ["X", "Y", "Z"] if p_C >= 3 else ["X"]
                for i, sc in enumerate(sub_chans):
                    channels_dict[f"normal.{sc}"] = pass_arr[:, :, i]
            else:
                # Custom passes (diffuse, specular, emission, etc.)
                if p_C == 1:
                    channels_dict[f"{layer_name}.A"] = pass_arr[:, :, 0]
                elif p_C >= 3:
                    sub_chans = ["R", "G", "B", "A"][:p_C]
                    for i, sc in enumerate(sub_chans):
                        channels_dict[f"{layer_name}.{sc}"] = pass_arr[:, :, i]

    try:
        _write_exr_openexr_multilayer(channels_dict, str(path), use_half, compression)
        return
    except ImportError:
        errors.append("OpenEXR python module not installed (pip install openexr)")
    except Exception as exc:
        print(f"[ACESSaveEXR Error] OpenEXR write failed: {exc}")
        errors.append(f"OpenEXR: {exc}")

    try:
        import cv2
        out = image
        if out.ndim == 3 and out.shape[-1] >= 3:
            order = [2, 1, 0] + list(range(3, out.shape[-1]))
            out = out[:, :, order]
        if not cv2.imwrite(str(path), out.astype(np.float32)):
            raise RuntimeError("cv2.imwrite returned false")
        return
    except Exception as exc:
        errors.append(f"opencv: {exc}")

    try:
        import tifffile
        tifffile.imwrite(str(path), image.astype(np.float32))
        return
    except Exception as exc:
        errors.append(f"tifffile: {exc}")

    raise RuntimeError("Could not write EXR. Backend errors: " + " | ".join(errors))


def _ocio_module():
    try:
        import PyOpenColorIO as ocio

        return ocio
    except Exception as exc:
        raise RuntimeError(
            "PyOpenColorIO is required for ACES Studio 1.3 OCIO transforms. "
            "Install the opencolorio pip package into ComfyUI embedded Python, "
            "or set engine to Built-in matrices. "
            f"Import error: {exc}"
        ) from exc


def _ocio_config(config_path: str):
    ocio = _ocio_module()
    path = (config_path or "").strip() or os.environ.get("OCIO") or DEFAULT_OCIO_CONFIG
    try:
        return ocio.Config.CreateFromFile(path)
    except Exception as exc:
        raise RuntimeError(f"Could not load OCIO config '{path}': {exc}") from exc


def _ocio_names(name: str):
    return OCIO_SPACE_ALIASES.get(name, [name])


def _ocio_processor(config, source: str, target: str):
    errors = []

    if target == "sRGB" or target in OCIO_SPACE_ALIASES.get("sRGB", []):
        try:
            import PyOpenColorIO as ocio
            displays = list(config.getDisplays())
            if "sRGB - Display" in displays:
                for src in _ocio_names(source):
                    try:
                        config.getColorSpace(src)
                        transform = ocio.DisplayViewTransform()
                        transform.setSrc(src)
                        transform.setDisplay("sRGB - Display")
                        views = list(config.getViews("sRGB - Display"))
                        if "ACES 1.0 - SDR Video" in views:
                            transform.setView("ACES 1.0 - SDR Video")
                        else:
                            transform.setView(views[0])
                        return config.getProcessor(transform)
                    except Exception as exc:
                        errors.append(f"DisplayViewTransform({src} -> sRGB): {exc}")
        except Exception as exc:
            errors.append(f"Display configuration failed (target): {exc}")

    # Special handling for sRGB as a SOURCE (Inverse Display View)
    if source == "sRGB" or source in OCIO_SPACE_ALIASES.get("sRGB", []):
        try:
            import PyOpenColorIO as ocio
            displays = list(config.getDisplays())
            if "sRGB - Display" in displays:
                for dst in _ocio_names(target):
                    try:
                        config.getColorSpace(dst)
                        transform = ocio.DisplayViewTransform()
                        transform.setSrc(dst) # Src is the Linear space we want to reach
                        transform.setDisplay("sRGB - Display")
                        views = list(config.getViews("sRGB - Display"))
                        if "ACES 1.0 - SDR Video" in views:
                            transform.setView("ACES 1.0 - SDR Video")
                        else:
                            transform.setView(views[0])
                        
                        processor = config.getProcessor(transform)
                        # Use the inverse of the DisplayViewTransform
                        return processor.getOptimizedProcessor(ocio.BIT_DEPTH_F32, ocio.BIT_DEPTH_F32, ocio.OPTIMIZATION_DEFAULT).getProcessor() \
                               if hasattr(processor, "getOptimizedProcessor") else config.getProcessor(transform, ocio.TRANSFORM_DIR_INVERSE)
                    except Exception as exc:
                        # Fallback for older OCIO versions or specific config layouts
                        try:
                            return config.getProcessor(transform, ocio.TRANSFORM_DIR_INVERSE)
                        except:
                            errors.append(f"DisplayViewTransform(sRGB -> {dst}): {exc}")
        except Exception as exc:
            errors.append(f"Display configuration failed (source): {exc}")

    for src in _ocio_names(source):
        for dst in _ocio_names(target):
            try:
                return config.getProcessor(src, dst)
            except Exception as exc:
                errors.append(f"{src} -> {dst}: {exc}")
    raise RuntimeError("Could not create OCIO processor. Tried: " + " | ".join(errors))


def _apply_ocio(arr: np.ndarray, config_path: str, source: str, target: str) -> np.ndarray:
    try:
        import PyOpenColorIO as ocio
    except ImportError:
        pass
        
    config = _ocio_config(config_path)
    cpu = _ocio_processor(config, source, target).getDefaultCPUProcessor()
    out = np.ascontiguousarray(arr.copy(), dtype=np.float32)
    B, H, W, C = out.shape
    for b in range(B):
        frame = np.ascontiguousarray(out[b], dtype=np.float32)
        desc = ocio.PackedImageDesc(frame, W, H, C)
        cpu.apply(desc)
        out[b] = frame
    return out


def _apply_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    shape = image.shape
    flat = image.reshape(-1, shape[-1])
    flat[:, :3] = flat[:, :3] @ matrix.T
    return flat.reshape(shape)


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb_encode(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def _colorspace_matrix(source: str, target: str):
    if source == target:
        return None
    linear_to_acescg = XYZ_D60_TO_ACESCG @ BRADFORD_D65_TO_D60 @ SRGB_TO_XYZ_D65
    acescg_to_linear = XYZ_D65_TO_SRGB @ BRADFORD_D60_TO_D65 @ ACESCG_TO_XYZ_D60
    matrices = {
        ("Linear sRGB", "ACEScg"): linear_to_acescg,
        ("ACEScg", "Linear sRGB"): acescg_to_linear,
        ("ACEScg", "ACES2065-1"): np.array(
            [
                [0.69545224, 0.14067870, 0.16386906],
                [0.04479456, 0.85967112, 0.09553432],
                [-0.00552588, 0.00402521, 1.00150067],
            ],
            dtype=np.float32,
        ),
    }
    matrices[("ACES2065-1", "ACEScg")] = np.linalg.inv(matrices[("ACEScg", "ACES2065-1")]).astype(np.float32)
    if (source, target) in matrices:
        return matrices[(source, target)]
    if source == "sRGB" and target == "ACEScg":
        return linear_to_acescg
    if source == "ACEScg" and target == "sRGB":
        return acescg_to_linear
    raise ValueError(f"Unsupported color transform: {source} to {target}")


class ACESLoadEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_file": (_list_input_exrs(),),
                "alpha_mode": (["composite checker", "ignore", "unpremultiply", "premultiply", "composite black", "composite gray", "composite white"],),
                "clamp_negative": (["false", "true"],),
                "frame_mode": (["all", "single", "range"], {"default": "all"}),
                "start_frame": ("INT", {"default": 1, "min": 0, "max": 99999}),
                "end_frame": ("INT", {"default": 100, "min": 0, "max": 99999}),
                "missing_frames": (["error", "black", "hold"], {"default": "error"}),
            },
            "optional": {
                "exr_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional path override"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "frame_count", "first_frame", "last_frame", "path")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, exr_file, alpha_mode, clamp_negative, frame_mode, start_frame, end_frame, missing_frames, exr_path=""):
        path = _resolve_input_path(exr_file, exr_path)
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")

        _assert_exr_read_allowed(path)

        seq_arr, count, f_start, f_end = _load_exr_sequence(str(path), frame_mode, start_frame, end_frame, missing_frames)
        
        if clamp_negative == "true":
            seq_arr = np.maximum(seq_arr, 0.0)
            
        tensor, mask = _image_to_tensor(seq_arr, alpha_mode)
        return (tensor, mask, count, f_start, f_end, str(path))


class ACESLoadEXRFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"default": "", "multiline": False}),
                "alpha_mode": (["composite checker", "ignore", "unpremultiply", "premultiply", "composite black", "composite gray", "composite white"],),
                "clamp_negative": (["false", "true"],),
                "frame_mode": (["all", "single", "range"], {"default": "all"}),
                "start_frame": ("INT", {"default": 1, "min": 0, "max": 99999}),
                "end_frame": ("INT", {"default": 100, "min": 0, "max": 99999}),
                "missing_frames": (["error", "black", "hold"], {"default": "error"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "mask", "frame_count", "first_frame", "last_frame", "path")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, exr_path, alpha_mode, clamp_negative, frame_mode, start_frame, end_frame, missing_frames):
        path = Path(exr_path.strip())
        if not path.is_absolute():
            path = Path(_input_dir()) / path
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")

        _assert_exr_read_allowed(path)

        seq_arr, count, f_start, f_end = _load_exr_sequence(str(path), frame_mode, start_frame, end_frame, missing_frames)
        
        if clamp_negative == "true":
            seq_arr = np.maximum(seq_arr, 0.0)
            
        tensor, mask = _image_to_tensor(seq_arr, alpha_mode)
        return (tensor, mask, count, f_start, f_end, str(path))


class ACESTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "engine": (["ACES Studio 1.3 OCIO", "Built-in matrices"], {"default": "ACES Studio 1.3 OCIO"}),
                "ocio_config": ("STRING", {"default": DEFAULT_OCIO_CONFIG, "multiline": False}),
                "source": (OCIO_SPACES,),
                "target": (OCIO_SPACES,),
                "reverse": (["false", "true"], {"default": "false"}),
                "clamp_output": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = CATEGORY

    def transform(self, image, engine, ocio_config, source, target, reverse, clamp_output):
        arr = _tensor_to_numpy(image).copy()
        
        if reverse == "true":
            source, target = target, source
            
        if engine == "ACES Studio 1.3 OCIO":
            arr = _apply_ocio(arr, ocio_config, source, target)
        else:
            if source not in ("sRGB", "Linear sRGB", "ACEScg", "ACES2065-1"):
                raise ValueError("Built-in matrices only support sRGB, Linear sRGB, ACEScg, and ACES2065-1.")
            if target not in ("sRGB", "Linear sRGB", "ACEScg", "ACES2065-1"):
                raise ValueError("Built-in matrices only support sRGB, Linear sRGB, ACEScg, and ACES2065-1.")
            if source == "sRGB":
                arr[..., :3] = _srgb_decode(arr[..., :3])
                source = "Linear sRGB"
            encode_srgb = target == "sRGB"
            if encode_srgb:
                target = "Linear sRGB"
            matrix = _colorspace_matrix(source, target)
            if matrix is not None:
                arr = _apply_matrix(arr, matrix)
            if encode_srgb:
                arr[..., :3] = _srgb_encode(arr[..., :3])
        if clamp_output == "true":
            arr = np.clip(arr, 0.0, 1.0)
        return (torch.from_numpy(arr),)


class ACESToneMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "look": (["ACES fitted", "Reinhard", "AgX", "Filmic (Blender)", "DaVinci"],),
                "output": (["sRGB", "Linear sRGB"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tonemap"
    CATEGORY = CATEGORY

    def tonemap(self, image, exposure, look, output):
        arr = _tensor_to_numpy(image).copy()
        rgb = np.maximum(arr[..., :3] * (2.0 ** exposure), 0.0)
        
        if look == "ACES fitted":
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
        elif look == "Reinhard":
            rgb = rgb / (1.0 + rgb)
        elif look == "AgX":
            min_ev, max_ev = -10.0, +6.5
            log_rgb = np.clip((np.log2(np.maximum(rgb, 1e-6)) - min_ev) / (max_ev - min_ev), 0.0, 1.0)
            rgb = log_rgb * log_rgb * (3.0 - 2.0 * log_rgb)
        elif look == "Filmic (Blender)":
            x = np.maximum(0.0, rgb - 0.004)
            rgb = (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06)
        elif look == "DaVinci":
            rgb = np.where(rgb > 0.008, 0.5 * np.log2(rgb + 0.008) + 0.5, rgb * 60.0)
            rgb = np.clip(rgb, 0.0, 1.0)

        rgb = np.clip(rgb, 0.0, 1.0)
        if output == "sRGB":
            rgb = _srgb_encode(rgb)
        arr[..., :3] = rgb
        return (torch.from_numpy(arr),)


class ACESSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING", {"default": _output_dir(), "multiline": False}),
                "filename": ("STRING", {"default": "render_%04d", "multiline": False}),
                "bit_depth": (BIT_DEPTHS, {"default": BIT_DEPTHS[0]}),
                "compression": (COMPRESSION_KEYS, {"default": COMPRESSION_KEYS[0]}),
            },
            "optional": {
                "depth": ("IMAGE,MASK",),
                "normal": ("IMAGE",),
                "mask": ("MASK",),
                "diffuse": ("IMAGE",),
                "specular": ("IMAGE",),
                "emission": ("IMAGE",),
                "start_frame": ("INT", {"default": 1, "min": 0, "max": 99999}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, image, output_dir, filename, bit_depth, compression,
             depth=None, normal=None, mask=None, diffuse=None, specular=None, emission=None, start_frame=1):
        import re
        arr = _tensor_to_numpy(image).copy()

        resolved_output_dir = Path(os.path.realpath(output_dir.strip()))
        _assert_path_allowed(resolved_output_dir, [_output_dir()], action="write to")
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        B = arr.shape[0]
        last_path = ""

        def extract_pass(tensor, batch_idx):
            if tensor is None:
                return None
            np_p = tensor.detach().cpu().numpy()
            if np_p.ndim == 2:
                return np_p
            if np_p.ndim == 3:
                if np_p.shape[0] == B:
                    return np_p[batch_idx]
                return np_p
            if np_p.ndim == 4:
                b_i = min(batch_idx, np_p.shape[0] - 1)
                return np_p[b_i]
            return np_p

        for b in range(B):
            frame_num = start_frame + b
            base_name = re.sub(r"%0(\d+)d", lambda m: f"{frame_num:0{m.group(1)}d}", filename)
            if not base_name.lower().endswith(".exr"):
                base_name += ".exr"
            base_name = os.path.basename(base_name)
            path = resolved_output_dir / base_name
            
            if B > 1 and base_name == os.path.basename(filename) + ".exr":
                path = path.with_name(f"{path.stem}_{frame_num:04d}.exr")

            extra_passes = {
                "depth": extract_pass(depth, b),
                "normal": extract_pass(normal, b),
                "mask": extract_pass(mask, b),
                "diffuse": extract_pass(diffuse, b),
                "specular": extract_pass(specular, b),
                "emission": extract_pass(emission, b),
            }
            extra_passes = {k: v for k, v in extra_passes.items() if v is not None}

            _write_exr(path, arr[b], bit_depth, compression, extra_passes=extra_passes)
            last_path = str(path)
            
        return {"ui": {"text": [last_path]}, "result": (last_path,)}


class EXRLayerLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"default": "", "multiline": False}),
                "layer_name": ("STRING", {"default": "beauty", "multiline": False, "placeholder": "e.g. beauty, depth, normal, mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "available_layers")
    FUNCTION = "load_layer"
    CATEGORY = CATEGORY

    def load_layer(self, exr_path, layer_name="beauty"):
        path = Path(exr_path.strip())
        if not path.is_absolute():
            path = Path(_input_dir()) / path
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")

        _assert_exr_read_allowed(path)

        layers, available = _read_exr_all_layers(path)
        
        target = layer_name.strip()
        if target not in layers:
            if "beauty" in layers:
                target = "beauty"
            elif layers:
                target = list(layers.keys())[0]
            else:
                raise ValueError(f"No valid layers found in EXR: {path}")

        img_arr = layers[target]
        tensor, mask = _image_to_tensor(img_arr, "ignore")
        avail_str = ", ".join(available)
        return (tensor, mask, avail_str)


class EXRMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("metadata_json", "width", "height", "channels", "compression")
    FUNCTION = "read_metadata"
    CATEGORY = CATEGORY

    def read_metadata(self, exr_path):
        import json
        path = Path(exr_path.strip())
        if not path.is_absolute():
            path = Path(_input_dir()) / path
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")

        _assert_exr_read_allowed(path)

        meta, W, H, channels, comp = _read_exr_header_info(path)
        meta_str = json.dumps(meta, indent=2)
        chan_str = ", ".join(channels)
        return (meta_str, W, H, chan_str, comp)


class NodexSyntheticHDRExpansion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "boost_multiplier": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "expand"
    CATEGORY      = "image/ACES + EXR"

    def expand(self, image: torch.Tensor, boost_multiplier: float, threshold: float) -> tuple:
        arr = image.detach().cpu().numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=4.0, neginf=0.0)
        rgb = np.clip(arr[..., :3], 0.0, 1.0)
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
        lum = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2])[..., None]
        boost = np.where(lum > threshold, 1.0 + (lum - threshold) * boost_multiplier, 1.0).astype(np.float32)
        hdr_linear = (linear * boost).astype(np.float32)
        out = arr.copy()
        out[..., :3] = hdr_linear
        return (torch.from_numpy(out),)

class NodexHDRExposureAdjust:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ev_adjust": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "adjust"
    CATEGORY = "image/ACES + EXR"

    def adjust(self, image: torch.Tensor, ev_adjust: float):
        multiplier = 2.0 ** ev_adjust
        out = image.clone()
        out[..., :3] = out[..., :3] * multiplier
        return (out,)

class NodexAutoBracketGenerator:
    """
    Automatically generates a batch of exposure brackets from a single SDR image by mathematically multiplying the RGB values.
    Note: This does not hallucinate lost highlight details, it simply fakes the exposure steps.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ev_values": ("STRING", {"default": "-2.0, 0.0, 2.0", "multiline": False, "tooltip": "Comma-separated EV values"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("bracket_batch",)
    FUNCTION = "generate"
    CATEGORY = "image/ACES + EXR"

    def generate(self, image: torch.Tensor, ev_values: str):
        evs = [float(e.strip()) for e in ev_values.split(",") if e.strip()]
        out_images = []
        for ev in evs:
            multiplier = 2.0 ** ev
            img_bracket = torch.clamp(image * multiplier, 0.0, 1.0)
            out_images.append(img_bracket)
        
        return (torch.cat(out_images, dim=0),)

class NodexExposureBracketMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "ev_values": ("STRING", {"default": "-2.0, 0.0, 2.0", "multiline": False, "tooltip": "Comma-separated EV values, e.g. -2.0, 0.0, 2.0"}),
                "adaptive_calibration": ("BOOLEAN", {"default": True, "tooltip": "Analyze actual brightness relationships instead of assuming perfect camera physics"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_image",)
    FUNCTION = "merge"
    CATEGORY = "image/ACES + EXR"

    def merge(self, image1: torch.Tensor, ev_values: str, adaptive_calibration: bool = True, image2=None, image3=None, image4=None, image5=None):
        import cv2
        import numpy as np

        images_list = [image1]
        if image2 is not None: images_list.append(image2)
        if image3 is not None: images_list.append(image3)
        if image4 is not None: images_list.append(image4)
        if image5 is not None: images_list.append(image5)
        
        try:
            images = torch.cat(images_list, dim=0)
        except Exception as e:
            raise ValueError(f"Failed to batch images. Ensure all connected images have the same dimensions. Error: {e}")

        arr = images.detach().cpu().numpy()
        B = arr.shape[0]
        evs = [float(e.strip()) for e in ev_values.split(",") if e.strip()]
        if len(evs) != B:
            raise ValueError(f"Number of EV values ({len(evs)}) must match number of images in batch ({B}).")

        times = np.array([2.0 ** ev for ev in evs], dtype=np.float32)
        arr_clipped = np.clip(arr, 0.0, 1.0)
        
        if adaptive_calibration and B > 1:
            try:
                ref_idx = min(range(len(evs)), key=lambda i: abs(evs[i]))
                ref_img = arr_clipped[ref_idx, ..., :3]
                ref_luma = 0.2126 * ref_img[..., 0] + 0.7152 * ref_img[..., 1] + 0.0722 * ref_img[..., 2]
                
                mask = (ref_luma > 0.2) & (ref_luma < 0.8)
                if np.sum(mask) > 1000:
                    new_times = []
                    for i in range(B):
                        if i == ref_idx:
                            new_times.append(times[i])
                            continue
                        tgt_img = arr_clipped[i, ..., :3]
                        tgt_luma = 0.2126 * tgt_img[..., 0] + 0.7152 * tgt_img[..., 1] + 0.0722 * tgt_img[..., 2]
                        ratios = tgt_luma[mask] / (ref_luma[mask] + 1e-6)
                        median_ratio = np.median(ratios)
                        time_ratio = median_ratio ** 1.8
                        calibrated_time = times[ref_idx] * time_ratio
                        new_times.append(calibrated_time)
                    times = np.array(new_times, dtype=np.float32)
                    print(f"[Nodex HDR] Adaptive Calibration Times: {times}")
            except Exception as e:
                print(f"[Nodex HDR] Adaptive Calibration failed, using nominal times: {e}")

        arr_uint8 = (arr_clipped * 255.0).astype(np.uint8)
        img_list = [arr_uint8[i, ..., :3] for i in range(B)]

        merge_debevec = cv2.createMergeDebevec()
        hdr = merge_debevec.process(img_list, times=times)

        out = np.ones((1, hdr.shape[0], hdr.shape[1], arr.shape[3] if arr.shape[3] == 4 else 3), dtype=np.float32)
        out[0, ..., :3] = hdr
        if arr.shape[3] == 4:
            out[0, ..., 3] = arr[0, ..., 3] 
        return (torch.from_numpy(out),)

class VispyEXRViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",),},
            "optional": {
                "image_b": ("IMAGE",),
                "exr_path": ("STRING", {"default": "", "tooltip": "Override: load EXR directly from disk"}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("image", "exr_path")
    FUNCTION      = "view"
    CATEGORY      = "image/ACES + EXR"
    OUTPUT_NODE   = True

    def view(self, image: torch.Tensor, image_b=None, exr_path: str = "") -> tuple:
        temp_dir = folder_paths.get_temp_directory()

        def save_sequence(tensor: torch.Tensor):
            """Save every frame of a [N, H, W, C] batch tensor to separate .bin files.

            Returns a list of dicts: [{filename, width, height}, ...]
            Each .bin file is a raw float32 RGBA buffer (H*W*4 floats).
            """
            frames = []
            for i in range(tensor.shape[0]):
                filename = f"nodex_hdr_{uuid.uuid4().hex[:8]}.bin"
                filepath = os.path.join(temp_dir, filename)
                arr = tensor[i].cpu().numpy().astype(np.float32)
                h, w = arr.shape[:2]
                c = arr.shape[2] if arr.ndim == 3 else 1
                if c == 1:
                    arr = np.stack([arr] * 3 + [np.ones_like(arr)], axis=-1)
                elif c == 3:
                    arr = np.concatenate(
                        [arr, np.ones((*arr.shape[:2], 1), np.float32)], axis=-1
                    )
                with open(filepath, "wb") as f:
                    f.write(arr.astype(np.float32).tobytes())
                frames.append({"filename": filename, "width": int(w), "height": int(h)})
            return frames

        if exr_path.strip() and os.path.exists(exr_path.strip()):
            from .exr_utils import load_exr

            # Security: confine exr_path reads to ComfyUI allowed dirs
            _assert_path_allowed(
                Path(exr_path.strip()),
                _get_all_allowed_dirs(),
                action="read"
            )

            filename = f"nodex_hdr_{uuid.uuid4().hex[:8]}.bin"
            filepath = os.path.join(temp_dir, filename)
            arr = load_exr(exr_path.strip())
            h, w = arr.shape[:2]
            c = arr.shape[2] if arr.ndim == 3 else 1
            if c == 1:
                arr = np.stack([arr] * 3 + [np.ones_like(arr)], axis=-1)
            elif c == 3:
                arr = np.concatenate(
                    [arr, np.ones((*arr.shape[:2], 1), np.float32)], axis=-1
                )
            with open(filepath, "wb") as f:
                f.write(arr.astype(np.float32).tobytes())
            sequence_a = [{"filename": filename, "width": int(w), "height": int(h)}]
            return {"ui": {"sequence_a": sequence_a}, "result": (image, exr_path)}
        else:
            sequence_a = save_sequence(image)
            ui_data = {"sequence_a": sequence_a}
            if image_b is not None:
                ui_data["sequence_b"] = save_sequence(image_b)
            return {"ui": ui_data, "result": (image, exr_path)}

NODE_CLASS_MAPPINGS = {
    "ACESLoadEXR": ACESLoadEXR,
    "ACESLoadEXRFromPath": ACESLoadEXRFromPath,
    "ACESTransform": ACESTransform,
    "ACESToneMap": ACESToneMap,
    "ACESSaveEXR": ACESSaveEXR,
    "EXRLayerLoad": EXRLayerLoad,
    "EXRMetadata": EXRMetadata,
    "VispyEXRViewer": VispyEXRViewer,
    "NodexSyntheticHDRExpansion": NodexSyntheticHDRExpansion,
    "NodexExposureBracketMerge": NodexExposureBracketMerge,
    "NodexHDRExposureAdjust": NodexHDRExposureAdjust,
    "NodexAutoBracketGenerator": NodexAutoBracketGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACESLoadEXR": "Load EXR",
    "ACESLoadEXRFromPath": "Load EXR From Path",
    "ACESTransform": "ACES Color Transform",
    "ACESToneMap": "ACES Tone Map",
    "ACESSaveEXR": "Save EXR (Multi-Layer)",
    "EXRLayerLoad": "Load EXR Layer",
    "EXRMetadata": "EXR Metadata Reader",
    "VispyEXRViewer": "Nodex HDR Viewer 🎨",
    "NodexSyntheticHDRExpansion": "Synthetic HDR Expansion 🚀",
    "NodexExposureBracketMerge": "Exposure Bracket Merge 📸",
    "NodexHDRExposureAdjust": "HDR Exposure Adjust 💡",
    "NodexAutoBracketGenerator": "Auto Bracket Generator 🎞️",
}
