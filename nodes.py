import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import folder_paths
except Exception:
    folder_paths = None


CATEGORY = "image/ACES + EXR"
DEFAULT_OCIO_CONFIG = "ocio://studio-config-v1.0.0_aces-v1.3_ocio-v2.1"

OCIO_SPACE_ALIASES = {
    "sRGB": ["sRGB", "Utility - sRGB - Texture", "Output - sRGB"],
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
        import imageio.v2 as iio
        import imageio
        try:
            imageio.plugins.freeimage.download()
        except Exception:
            pass
        return iio.imread(path, format="EXR-FI")
    except Exception as exc:
        errors.append(f"imageio(freeimage): {exc}")

    try:
        import imageio.v3 as iio
        img = iio.imread(path)
        if img.dtype == np.uint8:
            print("Warning: ComfyUI-ACES-EXR read EXR as 8-bit, values normalized and precision lost.")
            img = img.astype(np.float32) / 255.0
        return img
    except Exception as exc:
        errors.append(f"imageio(v3): {exc}")

    raise RuntimeError(
        "Could not read EXR. Install an EXR-capable backend such as OpenImageIO, "
        "or enable OpenEXR support for imageio/opencv. Backend errors: " + " | ".join(errors)
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


def _write_exr_openexr(img: np.ndarray, path: str, use_half: bool, compression_key: str):
    import OpenEXR
    import Imath
    
    comp_name = COMPRESSIONS.get(compression_key, "ZIP_COMPRESSION")
    imath_comp = getattr(Imath.Compression, comp_name)
    ptype_write = Imath.PixelType(Imath.PixelType.HALF if use_half else Imath.PixelType.FLOAT)
    dtype_write = np.float16 if use_half else np.float32

    H, W, C = img.shape
    channel_names = ["R", "G", "B", "A"] if C == 4 else ["R", "G", "B"]

    header = OpenEXR.Header(W, H)
    header["compression"] = Imath.Compression(imath_comp)
    header["channels"] = {ch: Imath.Channel(ptype_write) for ch in channel_names}

    out = OpenEXR.OutputFile(path, header)
    pixels = {}
    for i, ch in enumerate(channel_names):
        pixels[ch] = img[:, :, i].astype(dtype_write).tobytes()
    out.writePixels(pixels)
    out.close()


def _write_exr(path: Path, image: np.ndarray, bit_depth: str = "32f (float)", compression: str = "None (uncompressed)"):
    errors = []
    path.parent.mkdir(parents=True, exist_ok=True)
    
    use_half = "16f" in bit_depth
    
    try:
        _write_exr_openexr(image, str(path), use_half, compression)
        return
    except ImportError:
        errors.append("OpenEXR python module not installed (pip install openexr)")
    except Exception as exc:
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
        import imageio.v3 as iio

        iio.imwrite(path, image.astype(np.float32))
        return
    except Exception as exc:
        errors.append(f"imageio: {exc}")

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
                        errors.append(f"DisplayViewTransform({src}): {exc}")
        except Exception as exc:
            errors.append(f"Display configuration failed: {exc}")

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
                "exr_file": (_list_input_exrs(), {"image_upload": True}),
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

    def load(self, exr_file, alpha_mode, clamp_negative, frame_mode, start_frame, end_frame, missing_frames):
        path = _resolve_input_path(exr_file, "")
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")
            
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
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")
            
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
                "clamp_output": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform"
    CATEGORY = CATEGORY

    def transform(self, image, engine, ocio_config, source, target, clamp_output):
        arr = _tensor_to_numpy(image).copy()
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
                "look": (["ACES fitted", "Reinhard"],),
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
        else:
            rgb = rgb / (1.0 + rgb)
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
                "start_frame": ("INT", {"default": 1, "min": 0, "max": 99999}),
                "ocio_config": ("STRING", {"default": "", "multiline": False}),
                "input_transform": (OCIO_SPACES, {"default": "ACEScg"}),
                "colorspace": (OCIO_SPACES, {"default": "ACES2065-1"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, image, output_dir, filename, bit_depth, compression,
             start_frame=1, ocio_config="", input_transform="ACEScg", colorspace="ACES2065-1"):
        import re
        arr = _tensor_to_numpy(image).copy()
        
        if ocio_config and ocio_config.strip():
            arr = _apply_ocio(arr, ocio_config, input_transform, colorspace)
            
        B = arr.shape[0]
        last_path = ""
        for b in range(B):
            frame_num = start_frame + b
            base_name = re.sub(r"%0(\d+)d", lambda m: f"{frame_num:0{m.group(1)}d}", filename)
            if not base_name.lower().endswith(".exr"):
                base_name += ".exr"
            path = Path(output_dir.strip()) / base_name
            
            # Simple indexing fallback if no format string was used and multiple frames
            if B > 1 and base_name == filename + ".exr":
                path = path.with_name(f"{path.stem}_{frame_num:04d}.exr")
            
            _write_exr(path, arr[b], bit_depth, compression)
            last_path = str(path)
            
        return {"ui": {"text": [last_path]}, "result": (last_path,)}


NODE_CLASS_MAPPINGS = {
    "ACESLoadEXR": ACESLoadEXR,
    "ACESLoadEXRFromPath": ACESLoadEXRFromPath,
    "ACESTransform": ACESTransform,
    "ACESToneMap": ACESToneMap,
    "ACESSaveEXR": ACESSaveEXR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ACESLoadEXR": "Load EXR",
    "ACESLoadEXRFromPath": "Load EXR From Path",
    "ACESTransform": "ACES Color Transform",
    "ACESToneMap": "ACES Tone Map",
    "ACESSaveEXR": "Save EXR",
}
