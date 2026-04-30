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
    while image.ndim > 3:
        squeeze_axes = [axis for axis in range(image.ndim - 1) if image.shape[axis] == 1]
        if squeeze_axes:
            image = np.squeeze(image, axis=tuple(squeeze_axes))
        else:
            image = image[0]
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3:
        raise ValueError(f"Expected EXR image data with 2 or 3 dimensions after normalization, got shape {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    alpha = None
    if image.shape[-1] >= 4:
        alpha = np.clip(image[:, :, 3].copy(), 0.0, 1.0)
        rgb = image[:, :, :3].copy()
        if alpha_mode == "unpremultiply":
            denom = np.maximum(alpha[:, :, None], 1.0e-6)
            rgb = np.where(alpha[:, :, None] > 1.0e-6, rgb / denom, rgb)
        elif alpha_mode == "premultiply":
            rgb = rgb * alpha[:, :, None]
        elif alpha_mode.startswith("composite"):
            if alpha_mode == "composite black":
                background = np.zeros_like(rgb)
            elif alpha_mode == "composite white":
                background = np.ones_like(rgb)
            elif alpha_mode == "composite gray":
                background = np.full_like(rgb, 0.5)
            else:
                yy, xx = np.indices(alpha.shape)
                checker = (((xx // 64) + (yy // 64)) % 2).astype(np.float32)
                checker = checker * 0.35 + 0.35
                background = np.repeat(checker[:, :, None], 3, axis=2)
            rgb = rgb * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])
    else:
        rgb = image[:, :, :3].copy()

    mask = np.zeros(rgb.shape[:2], dtype=np.float32) if alpha is None else 1.0 - np.clip(alpha, 0.0, 1.0)
    return torch.from_numpy(rgb)[None,], torch.from_numpy(mask)[None,]


def _tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
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
        import imageio.v3 as iio

        return iio.imread(path)
    except Exception as exc:
        errors.append(f"imageio: {exc}")

    raise RuntimeError(
        "Could not read EXR. Install an EXR-capable backend such as OpenImageIO, "
        "or enable OpenEXR support for imageio/opencv. Backend errors: " + " | ".join(errors)
    )


def _write_exr(path: Path, image: np.ndarray):
    errors = []
    path.parent.mkdir(parents=True, exist_ok=True)
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
    for src in _ocio_names(source):
        for dst in _ocio_names(target):
            try:
                return config.getProcessor(src, dst)
            except Exception as exc:
                errors.append(f"{src} -> {dst}: {exc}")
    raise RuntimeError("Could not create OCIO processor. Tried: " + " | ".join(errors))


def _apply_ocio(arr: np.ndarray, config_path: str, source: str, target: str) -> np.ndarray:
    config = _ocio_config(config_path)
    cpu = _ocio_processor(config, source, target).getDefaultCPUProcessor()
    out = arr.copy()
    flat = out.reshape(-1, out.shape[-1])
    for pixel in flat[:, :3]:
        rgb = [float(pixel[0]), float(pixel[1]), float(pixel[2])]
        result = cpu.applyRGB(rgb)
        if result is None:
            pixel[:] = rgb
        else:
            pixel[:] = result
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
                "exr_path": ("STRING", {"default": "", "multiline": False}),
                "alpha_mode": (["composite checker", "ignore", "unpremultiply", "premultiply", "composite black", "composite gray", "composite white"],),
                "clamp_negative": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, exr_file, exr_path, alpha_mode, clamp_negative):
        path = _resolve_input_path(exr_file, exr_path)
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")
        image = _read_exr(path)
        if clamp_negative == "true":
            image = np.maximum(image, 0.0)
        tensor, mask = _image_to_tensor(image, alpha_mode)
        return (tensor, mask, str(path))


class ACESLoadEXRFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exr_path": ("STRING", {"default": "", "multiline": False}),
                "alpha_mode": (["composite checker", "ignore", "unpremultiply", "premultiply", "composite black", "composite gray", "composite white"],),
                "clamp_negative": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "path")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, exr_path, alpha_mode, clamp_negative):
        path = _resolve_input_path("", exr_path)
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")
        image = _read_exr(path)
        if clamp_negative == "true":
            image = np.maximum(image, 0.0)
        tensor, mask = _image_to_tensor(image, alpha_mode)
        return (tensor, mask, str(path))


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
                arr[:, :, :3] = _srgb_decode(arr[:, :, :3])
                source = "Linear sRGB"
            encode_srgb = target == "sRGB"
            if encode_srgb:
                target = "Linear sRGB"
            matrix = _colorspace_matrix(source, target)
            if matrix is not None:
                arr = _apply_matrix(arr, matrix)
            if encode_srgb:
                arr[:, :, :3] = _srgb_encode(arr[:, :, :3])
        if clamp_output == "true":
            arr = np.clip(arr, 0.0, 1.0)
        return (torch.from_numpy(arr)[None,],)


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
        rgb = np.maximum(arr[:, :, :3] * (2.0 ** exposure), 0.0)
        if look == "ACES fitted":
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
        else:
            rgb = rgb / (1.0 + rgb)
        rgb = np.clip(rgb, 0.0, 1.0)
        if output == "sRGB":
            rgb = _srgb_encode(rgb)
        arr[:, :, :3] = rgb
        return (torch.from_numpy(arr)[None,],)


class ACESSaveEXR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "aces_exr/image"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, image, filename_prefix):
        base = Path(_output_dir()) / filename_prefix
        path = base.with_suffix(".exr")
        index = 1
        while path.exists():
            path = base.with_name(f"{base.name}_{index:05d}").with_suffix(".exr")
            index += 1
        _write_exr(path, _tensor_to_numpy(image))
        return (str(path),)


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
