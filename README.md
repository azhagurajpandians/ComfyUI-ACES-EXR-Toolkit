# ComfyUI ACES EXR Toolkit

Professional ACES and EXR workflow nodes for ComfyUI.

This extension adds practical OpenEXR loading/saving, ACES Studio 1.3 OCIO color transforms, alpha-aware previews, and simple HDR tone mapping for image pipelines that need float EXR support.

## Features

- **Fast OCIO Processing**: Native bulk-pixel processing via PyOpenColorIO for lightning-fast ACES transformations.
- **Load EXR Sequences**: Auto-detects and loads numbered EXR sequences (`render.0001.exr`) directly into ComfyUI batch tensors.
- **16-bit & 32-bit Export**: Save half-float (16f) or float (32f) EXRs with standard compression codecs (ZIP, PIZ, DWAA) via `OpenEXR`.
- **Integrated UI Browser**: Visual file browser modal to easily pick absolute paths and output directories without copy-pasting.
- **Alpha Previews**: Preview transparent EXRs with checker, black, gray, or white alpha compositing.
- **Built-in Fallbacks**: Basic ACEScg / sRGB transforms work even if OCIO isn't installed.
- **Tone Mapping**: Apply simple display tone mapping for HDR previews.

## Nodes

| Node | Purpose |
| --- | --- |
| `Load EXR` | Load `.exr` files from ComfyUI `input`, with optional path override |
| `Load EXR From Path` | Load an EXR from any absolute path |
| `ACES Color Transform` | Transform between ACES, camera, sRGB, and utility color spaces |
| `ACES Tone Map` | Apply simple display tone mapping for HDR previews |
| `Save EXR` | Save ComfyUI image data as float EXR |

Nodes appear under:

```text
image/ACES + EXR
```

## Installation

Clone this repository into your ComfyUI `custom_nodes` folder:

```powershell
cd D:\Ai\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/azhagurajpandians/ComfyUI-ACES-EXR-Toolkit.git
```

Install the optional OpenColorIO dependency into the Python environment used by ComfyUI:

```powershell
D:\Ai\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r D:\Ai\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-ACES-EXR-Toolkit\requirements.txt
```

Restart ComfyUI after installation.

## Dependencies

Required runtime packages are normally already included with ComfyUI:

- `numpy`
- `torch`
- `opencv-python` or `opencv-python-headless`

Recommended for ACES Studio 1.3:

- `opencolorio` (The pip package is named `opencolorio`; the Python module it provides is `PyOpenColorIO`.)

Recommended for 16-bit Float and Compression export:

- `openexr`

## ACES Studio 1.3

The default OCIO config is:

```text
ocio://studio-config-v1.0.0_aces-v1.3_ocio-v2.1
```

That is the OpenColorIO built-in URI for:

```text
studio-config-v1.0.0_aces-v1.3_ocio-v2.1.ocio
```

Use `ACES Color Transform` with:

- `engine`: `ACES Studio 1.3 OCIO`
- `ocio_config`: `ocio://studio-config-v1.0.0_aces-v1.3_ocio-v2.1`
- `source`: the source color space, for example `ACEScg`
- `target`: the target color space, for example `sRGB` or `Utility - Linear - sRGB`

You can also point `ocio_config` to a local `.ocio` file, or set the `OCIO` environment variable.

## Loading EXR Files

For most EXR work, use `Load EXR From Path` and paste the absolute file path:

```text
D:\shots\render\beauty.exr
```

For ComfyUI input-folder workflows, copy EXRs into:

```text
ComfyUI\input
```

Then restart or refresh ComfyUI and use `Load EXR`.

## Alpha Preview

ComfyUI `Preview Image` does not display the separate mask output as transparency. For RGBA EXRs, choose an alpha mode that composites RGB for preview:

- `composite checker`
- `composite black`
- `composite gray`
- `composite white`

The alpha is still exposed separately as the `MASK` output.

## Common Workflows

### Preview an ACEScg EXR

```text
Load EXR From Path -> ACES Color Transform -> Preview Image
```

Recommended transform:

- `source`: `ACEScg`
- `target`: `sRGB`
- `clamp_output`: `true`

If the image is still too bright, insert `ACES Tone Map` before `Preview Image`.

### Convert Linear sRGB EXR to Display sRGB

```text
Load EXR From Path -> ACES Color Transform -> Preview Image
```

Recommended transform:

- `source`: `Linear sRGB`
- `target`: `sRGB`
- `clamp_output`: `true`

### Save EXR (Round-tripping back to ACES)

If you loaded an `ACES2065-1` image and converted it to `sRGB` to process/edit it in ComfyUI, you must convert it **back** to `ACES2065-1` before saving. Otherwise, the saved EXR will contain raw sRGB values, which will look extremely distorted (e.g., cyan/yellow) if you later load it and apply an ACES transform again.

You can do this directly inside the `Save EXR` node without needing an extra node:

```text
... sRGB image pipeline ... -> Save EXR
```

In the `Save EXR` node, configure the built-in transform:
- `ocio_config`: `ocio://studio-config-v1.0.0_aces-v1.3_ocio-v2.1`
- `input_transform`: `sRGB` (or `Utility - sRGB - Texture`)
- `colorspace`: `ACES - ACES2065-1`

This will automatically reverse the color space before writing the final `.exr` file.

The output is written to your selected output directory or ComfyUI's `output` folder as a 16-bit or 32-bit float EXR.

## Troubleshooting

### EXR does not show in file picker

Use `Load EXR From Path`. ComfyUI's upload widget is image-centric and can be unreliable for `.exr`.

### Preview is white

Your EXR likely has transparency and white RGB in transparent pixels. Set `alpha_mode` to `composite checker` or `composite black`.

### OCIO mode fails

Install OpenColorIO:

```powershell
python -m pip install opencolorio
```

For portable ComfyUI, use its embedded Python:

```powershell
D:\Ai\ComfyUI_windows_portable\python_embeded\python.exe -m pip install opencolorio
```

### Built-in matrix mode limitations

`Built-in matrices` supports only:

- `sRGB`
- `Linear sRGB`
- `ACEScg`
- `ACES2065-1`

Use `ACES Studio 1.3 OCIO` for camera spaces and full ACES config transforms.

## License

MIT License. See [LICENSE](LICENSE).
