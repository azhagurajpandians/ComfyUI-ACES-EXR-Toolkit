# Changelog

## 0.2.0

- Added sequence playback transport bar to Nodex HDR WebGL Viewer (Play/Pause, scrub, step, FPS selector).
- Added multi-layer EXR saving and layer-by-layer extraction nodes (`Save EXR (Multi-Layer)`, `Load EXR Layer`).
- Added EXR metadata inspector node (`EXR Metadata Reader`).
- Added tone mapping curves: AgX, Filmic, DaVinci, ACES fitted, and Reinhard.
- Added Synthetic Highlight Expansion and Exposure Bracket Merging nodes.
- Added pixel RGBA HUD on hover, luminance histogram, and bilinear filter toggle in HDR viewer.
- Added Comfy Registry support (`[tool.comfy]`, `.comfyignore`, GitHub Actions workflow).
- Fixed canvas sizing and layout scaling issues with LiteGraph node resizing and graph zoom.

## 0.1.0

- Initial public release.
- Add EXR loading from ComfyUI input or absolute path.
- Add alpha-aware preview compositing modes.
- Add ACES Studio 1.3 OCIO color transform node.
- Add built-in matrix fallback for sRGB, Linear sRGB, ACEScg, and ACES2065-1.
- Add HDR tone mapping node.
- Add float EXR save node.
