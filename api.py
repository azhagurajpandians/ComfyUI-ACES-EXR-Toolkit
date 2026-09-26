import os
from aiohttp import web
from server import PromptServer
import folder_paths

@PromptServer.instance.routes.get("/nodex_hdr/view")
async def view_hdr(request):
    filename = request.rel_url.query.get("filename")
    if not filename:
        return web.Response(status=400, text="No filename provided")
    
    # Secure filename by taking only the basename to prevent directory traversal
    filename = os.path.basename(filename)
    
    temp_dir = folder_paths.get_temp_directory()
    filepath = os.path.join(temp_dir, filename)
    
    if not os.path.exists(filepath):
        return web.Response(status=404, text="File not found")
        
    return web.FileResponse(filepath)


@PromptServer.instance.routes.post("/aces/sequence_info")
@PromptServer.instance.routes.get("/aces/sequence_info")
async def sequence_info(request):
    from pathlib import Path
    try:
        if request.method == "POST":
            data = await request.json()
            path_str = data.get("path", "")
            filename = data.get("filename", "")
        else:
            path_str = request.rel_url.query.get("path", "")
            filename = request.rel_url.query.get("filename", "")

        path_val = (path_str or "").strip().strip('"').strip("'")
        file_val = (filename or "").strip()

        target_path = None
        if path_val:
            p = Path(path_val)
            if not p.is_absolute() and folder_paths:
                p = Path(folder_paths.get_input_directory()) / p
            target_path = p
        elif file_val and not file_val.startswith("<"):
            if folder_paths:
                target_path = Path(folder_paths.get_input_directory()) / file_val
            else:
                target_path = Path(file_val)

        if not target_path or not target_path.exists():
            return web.json_response({"success": False, "error": f"File or path not found: {target_path}"})

        from .nodes import _detect_exr_sequence, _read_exr_header_info
        paths, seq_first, seq_last = _detect_exr_sequence(str(target_path))
        if not paths:
            return web.json_response({"success": False, "error": f"No EXR sequence found at: {target_path}"})

        # Read dimensions from the first frame
        meta, width, height, chans, comp = _read_exr_header_info(Path(paths[0]))
        return web.json_response({
            "success": True,
            "start_frame": int(seq_first),
            "end_frame": int(seq_last),
            "frame_count": len(paths),
            "width": int(width),
            "height": int(height),
            "first_file": str(paths[0]),
            "last_file": str(paths[-1])
        })
    except Exception as exc:
        return web.json_response({"success": False, "error": str(exc)})

