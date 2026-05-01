import os
from aiohttp import web
from server import PromptServer

@PromptServer.instance.routes.post("/aces-exr/listdir")
async def listdir(request):
    try:
        data = await request.json()
        path = data.get("path", "")

        # If no path is provided or requested "drives", list the drives on Windows
        if not path or path.lower() == "drives":
            if os.name == 'nt':
                import ctypes
                bitmask = ctypes.windll.kernel32.GetLogicalDrives()
                drives = []
                for i in range(26):
                    if bitmask & (1 << i):
                        drives.append(f"{chr(65 + i)}:\\")
                return web.json_response({
                    "path": "",
                    "folders": drives,
                    "files": []
                })
            else:
                path = "/"

        # Ensure the path exists and is a directory
        target_path = os.path.abspath(path)
        if not os.path.isdir(target_path):
            # If it's a file or invalid, fallback to its directory or root
            if os.path.exists(os.path.dirname(target_path)):
                target_path = os.path.dirname(target_path)
            else:
                return web.json_response({"error": "Invalid directory path"}, status=400)

        folders = []
        files = []

        with os.scandir(target_path) as it:
            for entry in it:
                try:
                    if entry.is_dir():
                        folders.append(entry.name)
                    elif entry.is_file() and entry.name.lower().endswith(".exr"):
                        files.append(entry.name)
                except Exception:
                    # Ignore files we don't have permission to read
                    pass

        # Sort alphabetically
        folders.sort(key=lambda s: s.lower())
        files.sort(key=lambda s: s.lower())

        # Include parent directory ".." if not at root
        parent = os.path.dirname(target_path)
        if parent and parent != target_path:
            folders.insert(0, "..")

        return web.json_response({
            "path": target_path,
            "folders": folders,
            "files": files
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
