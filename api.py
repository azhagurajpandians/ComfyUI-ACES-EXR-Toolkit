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
