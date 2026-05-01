import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";

// Custom Dialog for EXR File Browser
class EXRBrowserDialog extends ComfyDialog {
    constructor() {
        super();
        this.element.classList.add("aces-exr-browser");
        this.currentPath = "";
        this.onSelect = null;
        
        // Build UI structure
        this.pathDisplay = $el("div.aces-exr-path", {
            style: {
                marginBottom: "10px",
                padding: "5px",
                backgroundColor: "rgba(0,0,0,0.5)",
                wordBreak: "break-all",
                fontFamily: "monospace"
            }
        });
        
        this.fileList = $el("div.aces-exr-list", {
            style: {
                maxHeight: "400px",
                overflowY: "auto",
                backgroundColor: "rgba(0,0,0,0.3)",
                padding: "5px",
                display: "flex",
                flexDirection: "column",
                gap: "2px"
            }
        });
        
        const closeBtn = $el("button", {
            type: "button",
            textContent: "Cancel",
            onclick: () => this.close()
        });

        const drivesBtn = $el("button", {
            type: "button",
            textContent: "Drives",
            onclick: () => this.loadPath("drives"),
            style: { marginRight: "10px" }
        });

        this.selectDirBtn = $el("button", {
            type: "button",
            textContent: "Select Folder",
            style: { display: "none", marginRight: "10px", backgroundColor: "#2e7d32", color: "white" },
            onclick: () => {
                if (this.onSelect) this.onSelect(this.currentPath);
                this.close();
            }
        });

        const rightBtns = $el("div", {}, [this.selectDirBtn, closeBtn]);

        const buttonContainer = $el("div", {
            style: { marginTop: "10px", display: "flex", justifyContent: "space-between" }
        }, [
            drivesBtn,
            rightBtns
        ]);

        this.content = $el("div", {
            style: { width: "500px", display: "flex", flexDirection: "column", color: "white" }
        }, [
            this.titleEl = $el("h3", { textContent: "Select EXR File", style: { marginTop: 0 } }),
            this.pathDisplay,
            this.fileList,
            buttonContainer
        ]);
    }

    close() {
        super.close();
        // Remove from DOM to prevent leaking elements
        if (this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
    }

    async loadPath(targetPath) {
        try {
            const resp = await api.fetchApi("/aces-exr/listdir", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: targetPath })
            });
            const data = await resp.json();
            
            if (data.error) {
                alert("Error: " + data.error);
                return;
            }
            
            this.currentPath = data.path;
            this.pathDisplay.textContent = this.currentPath || "Select Drive";
            this.fileList.innerHTML = "";
            
            for (const folder of data.folders) {
                const item = $el("div", {
                    textContent: `📁 ${folder}`,
                    style: { cursor: "pointer", padding: "5px", backgroundColor: "rgba(255,255,255,0.05)" },
                    onmouseenter: (e) => e.target.style.backgroundColor = "rgba(255,255,255,0.1)",
                    onmouseleave: (e) => e.target.style.backgroundColor = "rgba(255,255,255,0.05)",
                    onclick: () => {
                        let nextPath = folder;
                        if (this.currentPath) {
                            if (folder === "..") {
                                nextPath = `${this.currentPath}/${folder}`;
                            } else {
                                nextPath = `${this.currentPath}/${folder}`;
                            }
                        }
                        this.loadPath(nextPath);
                    }
                });
                this.fileList.appendChild(item);
            }
            
            for (const file of data.files) {
                const item = $el("div", {
                    textContent: `📄 ${file}`,
                    style: { cursor: "pointer", padding: "5px", color: "#64b5f6" },
                    onmouseenter: (e) => e.target.style.backgroundColor = "rgba(255,255,255,0.1)",
                    onmouseleave: (e) => e.target.style.backgroundColor = "transparent",
                    onclick: () => {
                        if (this.onSelect) {
                            const sep = this.currentPath.includes("\\") || this.currentPath.match(/^[A-Z]:/i) ? "\\" : "/";
                            let finalPath = `${this.currentPath}${sep}${file}`;
                            // Cleanup double slashes
                            finalPath = finalPath.replace(/\\\\/g, "\\").replace(/\/\//g, "/");
                            this.onSelect(finalPath);
                        }
                        this.close();
                    }
                });
                this.fileList.appendChild(item);
            }
        } catch (e) {
            console.error(e);
            alert("Failed to load directory.");
        }
    }

    show(currentVal, onSelectCallback, dirMode = false) {
        this.onSelect = onSelectCallback;
        this.dirMode = dirMode;
        
        if (dirMode) {
            this.titleEl.textContent = "Select Output Directory";
            this.selectDirBtn.style.display = "inline-block";
        } else {
            this.titleEl.textContent = "Select EXR File";
            this.selectDirBtn.style.display = "none";
        }
        
        // Try to open the directory of the currently typed path, or fallback to drives
        let startPath = currentVal || "drives";
        if (!dirMode && startPath !== "drives" && startPath.lastIndexOf("\\") > 0) {
            startPath = startPath.substring(0, startPath.lastIndexOf("\\"));
        } else if (!dirMode && startPath !== "drives" && startPath.lastIndexOf("/") > 0) {
            startPath = startPath.substring(0, startPath.lastIndexOf("/"));
        }
        
        this.loadPath(startPath);
        // ComfyDialog show method replaces children if passed an element
        super.show(this.content);
        this.element.style.zIndex = 10001; // ensure above other modals
    }
}

app.registerExtension({
    name: "ComfyUI.ACES-EXR.PathBrowser",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ACESLoadEXRFromPath" || nodeData.name === "ACESSaveEXR") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const isSave = nodeData.name === "ACESSaveEXR";
                const widgetName = isSave ? "output_dir" : "exr_path";
                const btnLabel = isSave ? "Browse Dir..." : "Browse Path...";
                
                const pathWidget = this.widgets.find((w) => w.name === widgetName);
                if (pathWidget) {
                    const browseBtn = this.addWidget("button", btnLabel, "browse", () => {
                        const dialog = new EXRBrowserDialog();
                        dialog.show(pathWidget.value, (selectedPath) => {
                            pathWidget.value = selectedPath;
                            // Trigger update if needed
                            if (pathWidget.callback) {
                                pathWidget.callback(selectedPath);
                            }
                        }, isSave);
                    });
                    browseBtn.serialize = false; // don't save the button state
                }
                return r;
            };
        }
    }
});
