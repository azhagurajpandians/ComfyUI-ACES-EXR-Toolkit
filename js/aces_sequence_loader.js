import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

async function fetchSequenceInfo(path, filename) {
    try {
        const resp = await api.fetchApi("/aces/sequence_info", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: path || "", filename: filename || "" })
        });
        if (!resp.ok) return null;
        return await resp.json();
    } catch (e) {
        console.warn("[ACES EXR] Failed to fetch sequence info:", e);
        return null;
    }
}

async function autoDetectSequence(node, silent = false) {
    const fileWidget = node.widgets?.find(w => w.name === "exr_file");
    const pathWidget = node.widgets?.find(w => w.name === "exr_path");
    const startWidget = node.widgets?.find(w => w.name === "start_frame");
    const endWidget = node.widgets?.find(w => w.name === "end_frame");

    const pathVal = pathWidget?.value || "";
    const fileVal = fileWidget?.value || "";

    if (!pathVal && (!fileVal || fileVal.startsWith("<"))) {
        return;
    }

    const data = await fetchSequenceInfo(pathVal, fileVal);
    if (data && data.success) {
        if (startWidget && typeof data.start_frame === "number") {
            startWidget.value = data.start_frame;
            startWidget.callback?.(startWidget.value);
        }
        if (endWidget && typeof data.end_frame === "number") {
            endWidget.value = data.end_frame;
            endWidget.callback?.(endWidget.value);
        }

        const detectBtn = node.widgets?.find(w => w._is_aces_detect_btn);
        if (detectBtn) {
            const oldName = detectBtn.name;
            detectBtn.name = `✓ Frames: ${data.start_frame}-${data.end_frame} (${data.width}x${data.height})`;
            setTimeout(() => {
                detectBtn.name = "⚡ Auto Detect Range";
                app.graph?.setDirtyCanvas(true, true);
            }, 3000);
        }

        app.graph?.setDirtyCanvas(true, true);
        return data;
    } else if (!silent && data?.error) {
        console.warn("[ACES EXR] Sequence detection:", data.error);
        const detectBtn = node.widgets?.find(w => w._is_aces_detect_btn);
        if (detectBtn) {
            detectBtn.name = `⚠ ${data.error.slice(0, 25)}`;
            setTimeout(() => {
                detectBtn.name = "⚡ Auto Detect Range";
                app.graph?.setDirtyCanvas(true, true);
            }, 3000);
            app.graph?.setDirtyCanvas(true, true);
        }
    }
    return null;
}

function attachSequenceHooks(node) {
    if (node._aces_hooks_attached) return;
    node._aces_hooks_attached = true;

    const fileWidget = node.widgets?.find(w => w.name === "exr_file");
    const pathWidget = node.widgets?.find(w => w.name === "exr_path");

    if (fileWidget) {
        const origFileCb = fileWidget.callback;
        fileWidget.callback = function (v) {
            const res = origFileCb ? origFileCb.apply(this, arguments) : undefined;
            autoDetectSequence(node, true);
            return res;
        };
    }

    if (pathWidget) {
        const origPathCb = pathWidget.callback;
        pathWidget.callback = function (v) {
            const res = origPathCb ? origPathCb.apply(this, arguments) : undefined;
            autoDetectSequence(node, true);
            return res;
        };
        if (pathWidget.inputEl) {
            pathWidget.inputEl.addEventListener("change", () => autoDetectSequence(node, true));
            pathWidget.inputEl.addEventListener("blur", () => autoDetectSequence(node, true));
        }
    }

    // Add Auto Detect Range button
    const btn = node.addWidget("button", "⚡ Auto Detect Range", null, () => {
        autoDetectSequence(node, false);
    });
    btn.serialize = false;
    btn._is_aces_detect_btn = true;

    // Check on startup/load after graph initializes
    setTimeout(() => {
        const pVal = pathWidget?.value;
        const fVal = fileWidget?.value;
        if (pVal || (fVal && !fVal.startsWith("<"))) {
            autoDetectSequence(node, true);
        }
    }, 200);
}

app.registerExtension({
    name: "AcesEXR.SequenceLoader",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ACESLoadEXR" && nodeData.name !== "ACESLoadEXRFromPath") {
            return;
        }

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const res = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
            attachSequenceHooks(this);
            return res;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const res = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
            attachSequenceHooks(this);
            return res;
        };
    }
});
