import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ─── Vertex Shader ─────────────────────────────────────────────────────────────
// Full-screen NDC quad. Letterboxing, zoom and pan are handled in the frag shader.
const VERT_SHADER = `
attribute vec2 a_position;
varying vec2 v_texcoord;
void main() {
    v_texcoord = (a_position + 1.0) * 0.5;
    v_texcoord.y = 1.0 - v_texcoord.y;
    gl_Position = vec4(a_position, 0.0, 1.0);
}`;

// ─── Fragment Shader ───────────────────────────────────────────────────────────
// Handles: letterbox aspect-ratio fit, zoom/pan, channel isolation,
//          EV exposure, gamma/sRGB, false color, A/B split.
const FRAG_SHADER = `
precision highp float;
uniform sampler2D u_texture;
uniform sampler2D u_textureB;
uniform bool  u_use_b;
uniform float u_split_pos;
uniform float u_exposure;
uniform float u_gamma;
uniform int   u_channel;
uniform bool  u_false_color;
uniform vec2  u_canvas_size;
uniform vec2  u_img_size;
uniform float u_zoom;
uniform vec2  u_pan;
varying vec2  v_texcoord;   // 0-1 across the full canvas quad

float srgb(float v) {
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0/2.4) - 0.055;
}

// Canvas UV → image UV, accounting for letterbox centering, zoom and pan.
vec2 toImgUV(vec2 cv) {
    float ca = u_canvas_size.x / max(u_canvas_size.y, 1.0);
    float ia = u_img_size.x   / max(u_img_size.y,    1.0);
    float sx  = (ca > ia) ? ia/ca : 1.0;   // letterbox scale X
    float sy  = (ca > ia) ? 1.0  : ca/ia;  // letterbox scale Y
    vec2 uv   = (cv - 0.5) / vec2(sx, sy) + 0.5;
    // apply viewer zoom and pan
    return (uv - 0.5) / u_zoom + 0.5 + u_pan;
}

void main() {
    vec2 cv  = v_texcoord;        // canvas UV (0-1)
    vec2 iuv = toImgUV(cv);       // image UV  (may be outside 0-1)

    // Pixels outside the image → dark background
    if (iuv.x < 0.0 || iuv.x > 1.0 || iuv.y < 0.0 || iuv.y > 1.0) {
        gl_FragColor = vec4(0.06, 0.06, 0.06, 1.0);
        return;
    }

    // A/B split divider line (canvas-space)
    if (u_use_b && abs(cv.x - u_split_pos) < 0.0015) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    // Sample image (A or B depending on split)
    vec4 c = (u_use_b && cv.x > u_split_pos)
           ? texture2D(u_textureB, iuv)
           : texture2D(u_texture,  iuv);

    // EV exposure
    vec3 rgb = c.rgb * pow(2.0, u_exposure);

    // Channel isolation
    if      (u_channel == 1) rgb = vec3(rgb.r);
    else if (u_channel == 2) rgb = vec3(rgb.g);
    else if (u_channel == 3) rgb = vec3(rgb.b);
    else if (u_channel == 4) rgb = vec3(c.a);
    else if (u_channel == 5) rgb = vec3(dot(rgb, vec3(0.2126, 0.7152, 0.0722)));

    // False color / tonemapping
    if (u_false_color) {
        float l = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        if      (l > 1.00) rgb = vec3(1.0, 0.0, 0.0);
        else if (l > 0.80) rgb = vec3(1.0, 0.5, 0.0);
        else if (l > 0.60) rgb = vec3(1.0, 1.0, 0.0);
        else if (l > 0.45) rgb = vec3(0.5, 0.5, 0.5);
        else if (l > 0.35) rgb = vec3(0.0, 1.0, 0.0);
        else if (l > 0.20) rgb = vec3(0.0, 1.0, 1.0);
        else if (l > 0.05) rgb = vec3(0.0, 0.0, 1.0);
        else               rgb = vec3(0.5, 0.0, 0.5);
    } else if (u_gamma < 0.0) {
        // sRGB transfer function
        rgb = vec3(srgb(clamp(rgb.r, 0.0, 1.0)),
                   srgb(clamp(rgb.g, 0.0, 1.0)),
                   srgb(clamp(rgb.b, 0.0, 1.0)));
    } else {
        rgb = pow(max(rgb, 0.0), vec3(1.0 / u_gamma));
    }

    gl_FragColor = vec4(rgb, 1.0);
}`;

// ─── WebGL helpers ─────────────────────────────────────────────────────────────

function mkShader(gl, type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error("[NodEx HDR] Shader compile error:", gl.getShaderInfoLog(s));
        gl.deleteShader(s);
        return null;
    }
    return s;
}

function mkProg(gl, vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, mkShader(gl, gl.VERTEX_SHADER,   vs));
    gl.attachShader(p, mkShader(gl, gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.error("[NodEx HDR] Program link error:", gl.getProgramInfoLog(p));
        return null;
    }
    return p;
}

function mkTex(gl, filter) {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    return t;
}

// ─── DOM helpers ───────────────────────────────────────────────────────────────

function mkBtn(txt, tip = "") {
    const b = document.createElement("button");
    b.innerText = txt;
    b.title = tip;
    Object.assign(b.style, {
        background: "#252525", color: "#bbb",
        border: "1px solid #383838", padding: "2px 8px",
        borderRadius: "4px", cursor: "pointer",
        fontSize: "11px", fontFamily: "inherit",
        lineHeight: "1.5", userSelect: "none",
        transition: "background 0.12s, color 0.12s, border-color 0.12s"
    });
    return b;
}

// Toggle a button's visual "active" state with a colour
function setOn(b, on, bg = "#e8a020", fg = "#111") {
    b._on = on;
    b.style.background  = on ? bg  : "#252525";
    b.style.color       = on ? fg  : "#bbb";
    b.style.borderColor = on ? bg  : "#383838";
}

function mkRow(extra = {}) {
    const d = document.createElement("div");
    Object.assign(d.style, {
        display: "flex", flexWrap: "wrap",
        gap: "4px", alignItems: "center", ...extra
    });
    return d;
}

// ─── Main Extension ────────────────────────────────────────────────────────────

app.registerExtension({
    name: "Nodex.HDRViewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VispyEXRViewer") return;

        // ── Hook: re-render whenever the graph zoom scale changes ─────────────
        // onDrawBackground fires on every LiteGraph redraw, including zoom/pan.
        const _drawBg = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            _drawBg?.call(this, ctx);
            const dw = this.widgets?.find(w => w.name === "hdr_canvas");
            if (!dw?.render) return;
            const s = app.canvas?.ds?.scale ?? 1;
            if (dw._lastScale !== s) {
                dw._lastScale = s;
                requestAnimationFrame(dw.render);
            }
        };

        // ── Hook: set default node size on creation ───────────────────────────
        const _onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            _onNodeCreated?.apply(this, arguments);
            if (!this.size || this.size[0] < 500) {
                this.size = [640, 520];
            }
        };

        // ── Hook: re-render when the node itself is resized ───────────────────
        const _onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            _onResize?.apply(this, arguments);
            const dw = this.widgets?.find(w => w.name === "hdr_canvas");
            if (dw) {
                const nodeW = size ? size[0] : (this.size ? this.size[0] : 640);
                const nodeH = size ? size[1] : (this.size ? this.size[1] : 520);
                const topY  = dw.y || 110;
                const w     = Math.max(280, nodeW - 24);
                const h     = Math.max(200, nodeH - topY - 16);
                if (dw.element) {
                    dw.element.style.width  = w + "px";
                    dw.element.style.height = h + "px";
                }
                if (dw.wrap) {
                    dw.wrap.style.width  = w + "px";
                    dw.wrap.style.height = h + "px";
                }
                if (dw.render) requestAnimationFrame(dw.render);
            }
        };

        // ── Main execution callback ───────────────────────────────────────────
        const _onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = async function (msg) {
            _onExecuted?.apply(this, arguments);

            // Accept new (sequence_a/b) format only
            const seqA = msg.sequence_a;
            if (!seqA?.length) return;
            const seqB = msg.sequence_b ?? null;

            let dw = this.widgets?.find(w => w.name === "hdr_canvas");

            // ─────────────────────────────────────────────────────────────────
            // First-time DOM + WebGL build
            // ─────────────────────────────────────────────────────────────────
            if (!dw) {

                // ── Outer wrapper ─────────────────────────────────────────────
                const wrap = document.createElement("div");
                Object.assign(wrap.style, {
                    display: "flex", flexDirection: "column", gap: "4px",
                    marginTop: "10px", background: "#161616",
                    padding: "8px", borderRadius: "8px",
                    boxSizing: "border-box", width: "100%", height: "100%"
                });

                // ── Controls row ──────────────────────────────────────────────
                const ctrl = mkRow({ marginBottom: "2px" });
                ctrl.addEventListener("mousedown", e => e.stopPropagation());

                // EV
                const evLbl = document.createElement("span");
                evLbl.innerText = "EV:";
                Object.assign(evLbl.style, { color: "#666", fontSize: "11px" });
                const evSlider = document.createElement("input");
                Object.assign(evSlider, { type: "range", min: -10, max: 10, step: 0.1, value: 0 });
                evSlider.style.width = "60px";
                const evVal = document.createElement("span");
                evVal.innerText = "0.0";
                Object.assign(evVal.style, { fontSize: "11px", color: "#ccc", minWidth: "24px" });

                // Gamma
                const gLbl = document.createElement("span");
                gLbl.innerText = "γ:";
                Object.assign(gLbl.style, { color: "#666", fontSize: "11px" });
                const gSlider = document.createElement("input");
                Object.assign(gSlider, { type: "range", min: 0.1, max: 5.0, step: 0.1, value: 2.2 });
                gSlider.style.width = "60px";
                const gVal = document.createElement("span");
                gVal.innerText = "2.2";
                Object.assign(gVal.style, { fontSize: "11px", color: "#ccc", minWidth: "24px" });

                // Buttons
                const resetBtn  = mkBtn("Reset",       "Reset EV, Gamma & Zoom  [R]");
                const srgbBtn   = mkBtn("sRGB",        "sRGB tone curve  [S]");
                const fcBtn     = mkBtn("False Color", "False color overlay  [F]");
                const abBtn     = mkBtn("A/B",         "A/B split compare");
                const smoothBtn = mkBtn("Smooth",      "Toggle bilinear filter  [L]");
                const histBtn   = mkBtn("Hist",        "Histogram overlay  [H]");
                const copyBtn   = mkBtn("⎘ Copy",      "Copy current view to clipboard");
                abBtn.style.display = "none";

                // Channel select
                const chSel = document.createElement("select");
                Object.assign(chSel.style, {
                    background: "#1e1e1e", color: "#ccc",
                    border: "1px solid #383838", borderRadius: "4px",
                    fontSize: "11px", padding: "1px 4px"
                });
                [["0","RGB"],["1","R"],["2","G"],["3","B"],["4","A"],["5","Luma"]]
                    .forEach(([v,t]) => {
                        const o = document.createElement("option");
                        o.value = v; o.text = t; chSel.appendChild(o);
                    });

                [evLbl, evSlider, evVal, gLbl, gSlider, gVal,
                 resetBtn, srgbBtn, fcBtn, abBtn, smoothBtn, histBtn, copyBtn, chSel]
                    .forEach(el => ctrl.appendChild(el));

                // ── Transport / playback row ───────────────────────────────────
                const transp = mkRow({ flexWrap: "nowrap", display: "none", color: "#aaa", fontSize: "11px" });
                transp.addEventListener("mousedown", e => e.stopPropagation());

                const firstBtn = mkBtn("⏮", "First frame  [Home]");
                const prevBtn  = mkBtn("◀",  "Prev frame  [←]");
                const playBtn  = mkBtn("▶",  "Play / Pause  [Space]");
                const nextBtn  = mkBtn("▶|", "Next frame  [→]");
                const lastBtn  = mkBtn("⏭",  "Last frame  [End]");

                const frameLbl = document.createElement("span");
                Object.assign(frameLbl.style, { color: "#ccc", minWidth: "80px", fontSize: "11px" });
                frameLbl.innerText = "1 / 1";

                const scrubSlider = document.createElement("input");
                Object.assign(scrubSlider, { type: "range", min: 0, max: 0, step: 1, value: 0 });
                Object.assign(scrubSlider.style, { flex: "1", minWidth: "40px" });

                const fpsLbl = document.createElement("span");
                fpsLbl.innerText = "FPS:";
                Object.assign(fpsLbl.style, { color: "#666", fontSize: "11px" });
                const fpsSel = document.createElement("select");
                Object.assign(fpsSel.style, {
                    background: "#1e1e1e", color: "#ccc",
                    border: "1px solid #383838", borderRadius: "4px", fontSize: "11px"
                });
                [[6,"6"],[12,"12"],[24,"24"],[30,"30"],[48,"48"],[60,"60"]].forEach(([v,t]) => {
                    const o = document.createElement("option");
                    o.value = v; o.text = t;
                    if (v === 24) o.selected = true;
                    fpsSel.appendChild(o);
                });

                [firstBtn, prevBtn, playBtn, nextBtn, lastBtn,
                 frameLbl, scrubSlider, fpsLbl, fpsSel]
                    .forEach(el => transp.appendChild(el));

                // ── Canvas area ───────────────────────────────────────────────
                const canvasWrap = document.createElement("div");
                Object.assign(canvasWrap.style, {
                    position: "relative", flex: "1", minHeight: "200px",
                    background: "#0c0c0c", borderRadius: "6px", overflow: "hidden"
                });

                // WebGL canvas
                const canvas = document.createElement("canvas");
                canvas.width = 512; canvas.height = 256;
                Object.assign(canvas.style, {
                    position: "absolute", top: "0", left: "0",
                    width: "100%", height: "100%", display: "block"
                });

                // Pixel info HUD (bottom-left overlay)
                const hudDiv = document.createElement("div");
                Object.assign(hudDiv.style, {
                    position: "absolute", bottom: "6px", left: "6px",
                    background: "rgba(0,0,0,0.72)", color: "#00ff88",
                    fontFamily: "monospace", fontSize: "11px", lineHeight: "1.6",
                    padding: "4px 8px", borderRadius: "5px",
                    pointerEvents: "none", display: "none"
                });

                // Histogram 2D-canvas overlay (top-right)
                const histCvs = document.createElement("canvas");
                histCvs.width = 256; histCvs.height = 80;
                Object.assign(histCvs.style, {
                    position: "absolute", top: "6px", right: "6px",
                    width: "128px", height: "40px",
                    background: "rgba(0,0,0,0.55)", borderRadius: "4px",
                    pointerEvents: "none", display: "none"
                });

                // Info badge (top-left)
                const badge = document.createElement("div");
                Object.assign(badge.style, {
                    position: "absolute", top: "6px", left: "6px",
                    background: "rgba(0,0,0,0.55)", color: "#555",
                    fontFamily: "monospace", fontSize: "10px",
                    padding: "2px 6px", borderRadius: "4px",
                    pointerEvents: "none", userSelect: "none"
                });

                // Loading overlay (centred)
                const loadDiv = document.createElement("div");
                Object.assign(loadDiv.style, {
                    position: "absolute", top: "0", left: "0",
                    width: "100%", height: "100%",
                    display: "none", flexDirection: "column",
                    alignItems: "center", justifyContent: "center",
                    background: "rgba(0,0,0,0.6)", gap: "8px"
                });
                const loadText = document.createElement("div");
                Object.assign(loadText.style, { color: "#ccc", fontSize: "12px", fontFamily: "monospace" });
                const loadBarOuter = document.createElement("div");
                Object.assign(loadBarOuter.style, {
                    width: "60%", height: "4px",
                    background: "#333", borderRadius: "2px", overflow: "hidden"
                });
                const loadBarFill = document.createElement("div");
                Object.assign(loadBarFill.style, {
                    height: "100%", background: "#4af",
                    width: "0%", transition: "width 0.15s"
                });
                loadBarOuter.appendChild(loadBarFill);
                loadDiv.appendChild(loadText);
                loadDiv.appendChild(loadBarOuter);

                canvasWrap.appendChild(canvas);
                canvasWrap.appendChild(hudDiv);
                canvasWrap.appendChild(histCvs);
                canvasWrap.appendChild(badge);
                canvasWrap.appendChild(loadDiv);

                wrap.appendChild(ctrl);
                wrap.appendChild(transp);
                wrap.appendChild(canvasWrap);

                dw = this.addDOMWidget("hdr_canvas", "dom", wrap, {
                    getValue() { return ""; },
                    setValue() {}
                });
                dw.wrap = wrap;
                dw.computeSize = function (width) {
                    const nodeW = this.node ? this.node.size[0] : (width || 640);
                    const nodeH = this.node ? this.node.size[1] : 520;
                    const topY  = this.y || 110;
                    return [Math.max(280, nodeW - 24), Math.max(200, nodeH - topY - 16)];
                };

                // Persist refs on dw
                Object.assign(dw, {
                    canvas, hudDiv, histCvs, badge,
                    loadDiv, loadText, loadBarFill,
                    evSlider, evVal, gSlider, gVal,
                    resetBtn, srgbBtn, fcBtn, abBtn, smoothBtn, histBtn, copyBtn,
                    chSel, transp, firstBtn, prevBtn, playBtn, nextBtn, lastBtn,
                    frameLbl, scrubSlider, fpsSel,
                    // viewer state
                    srgbOn: false, fcOn: false, abOn: false,
                    smoothOn: false, histOn: false,
                    splitPos: 0.5,
                    viewZoom: 1.0, viewPanX: 0.0, viewPanY: 0.0,
                    imgWidth: 1, imgHeight: 1,
                    imgWidthB: 1, imgHeightB: 1,
                    framesDataA: [], framesDataB: [],
                    currentFrame: 0, playInterval: null,
                    _lastFrameA: -1, _lastFrameB: -1,
                    _lastScale: 1
                });

                // ── WebGL init ────────────────────────────────────────────────
                const gl   = canvas.getContext("webgl2") || canvas.getContext("webgl");
                const isGL2 = (gl instanceof WebGL2RenderingContext);
                gl.getExtension("OES_texture_float");
                gl.getExtension("OES_texture_float_linear");

                const prog = mkProg(gl, VERT_SHADER, FRAG_SHADER);
                gl.useProgram(prog);

                const buf = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buf);
                gl.bufferData(gl.ARRAY_BUFFER,
                    new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
                const aPos = gl.getAttribLocation(prog, "a_position");
                gl.enableVertexAttribArray(aPos);
                gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

                const tex  = mkTex(gl, gl.NEAREST);
                const texB = mkTex(gl, gl.NEAREST);
                Object.assign(dw, { gl, prog, tex, texB, isGL2 });

                // ── Render function ───────────────────────────────────────────
                // FIX 1: Syncs canvas pixel buffer to actual display size
                // ── Render function ───────────────────────────────────────────
                // Uses clientWidth/clientHeight (unscaled CSS layout pixels) so canvas.width
                // does not shrink in a feedback loop when ComfyUI graph zoom is active.
                const render = () => {
                    if (!dw.framesDataA.length) return;

                    // ── Sync canvas resolution to unscaled CSS layout size ──
                    const cw   = canvas.clientWidth  || 512;
                    const ch   = canvas.clientHeight || 256;
                    const dpr  = window.devicePixelRatio || 1;
                    const pw   = Math.max(1, Math.round(cw * dpr));
                    const ph   = Math.max(1, Math.round(ch * dpr));

                    // Canvas not laid out yet — retry after next layout pass
                    if (cw < 2 || ch < 2) {
                        setTimeout(() => requestAnimationFrame(render), 80);
                        return;
                    }

                    if (canvas.width !== pw || canvas.height !== ph) {
                        canvas.width  = pw;
                        canvas.height = ph;
                    }

                    // ── Upload frame A to GPU (only when it changes) ──────────
                    const fi = dw.currentFrame;
                    if (dw._lastFrameA !== fi) {
                        const fa = dw.framesDataA[fi];
                        if (fa) {
                            const ifmt = dw.isGL2 ? gl.RGBA32F : gl.RGBA;
                            gl.bindTexture(gl.TEXTURE_2D, dw.tex);
                            gl.texImage2D(gl.TEXTURE_2D, 0, ifmt,
                                dw.imgWidth, dw.imgHeight, 0,
                                gl.RGBA, gl.FLOAT, fa);
                        }
                        dw._lastFrameA = fi;
                    }

                    // ── Upload frame B if changed ─────────────────────────────
                    const fiB = Math.max(0, Math.min(fi, dw.framesDataB.length - 1));
                    if (dw._lastFrameB !== fiB && dw.framesDataB.length > 0) {
                        const fb = dw.framesDataB[fiB];
                        if (fb) {
                            const ifmt = dw.isGL2 ? gl.RGBA32F : gl.RGBA;
                            gl.bindTexture(gl.TEXTURE_2D, dw.texB);
                            gl.texImage2D(gl.TEXTURE_2D, 0, ifmt,
                                dw.imgWidthB, dw.imgHeightB, 0,
                                gl.RGBA, gl.FLOAT, fb);
                        }
                        dw._lastFrameB = fiB;
                    }

                    // ── WebGL draw ────────────────────────────────────────────
                    gl.viewport(0, 0, canvas.width, canvas.height);
                    gl.clearColor(0.06, 0.06, 0.06, 1.0);
                    gl.clear(gl.COLOR_BUFFER_BIT);
                    gl.useProgram(prog);

                    const ul = n => gl.getUniformLocation(prog, n);
                    gl.uniform1f(ul("u_exposure"),    parseFloat(evSlider.value));
                    gl.uniform1f(ul("u_gamma"),       dw.srgbOn ? -1.0 : parseFloat(gSlider.value));
                    gl.uniform1i(ul("u_channel"),     parseInt(chSel.value));
                    gl.uniform1i(ul("u_false_color"), dw.fcOn ? 1 : 0);
                    gl.uniform1i(ul("u_use_b"),       (dw.abOn && dw.framesDataB.length > 0) ? 1 : 0);
                    gl.uniform1f(ul("u_split_pos"),   dw.splitPos);
                    gl.uniform2f(ul("u_canvas_size"), canvas.width, canvas.height);
                    gl.uniform2f(ul("u_img_size"),    dw.imgWidth,  dw.imgHeight);
                    gl.uniform1f(ul("u_zoom"),        dw.viewZoom);
                    gl.uniform2f(ul("u_pan"),         dw.viewPanX,  dw.viewPanY);

                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, dw.tex);
                    gl.uniform1i(ul("u_texture"),  0);

                    gl.activeTexture(gl.TEXTURE1);
                    gl.bindTexture(gl.TEXTURE_2D, dw.texB);
                    gl.uniform1i(ul("u_textureB"), 1);

                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

                    // Optional histogram 2D overlay
                    if (dw.histOn) drawHistogram();

                    // Update transport frame counter
                    const total = dw.framesDataA.length;
                    if (total > 1) {
                        dw.frameLbl.innerText = `${fi + 1} / ${total}`;
                        dw.scrubSlider.value  = fi;
                    }
                };
                dw.render = render;

                // ── Histogram 2D canvas overlay ───────────────────────────────
                function drawHistogram() {
                    const fd = dw.framesDataA[dw.currentFrame];
                    if (!fd) return;
                    const W = histCvs.width, H = histCvs.height;
                    const ctx2 = histCvs.getContext("2d");
                    const bins = new Uint32Array(W);
                    const nPx  = fd.length / 4;
                    const ev   = parseFloat(evSlider.value);
                    // Subsample for performance on large images
                    const step = Math.max(1, Math.floor(nPx / 200000));
                    for (let i = 0; i < nPx; i += step) {
                        const l = (0.2126 * fd[i*4] + 0.7152 * fd[i*4+1] + 0.0722 * fd[i*4+2])
                                  * Math.pow(2, ev);
                        const bin = Math.min(W - 1, Math.max(0, Math.floor(l * (W - 1))));
                        bins[bin]++;
                    }
                    const maxB = Math.max(1, ...bins);
                    ctx2.clearRect(0, 0, W, H);
                    ctx2.fillStyle = "rgba(0,0,0,0.75)";
                    ctx2.fillRect(0, 0, W, H);
                    // Grid lines at 0.25, 0.5, 0.75
                    ctx2.strokeStyle = "rgba(255,255,255,0.1)";
                    ctx2.lineWidth   = 1;
                    [0.25, 0.5, 0.75].forEach(x => {
                        ctx2.beginPath();
                        ctx2.moveTo(x * W, 0);
                        ctx2.lineTo(x * W, H);
                        ctx2.stroke();
                    });
                    // Histogram bars
                    ctx2.fillStyle = "rgba(80,180,255,0.9)";
                    for (let x = 0; x < W; x++) {
                        const h = (bins[x] / maxB) * H;
                        ctx2.fillRect(x, H - h, 1, h);
                    }
                }

                // ── Pixel info HUD ────────────────────────────────────────────
                canvas.addEventListener("mousemove", e => {
                    if (!dw.framesDataA.length) return;
                    const rect = canvas.getBoundingClientRect();
                    // Canvas-space UV
                    const cx = (e.clientX - rect.left) / rect.width;
                    const cy = (e.clientY - rect.top)  / rect.height;

                    // Undo letterbox
                    const ca = rect.width / rect.height;
                    const ia = dw.imgWidth / dw.imgHeight;
                    const sx = ca > ia ? ia / ca : 1.0;
                    const sy = ca > ia ? 1.0  : ca / ia;
                    let iu = (cx - 0.5) / sx + 0.5;
                    let iv = (cy - 0.5) / sy + 0.5;

                    // Undo zoom/pan
                    iu = (iu - 0.5) / dw.viewZoom + 0.5 + dw.viewPanX;
                    iv = (iv - 0.5) / dw.viewZoom + 0.5 + dw.viewPanY;

                    const px = Math.floor(iu * dw.imgWidth);
                    const py = Math.floor(iv * dw.imgHeight);

                    if (px < 0 || px >= dw.imgWidth || py < 0 || py >= dw.imgHeight) {
                        hudDiv.style.display = "none";
                        return;
                    }
                    const fd  = dw.framesDataA[dw.currentFrame];
                    if (!fd) return;
                    const idx = (py * dw.imgWidth + px) * 4;
                    const f   = v => v.toFixed(4);
                    hudDiv.innerHTML =
                        `X:${px} Y:${py}<br>` +
                        `R:${f(fd[idx])} G:${f(fd[idx+1])} B:${f(fd[idx+2])} A:${f(fd[idx+3])}`;
                    hudDiv.style.display = "block";
                });
                canvas.addEventListener("mouseleave", () => { hudDiv.style.display = "none"; });

                // ── Zoom & Pan ────────────────────────────────────────────────
                // Scroll wheel → zoom centred on cursor
                // Alt+drag (or Middle-drag) → pan
                let panning = false, panSX = 0, panSY = 0, panSPX = 0, panSPY = 0;
                let splitDragging = false;

                canvas.addEventListener("mousedown", e => {
                    e.stopPropagation();
                    if (dw.abOn && dw.framesDataB.length > 0 && e.button === 0 && !e.altKey) {
                        const rect = canvas.getBoundingClientRect();
                        const cx   = (e.clientX - rect.left) / rect.width;
                        if (Math.abs(cx - dw.splitPos) < 0.04) {
                            splitDragging = true;
                            return;
                        }
                    }
                    // Middle-click OR Alt+Left-click → pan
                    if (e.button === 1 || (e.button === 0 && e.altKey)) {
                        panning = true;
                        panSX = e.clientX; panSY = e.clientY;
                        panSPX = dw.viewPanX; panSPY = dw.viewPanY;
                        canvas.style.cursor = "grabbing";
                        e.preventDefault();
                    }
                });
                window.addEventListener("mousemove", e => {
                    if (splitDragging && dw.abOn) {
                        const rect = canvas.getBoundingClientRect();
                        dw.splitPos = Math.max(0, Math.min(1,
                            (e.clientX - rect.left) / rect.width));
                        render();
                    }
                    if (panning) {
                        const rect = canvas.getBoundingClientRect();
                        const dx = (e.clientX - panSX) / (rect.width  * dw.viewZoom);
                        const dy = (e.clientY - panSY) / (rect.height * dw.viewZoom);
                        dw.viewPanX = panSPX - dx;
                        dw.viewPanY = panSPY - dy;
                        render();
                    }
                });
                window.addEventListener("mouseup", () => {
                    splitDragging = panning = false;
                    canvas.style.cursor = "";
                });

                canvas.addEventListener("wheel", e => {
                    e.preventDefault();
                    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
                    dw.viewZoom = Math.max(0.05, Math.min(64, dw.viewZoom * factor));
                    render();
                }, { passive: false });

                // ── Control button events ─────────────────────────────────────
                evSlider.oninput = () => {
                    evVal.innerText = parseFloat(evSlider.value).toFixed(1);
                    render();
                };
                gSlider.oninput = () => {
                    gVal.innerText = parseFloat(gSlider.value).toFixed(1);
                    render();
                };
                resetBtn.onclick = () => {
                    evSlider.value = "0";  evVal.innerText = "0.0";
                    gSlider.value  = "2.2"; gVal.innerText = "2.2";
                    dw.viewZoom = 1.0; dw.viewPanX = 0; dw.viewPanY = 0;
                    render();
                };
                srgbBtn.onclick = () => {
                    dw.srgbOn = !dw.srgbOn;
                    setOn(srgbBtn, dw.srgbOn, "#e8a020", "#111");
                    gSlider.disabled = dw.srgbOn;
                    render();
                };
                fcBtn.onclick = () => {
                    dw.fcOn = !dw.fcOn;
                    setOn(fcBtn, dw.fcOn, "#e82020", "#fff");
                    render();
                };
                abBtn.onclick = () => {
                    dw.abOn = !dw.abOn;
                    setOn(abBtn, dw.abOn, "#2080e8", "#fff");
                    render();
                };
                smoothBtn.onclick = () => {
                    dw.smoothOn = !dw.smoothOn;
                    setOn(smoothBtn, dw.smoothOn, "#20b870", "#111");
                    const f = dw.smoothOn ? gl.LINEAR : gl.NEAREST;
                    [dw.tex, dw.texB].forEach(t => {
                        gl.bindTexture(gl.TEXTURE_2D, t);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
                    });
                    // Force re-upload with new filter
                    dw._lastFrameA = -1; dw._lastFrameB = -1;
                    render();
                };
                histBtn.onclick = () => {
                    dw.histOn = !dw.histOn;
                    setOn(histBtn, dw.histOn, "#9040e8", "#fff");
                    histCvs.style.display = dw.histOn ? "block" : "none";
                    render();
                };
                copyBtn.onclick = () => {
                    render();
                    canvas.toBlob(async blob => {
                        try {
                            await navigator.clipboard.write(
                                [new ClipboardItem({ "image/png": blob })]
                            );
                            const orig = copyBtn.innerText;
                            copyBtn.innerText = "✓ Copied!";
                            setTimeout(() => { copyBtn.innerText = orig; }, 1800);
                        } catch (err) {
                            console.error("[NodEx HDR] Clipboard error:", err);
                            copyBtn.innerText = "✗ Failed";
                            setTimeout(() => { copyBtn.innerText = "⎘ Copy"; }, 1800);
                        }
                    }, "image/png");
                };
                chSel.onchange = render;

                // ── Transport / playback events ────────────────────────────────
                const stopPlay = () => {
                    if (dw.playInterval) {
                        clearInterval(dw.playInterval);
                        dw.playInterval = null;
                    }
                    playBtn.innerText = "▶";
                    setOn(playBtn, false);
                };
                const startPlay = () => {
                    stopPlay();
                    const fps = parseInt(fpsSel.value) || 24;
                    dw.playInterval = setInterval(() => {
                        dw.currentFrame = (dw.currentFrame + 1) % dw.framesDataA.length;
                        render();
                    }, 1000 / fps);
                    playBtn.innerText = "⏸";
                    setOn(playBtn, true, "#2080e8", "#fff");
                };
                dw.stopPlay  = stopPlay;
                dw.startPlay = startPlay;

                firstBtn.onclick = () => { stopPlay(); dw.currentFrame = 0; render(); };
                lastBtn.onclick  = () => {
                    stopPlay();
                    dw.currentFrame = Math.max(0, dw.framesDataA.length - 1);
                    render();
                };
                prevBtn.onclick = () => {
                    stopPlay();
                    dw.currentFrame = Math.max(0, dw.currentFrame - 1);
                    render();
                };
                nextBtn.onclick = () => {
                    stopPlay();
                    dw.currentFrame = Math.min(dw.framesDataA.length - 1, dw.currentFrame + 1);
                    render();
                };
                playBtn.onclick = () => {
                    if (dw.playInterval) stopPlay(); else startPlay();
                };
                scrubSlider.oninput = () => {
                    stopPlay();
                    dw.currentFrame = parseInt(scrubSlider.value);
                    render();
                };
                fpsSel.onchange = () => {
                    if (dw.playInterval) { stopPlay(); startPlay(); }
                };

                // ── Keyboard shortcuts ─────────────────────────────────────────
                document.addEventListener("keydown", e => {
                    if (!dw.framesDataA.length) return;
                    const tag = document.activeElement?.tagName?.toLowerCase();
                    if (tag === "input" || tag === "textarea" || tag === "select") return;
                    switch (e.key.toLowerCase()) {
                        case "r":          resetBtn.click();   break;
                        case "f":          fcBtn.click();      break;
                        case "s":          srgbBtn.click();    break;
                        case "h":          histBtn.click();    break;
                        case "l":          smoothBtn.click();  break;
                        case " ":          e.preventDefault(); playBtn.click(); break;
                        case "arrowleft":  e.preventDefault(); prevBtn.click();  break;
                        case "arrowright": e.preventDefault(); nextBtn.click();  break;
                        case "home":       firstBtn.click();   break;
                        case "end":        lastBtn.click();    break;
                    }
                });

                // ── ResizeObserver ─────────────────────────────────────────────
                if (window.ResizeObserver) {
                    new ResizeObserver(() => requestAnimationFrame(render)).observe(canvasWrap);
                }

            } // ── end of first-time DOM build ────────────────────────────────

            // ─────────────────────────────────────────────────────────────────
            // Every execution: stop playback, fetch new frames, restart
            // ─────────────────────────────────────────────────────────────────

            // Stop any in-progress playback
            if (dw.playInterval) {
                clearInterval(dw.playInterval);
                dw.playInterval = null;
                dw.playBtn.innerText = "▶";
                setOn(dw.playBtn, false);
            }

            // Reset frame state
            dw.framesDataA  = [];
            dw.framesDataB  = [];
            dw._lastFrameA  = -1;
            dw._lastFrameB  = -1;
            dw.currentFrame = 0;
            dw.imgWidth     = seqA[0].width;
            dw.imgHeight    = seqA[0].height;

            // Show loading overlay
            dw.loadDiv.style.display     = "flex";
            dw.loadBarFill.style.width   = "0%";
            dw.loadText.innerText        = "Loading…";

            const totalA = seqA.length;
            const hasB   = !!(seqB?.length);

            // Fetch sequence A
            for (let i = 0; i < totalA; i++) {
                dw.loadText.innerText      = `Loading  ${i + 1} / ${totalA}…`;
                dw.loadBarFill.style.width = `${((i + 1) / totalA * (hasB ? 50 : 100)).toFixed(0)}%`;
                try {
                    const r = await fetch(`/nodex_hdr/view?filename=${seqA[i].filename}`);
                    dw.framesDataA.push(new Float32Array(await r.arrayBuffer()));
                } catch (err) {
                    console.error("[NodEx HDR] Frame A fetch error:", err);
                }
            }

            // Fetch sequence B (optional)
            if (hasB) {
                dw.imgWidthB  = seqB[0].width;
                dw.imgHeightB = seqB[0].height;
                const totalB  = seqB.length;
                for (let i = 0; i < totalB; i++) {
                    dw.loadText.innerText      = `Loading B  ${i + 1} / ${totalB}…`;
                    dw.loadBarFill.style.width = `${(50 + (i + 1) / totalB * 50).toFixed(0)}%`;
                    try {
                        const r = await fetch(`/nodex_hdr/view?filename=${seqB[i].filename}`);
                        dw.framesDataB.push(new Float32Array(await r.arrayBuffer()));
                    } catch (err) {
                        console.error("[NodEx HDR] Frame B fetch error:", err);
                    }
                }
                dw.abBtn.style.display = "block";
                if (!dw.abOn) dw.abBtn.click();
            } else {
                dw.imgWidthB = 1; dw.imgHeightB = 1;
                dw.abBtn.style.display = "none";
                if (dw.abOn) { dw.abOn = false; setOn(dw.abBtn, false); }
            }

            // Show/hide transport bar based on frame count
            const nFrames = dw.framesDataA.length;
            if (nFrames > 1) {
                dw.transp.style.display    = "flex";
                dw.scrubSlider.max         = nFrames - 1;
                dw.scrubSlider.value       = 0;
                dw.frameLbl.innerText      = `1 / ${nFrames}`;
                // Do NOT auto-play — user must press ▶ to start
            } else {
                dw.transp.style.display = "none";
            }

            // Update info badge
            const seqSuffix = nFrames > 1 ? `  ·  ${nFrames} frames` : "";
            dw.badge.innerText = `${dw.imgWidth} × ${dw.imgHeight}${seqSuffix}`;

            // Sync widget container dimensions to node size
            if (this.size) {
                const nodeW = this.size[0];
                const nodeH = this.size[1];
                const topY  = dw.y || 110;
                const w     = Math.max(280, nodeW - 24);
                const h     = Math.max(200, nodeH - topY - 16);
                if (dw.element) {
                    dw.element.style.width  = w + "px";
                    dw.element.style.height = h + "px";
                }
                if (dw.wrap) {
                    dw.wrap.style.width  = w + "px";
                    dw.wrap.style.height = h + "px";
                }
            }

            // Hide loading overlay and render first frame.
            dw.loadDiv.style.display = "none";
            dw.render();
            setTimeout(() => requestAnimationFrame(dw.render), 120);
            setTimeout(() => requestAnimationFrame(dw.render), 500);
        };
    }
});
