import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const VERT_SHADER = `
attribute vec2 a_position;
varying vec2 v_texcoord;
uniform vec2 u_scale;
void main() {
    v_texcoord = (a_position + 1.0) * 0.5;
    v_texcoord.y = 1.0 - v_texcoord.y; // flip y
    gl_Position = vec4(a_position * u_scale, 0.0, 1.0);
}
`;

const FRAG_SHADER = `
precision highp float;
uniform sampler2D u_texture;
uniform sampler2D u_textureB;
uniform bool u_use_b;
uniform float u_split_pos;

uniform float u_exposure;
uniform float u_gamma;
uniform int u_channel; // 0=RGB, 1=R, 2=G, 3=B, 4=A, 5=Luma
uniform bool u_false_color;
varying vec2 v_texcoord;

float srgb(float v) {
    if (v <= 0.0031308) return 12.92 * v;
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

void main() {
    vec4 c = texture2D(u_texture, v_texcoord);
    if (u_use_b && v_texcoord.x > u_split_pos) {
        c = texture2D(u_textureB, v_texcoord);
    }
    vec3 rgb = c.rgb * pow(2.0, u_exposure);

    if (u_channel == 1) rgb = vec3(rgb.r);
    else if (u_channel == 2) rgb = vec3(rgb.g);
    else if (u_channel == 3) rgb = vec3(rgb.b);
    else if (u_channel == 4) rgb = vec3(c.a);
    else if (u_channel == 5) {
        float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        rgb = vec3(luma);
    }

    if (u_false_color) {
        float l = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
        if (l > 1.0) rgb = vec3(1.0, 0.0, 0.0); // Red
        else if (l > 0.8) rgb = vec3(1.0, 0.5, 0.0); // Orange
        else if (l > 0.6) rgb = vec3(1.0, 1.0, 0.0); // Yellow
        else if (l > 0.45) rgb = vec3(0.5, 0.5, 0.5); // Light Gray
        else if (l > 0.35) rgb = vec3(0.0, 1.0, 0.0); // Green
        else if (l > 0.2) rgb = vec3(0.0, 1.0, 1.0); // Cyan
        else if (l > 0.05) rgb = vec3(0.0, 0.0, 1.0); // Blue
        else rgb = vec3(0.5, 0.0, 0.5); // Purple
    } else {
        if (u_gamma < 0.0) {
            rgb = vec3(srgb(clamp(rgb.r, 0.0, 1.0)),
                       srgb(clamp(rgb.g, 0.0, 1.0)),
                       srgb(clamp(rgb.b, 0.0, 1.0)));
        } else {
            rgb = pow(max(rgb, 0.0), vec3(1.0 / u_gamma));
        }
    }

    if (u_use_b && abs(v_texcoord.x - u_split_pos) < 0.002) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); // Split line
        return;
    }
    gl_FragColor = vec4(rgb, 1.0);
}
`;

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vert, frag) {
    const vertShader = createShader(gl, gl.VERTEX_SHADER, vert);
    const fragShader = createShader(gl, gl.FRAGMENT_SHADER, frag);
    const prog = gl.createProgram();
    gl.attachShader(prog, vertShader);
    gl.attachShader(prog, fragShader);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(prog));
        return null;
    }
    return prog;
}

app.registerExtension({
    name: "Nodex.HDRViewer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VispyEXRViewer") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = async function (message) {
                onExecuted?.apply(this, arguments);
                
                if (message.hdr_data && message.hdr_data[0]) {
                    const data = message.hdr_data[0];
                    const url = `/nodex_hdr/view?filename=${data.filename}`;
                    
                    let domWidget = this.widgets?.find(w => w.name === "hdr_canvas");
                    let canvas, gl;
                    let controlsDiv;
                    
                    if (!domWidget) {
                        const container = document.createElement("div");
                        container.style.width = "100%";
                        container.style.height = "100%";
                        container.style.display = "flex";
                        container.style.flexDirection = "column";
                        container.style.gap = "4px";
                        container.style.marginTop = "10px";
                        container.style.background = "#1e1e1e";
                        container.style.padding = "8px";
                        container.style.borderRadius = "8px";
                        container.style.boxSizing = "border-box";
                        
                        controlsDiv = document.createElement("div");
                        controlsDiv.style.display = "flex";
                        controlsDiv.style.flexWrap = "wrap";
                        controlsDiv.style.gap = "8px";
                        controlsDiv.style.color = "#ccc";
                        controlsDiv.style.fontSize = "11px";
                        controlsDiv.style.alignItems = "center";
                        controlsDiv.style.marginBottom = "4px";
                        
                        // EV Slider
                        const evLabel = document.createElement("label");
                        evLabel.innerText = "EV:";
                        const evInput = document.createElement("input");
                        evInput.type = "range";
                        evInput.min = "-10";
                        evInput.max = "10";
                        evInput.step = "0.1";
                        evInput.value = "0";
                        evInput.style.width = "60px";
                        const evVal = document.createElement("span");
                        evVal.innerText = "0.0";
                        
                        // Gamma Slider
                        const gLabel = document.createElement("label");
                        gLabel.innerText = "Gamma:";
                        const gInput = document.createElement("input");
                        gInput.type = "range";
                        gInput.min = "0.1";
                        gInput.max = "5.0";
                        gInput.step = "0.1";
                        gInput.value = "2.2";
                        gInput.style.width = "60px";
                        const gVal = document.createElement("span");
                        gVal.innerText = "2.2";
                        
                        // Reset Button
                        const resetBtn = document.createElement("button");
                        resetBtn.innerText = "Reset";
                        resetBtn.title = "Reset EV and Gamma";
                        resetBtn.style.background = "#444";
                        resetBtn.style.color = "#ccc";
                        resetBtn.style.border = "1px solid #555";
                        resetBtn.style.padding = "2px 6px";
                        resetBtn.style.borderRadius = "3px";
                        resetBtn.style.cursor = "pointer";
                        
                        // sRGB Button
                        const srgbBtn = document.createElement("button");
                        srgbBtn.innerText = "sRGB";
                        srgbBtn.style.background = "#333";
                        srgbBtn.style.color = "#ccc";
                        srgbBtn.style.border = "1px solid #444";
                        srgbBtn.style.padding = "2px 6px";
                        srgbBtn.style.borderRadius = "3px";
                        srgbBtn.style.cursor = "pointer";
                        let srgbOn = false;

                        // False Color Button
                        const fcBtn = document.createElement("button");
                        fcBtn.innerText = "False Color";
                        fcBtn.style.background = "#333";
                        fcBtn.style.color = "#ccc";
                        fcBtn.style.border = "1px solid #444";
                        fcBtn.style.padding = "2px 6px";
                        fcBtn.style.borderRadius = "3px";
                        fcBtn.style.cursor = "pointer";
                        let fcOn = false;

                        // A/B Toggle Button
                        const abBtn = document.createElement("button");
                        abBtn.innerText = "A/B Split";
                        abBtn.style.background = "#333";
                        abBtn.style.color = "#ccc";
                        abBtn.style.border = "1px solid #444";
                        abBtn.style.padding = "2px 6px";
                        abBtn.style.borderRadius = "3px";
                        abBtn.style.cursor = "pointer";
                        abBtn.style.display = "none"; // Hide until B is loaded
                        let abOn = false;
                        
                        // Channel Select
                        const chSelect = document.createElement("select");
                        ["RGB", "R", "G", "B", "A", "Luma"].forEach((ch, i) => {
                            const opt = document.createElement("option");
                            opt.value = i;
                            opt.text = ch;
                            chSelect.appendChild(opt);
                        });
                        chSelect.style.background = "#222";
                        chSelect.style.color = "#ccc";
                        chSelect.style.border = "1px solid #444";
                        chSelect.style.borderRadius = "3px";
                        
                        canvas = document.createElement("canvas");
                        canvas.style.width = "100%";
                        canvas.style.height = "100%";
                        canvas.style.flexGrow = "1";
                        canvas.style.minHeight = "256px"; 
                        canvas.style.background = "#000";
                        canvas.style.borderRadius = "4px";
                        
                        controlsDiv.appendChild(evLabel);
                        controlsDiv.appendChild(evInput);
                        controlsDiv.appendChild(evVal);
                        controlsDiv.appendChild(gLabel);
                        controlsDiv.appendChild(gInput);
                        controlsDiv.appendChild(gVal);
                        controlsDiv.appendChild(resetBtn);
                        controlsDiv.appendChild(srgbBtn);
                        controlsDiv.appendChild(fcBtn);
                        controlsDiv.appendChild(abBtn);
                        controlsDiv.appendChild(chSelect);
                        
                        container.appendChild(controlsDiv);
                        container.appendChild(canvas);
                        
                        domWidget = this.addDOMWidget("hdr_canvas", "dom", container);
                        domWidget.canvas = canvas;
                        domWidget.evInput = evInput;
                        domWidget.evVal = evVal;
                        domWidget.gInput = gInput;
                        domWidget.gVal = gVal;
                        domWidget.srgbBtn = srgbBtn;
                        domWidget.srgbOn = srgbOn;
                        domWidget.fcBtn = fcBtn;
                        domWidget.fcOn = fcOn;
                        domWidget.abBtn = abBtn;
                        domWidget.abOn = abOn;
                        domWidget.splitPos = 0.5;
                        domWidget.chSelect = chSelect;
                        domWidget.imgWidth = 1;
                        domWidget.imgHeight = 1;
                        
                        // Prevent dragging canvas from moving node, enable split sliding
                        let isDraggingSplit = false;
                        canvas.onmousedown = (e) => {
                            e.stopPropagation();
                            if (domWidget.abOn) isDraggingSplit = true;
                        };
                        canvas.onmousemove = (e) => {
                            if (isDraggingSplit && domWidget.abOn) {
                                const rect = canvas.getBoundingClientRect();
                                domWidget.splitPos = Math.max(0.0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
                                render();
                            }
                        };
                        window.addEventListener("mouseup", () => isDraggingSplit = false);
                        controlsDiv.onmousedown = (e) => e.stopPropagation();
                        
                        // WebGL Init
                        gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
                        if (!gl.getExtension("OES_texture_float")) {
                            console.error("WebGL Float textures not supported!");
                        }
                        gl.getExtension("OES_texture_float_linear"); 
                        
                        const prog = createProgram(gl, VERT_SHADER, FRAG_SHADER);
                        gl.useProgram(prog);
                        
                        const posBuffer = gl.createBuffer();
                        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
                        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                            -1,-1,  1,-1,  -1,1,  1,1
                        ]), gl.STATIC_DRAW);
                        
                        const aPos = gl.getAttribLocation(prog, "a_position");
                        gl.enableVertexAttribArray(aPos);
                        gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
                        
                        const tex = gl.createTexture();
                        gl.bindTexture(gl.TEXTURE_2D, tex);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

                        const texB = gl.createTexture();
                        gl.bindTexture(gl.TEXTURE_2D, texB);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                        
                        domWidget.gl = gl;
                        domWidget.prog = prog;
                        domWidget.tex = tex;
                        domWidget.texB = texB;
                        
                        // Setup render loop triggers
                        const render = () => {
                            if (!domWidget.texData) return;
                            
                            const rect = canvas.getBoundingClientRect();
                            if (rect.width > 0 && rect.height > 0) {
                                if (canvas.width !== rect.width || canvas.height !== rect.height) {
                                    canvas.width = rect.width;
                                    canvas.height = rect.height;
                                }
                            }
                            gl.viewport(0, 0, canvas.width, canvas.height);
                            gl.clearColor(0.0, 0.0, 0.0, 1.0);
                            gl.clear(gl.COLOR_BUFFER_BIT);
                            
                            gl.useProgram(prog);
                            
                            // Aspect Ratio Fitting (Letterbox)
                            const canvasAspect = canvas.width / canvas.height;
                            const imgAspect = domWidget.imgWidth / domWidget.imgHeight;
                            let scaleX = 1.0;
                            let scaleY = 1.0;
                            if (canvasAspect > imgAspect) {
                                scaleX = imgAspect / canvasAspect;
                            } else {
                                scaleY = canvasAspect / imgAspect;
                            }
                            gl.uniform2f(gl.getUniformLocation(prog, "u_scale"), scaleX, scaleY);
                            
                            gl.uniform1f(gl.getUniformLocation(prog, "u_exposure"), parseFloat(evInput.value));
                            gl.uniform1f(gl.getUniformLocation(prog, "u_gamma"), domWidget.srgbOn ? -1.0 : parseFloat(gInput.value));
                            gl.uniform1i(gl.getUniformLocation(prog, "u_channel"), parseInt(chSelect.value));
                            gl.uniform1i(gl.getUniformLocation(prog, "u_false_color"), domWidget.fcOn ? 1 : 0);
                            
                            gl.uniform1i(gl.getUniformLocation(prog, "u_use_b"), domWidget.abOn ? 1 : 0);
                            gl.uniform1f(gl.getUniformLocation(prog, "u_split_pos"), domWidget.splitPos);
                            
                            gl.activeTexture(gl.TEXTURE0);
                            gl.bindTexture(gl.TEXTURE_2D, domWidget.tex);
                            gl.uniform1i(gl.getUniformLocation(prog, "u_texture"), 0);

                            if (domWidget.hasTexB) {
                                gl.activeTexture(gl.TEXTURE1);
                                gl.bindTexture(gl.TEXTURE_2D, domWidget.texB);
                                gl.uniform1i(gl.getUniformLocation(prog, "u_textureB"), 1);
                            }
                            
                            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
                        };
                        
                        evInput.oninput = () => { evVal.innerText = parseFloat(evInput.value).toFixed(1); render(); };
                        gInput.oninput = () => { gVal.innerText = parseFloat(gInput.value).toFixed(1); render(); };
                        
                        resetBtn.onclick = () => {
                            evInput.value = "0";
                            evVal.innerText = "0.0";
                            gInput.value = "2.2";
                            gVal.innerText = "2.2";
                            render();
                        };

                        srgbBtn.onclick = () => {
                            domWidget.srgbOn = !domWidget.srgbOn;
                            srgbBtn.style.background = domWidget.srgbOn ? "#e8a020" : "#333";
                            srgbBtn.style.color = domWidget.srgbOn ? "#111" : "#ccc";
                            gInput.disabled = domWidget.srgbOn;
                            render();
                        };
                        
                        fcBtn.onclick = () => {
                            domWidget.fcOn = !domWidget.fcOn;
                            fcBtn.style.background = domWidget.fcOn ? "#e82020" : "#333";
                            fcBtn.style.color = domWidget.fcOn ? "#fff" : "#ccc";
                            render();
                        };

                        abBtn.onclick = () => {
                            domWidget.abOn = !domWidget.abOn;
                            abBtn.style.background = domWidget.abOn ? "#2080e8" : "#333";
                            abBtn.style.color = domWidget.abOn ? "#fff" : "#ccc";
                            render();
                        };

                        chSelect.onchange = render;
                        
                        domWidget.render = render;
                        
                        if (window.ResizeObserver) {
                            new ResizeObserver(() => requestAnimationFrame(render)).observe(canvas);
                        }
                    }
                    
                    canvas = domWidget.canvas;
                    gl = domWidget.gl;
                    
                    domWidget.imgWidth = data.width;
                    domWidget.imgHeight = data.height;
                    
                    // We remove the strict styling logic that forced height based on node width.
                    // WebGL u_scale handles letterboxing automatically!
                    
                    // Fetch Float32 Data
                    try {
                        const res = await fetch(url);
                        const buffer = await res.arrayBuffer();
                        const f32 = new Float32Array(buffer);
                        
                        domWidget.texData = f32;
                        
                        gl.bindTexture(gl.TEXTURE_2D, domWidget.tex);
                        let internalFormat = gl.RGBA;
                        if (gl instanceof WebGL2RenderingContext) {
                            internalFormat = gl.RGBA32F;
                        }
                        
                        gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, data.width, data.height, 0, gl.RGBA, gl.FLOAT, f32);

                        // Load Texture B if it exists
                        if (message.hdr_data.length > 1) {
                            const dataB = message.hdr_data[1];
                            const urlB = `/nodex_hdr/view?filename=${dataB.filename}`;
                            const resB = await fetch(urlB);
                            const bufferB = await resB.arrayBuffer();
                            const f32B = new Float32Array(bufferB);
                            
                            gl.bindTexture(gl.TEXTURE_2D, domWidget.texB);
                            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, dataB.width, dataB.height, 0, gl.RGBA, gl.FLOAT, f32B);
                            domWidget.hasTexB = true;
                            domWidget.abBtn.style.display = "block";
                            if (!domWidget.abOn) {
                                domWidget.abBtn.click(); // Auto enable A/B if B is connected
                            }
                        } else {
                            domWidget.hasTexB = false;
                            domWidget.abBtn.style.display = "none";
                            domWidget.abOn = false;
                        }
                        
                        domWidget.render();
                        
                    } catch(e) {
                        console.error("Failed to load HDR data:", e);
                    }
                }
            };
        }
    }
});
