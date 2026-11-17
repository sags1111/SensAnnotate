/* Streamlit components v1 message protocol; drawing is local until Save. */
(() => {
  "use strict";
  const get = id => document.getElementById(id);
  const svg = get("surface"), tool = get("tool"), sensitivity = get("sensitivity");
  const maxPoints = 1000;
  let args = null, points = [], start = null, hover = null, pointer = null;
  let dragging = false, disabled = false, lastHeight = 0;
  let strokeClosed = false, strokeLength = 0;
  let initialized = false;

  function send(type, data = {}) {
    window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
  }
  function fitFrame() {
    const height = Math.ceil(get("root").getBoundingClientRect().height) + 4;
    if (height !== lastHeight) {
      lastHeight = height;
      send("streamlit:setFrameHeight", {height});
    }
  }
  function status(message) { get("status").textContent = message; fitFrame(); }
  function point(event) {
    const transform = svg.getScreenCTM();
    if (!transform) return null;
    const p = new DOMPoint(event.clientX, event.clientY).matrixTransform(transform.inverse());
    return [Math.max(0, Math.min(args.image_width, p.x)),
            Math.max(0, Math.min(args.image_height, p.y))];
  }
  function distance(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  function screenDistance(a, b) {
    const matrix = svg.getScreenCTM();
    const dx = a[0] - b[0], dy = a[1] - b[1];
    return matrix ? Math.hypot(matrix.a * dx + matrix.c * dy,
                              matrix.b * dx + matrix.d * dy) : distance(a, b);
  }
  function area() {
    return Math.abs(points.reduce((sum, p, i) => {
      const next = points[(i + 1) % points.length];
      return sum + p[0] * next[1] - next[0] * p[1];
    }, 0)) / 2;
  }
  function paint() {
    if (!args) return;
    const color = sensitivity.value === "auto" ? "#4aa3ff" :
      sensitivity.value === "non_sensitive" ? args.non_sensitive_color : args.sensitive_color;
    const value = points.map(p => p.join(",")).join(" ");
    for (const id of ["fill", "line", "guide"]) get(id).setAttribute("stroke", color);
    get("fill").setAttribute("fill", color);
    get("fill").setAttribute("points", points.length >= 3 ? value : "");
    get("line").setAttribute("points", value);
    get("guide").setAttribute("points", tool.value === "polygon" && hover && points.length
      ? [points[points.length - 1], hover, points[0]].map(p => p.join(",")).join(" ") : "");
    get("vertices").replaceChildren();
    if (tool.value === "polygon") {
      for (const p of points) {
        const circle = document.createElementNS(svg.namespaceURI, "circle");
        circle.setAttribute("cx", p[0]); circle.setAttribute("cy", p[1]);
        circle.setAttribute("r", 4); circle.setAttribute("fill", color);
        circle.setAttribute("stroke", "white"); circle.setAttribute("stroke-width", 1);
        get("vertices").appendChild(circle);
      }
    }
    get("save").disabled = disabled || dragging || points.length < 3 || area() < 1;
    get("undo").disabled = disabled || !points.length || dragging;
    get("clear").disabled = disabled || !points.length;
    get("undo").textContent = tool.value === "rectangle" ? "Undo shape" : "Undo point";
  }
  function cancelPointer() {
    const activePointer = pointer;
    pointer = null; dragging = false; start = null;
    if (activePointer !== null && svg.hasPointerCapture(activePointer)) svg.releasePointerCapture(activePointer);
  }
  function finishStroke(message) {
    cancelPointer(); paint(); status(message);
  }
  function clear() {
    cancelPointer(); points = []; hover = null; strokeClosed = false; strokeLength = 0; paint();
    status("Draft cleared. Draw a new outline.");
  }
  function help() {
    get("help").textContent = {
      polygon: "Click around the boundary to add points. The outline closes when you save. Undo removes the last point.",
      freehand: "Hold and drag around the boundary. Return near the start to close the loop, or release and Save to close it. Avoid crossing the outline.",
      rectangle: "Drag between opposite corners. Check the live outline, then save."
    }[tool.value];
    fitFrame();
  }
  function rectangle(end) {
    const x1 = Math.min(start[0], end[0]), x2 = Math.max(start[0], end[0]);
    const y1 = Math.min(start[1], end[1]), y2 = Math.max(start[1], end[1]);
    points = [[x1,y1], [x2,y1], [x2,y2], [x1,y2]];
  }
  function appendFreehand(p) {
    if (strokeClosed || !points.length) return;
    // Finish a natural lasso once; movement after returning to its start must
    // not create a tail that intersects the already completed polygon.
    if (points.length >= 4 && strokeLength >= 32 && screenDistance(points[0], p) <= 8) {
      points.push([...points[0]]);
      strokeClosed = true;
      status("Loop closed. Release, check the outline, and Save region.");
      return;
    }
    const step = screenDistance(points[points.length - 1], p);
    if (step < 2) return;
    if (points.length >= maxPoints - 1) {
      status("Point limit reached. Release and save, or clear and draw a shorter outline.");
      return;
    }
    strokeLength += step;
    points.push(p);
  }
  svg.addEventListener("pointerdown", event => {
    if (!args || disabled || event.button !== 0 || pointer !== null) return;
    event.preventDefault(); svg.focus({preventScroll: true});
    const p = point(event);
    if (!p) return;
    if (tool.value === "polygon") {
      if (points.length >= maxPoints) { status("Point limit reached. Save or clear the outline."); return; }
      if (!points.length || distance(points[points.length - 1], p) > 1) points.push(p);
      hover = null;
      status(`${points.length} points. Add more points or save the region.`);
    } else {
      start = p; points = [p]; dragging = true; pointer = event.pointerId;
      strokeClosed = false; strokeLength = 0;
      svg.setPointerCapture(pointer);
      status("Drawing — the shaded area is your current outline.");
    }
    paint();
  });
  window.addEventListener("pointermove", event => {
    if (!args || disabled) return;
    if (!dragging && !svg.contains(event.target)) return;
    const p = point(event);
    if (!p) return;
    if (dragging && event.pointerId === pointer) {
      // Recover if release happened outside the iframe and pointerup was lost.
      if (event.buttons === 0) {
        finishStroke("Stroke ended. Your draft is kept; check it and Save region.");
        return;
      }
      if (tool.value === "rectangle") rectangle(p);
      else appendFreehand(p);
    } else if (tool.value === "polygon") hover = p;
    paint();
  });
  window.addEventListener("pointerup", event => {
    if (!dragging || event.pointerId !== pointer) return;
    const p = point(event);
    if (p) {
      if (tool.value === "rectangle") rectangle(p);
      else appendFreehand(p);
    }
    finishStroke("Check the outline, then Save region. Clear starts again.");
  });
  svg.addEventListener("pointercancel", event => {
    if (dragging && event.pointerId === pointer)
      finishStroke("Drawing interrupted. Your draft is kept; check it, undo, or clear it.");
  });
  svg.addEventListener("lostpointercapture", event => {
    // Losing SVG capture does not necessarily mean the button was released.
    // Continue via window events while held; otherwise retain the draft.
    if (dragging && event.pointerId === pointer && event.buttons === 0)
      finishStroke("Stroke ended. Your draft is kept; check it and Save region.");
  });
  window.addEventListener("blur", () => {
    if (dragging) finishStroke("Drawing paused. Your draft is kept; check it before saving.");
  });
  svg.addEventListener("dragstart", event => event.preventDefault());
  svg.addEventListener("pointerleave", () => { hover = null; paint(); });
  get("undo").addEventListener("click", () => {
    if (tool.value === "rectangle") points = [];
    else points.pop();
    strokeClosed = false;
    hover = null; paint(); status("Last point or shape removed.");
  });
  get("clear").addEventListener("click", clear);
  tool.addEventListener("change", () => { clear(); help(); });
  sensitivity.addEventListener("change", paint);
  get("save").addEventListener("click", () => {
    if (get("save").disabled) return;
    send("streamlit:setComponentValue", {dataType: "json", value: {
      event_id: [Date.now(), ...crypto.getRandomValues(new Uint32Array(4))].join("-"),
      tool: tool.value, sensitivity: sensitivity.value,
      points: points.map(([x,y]) => [x / args.image_width, y / args.image_height])
    }});
    status("Submitted. If the app reports an invalid outline, undo or clear it and save again.");
  });
  svg.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.preventDefault(); clear(); }
    if (event.key === "Backspace" && !get("undo").disabled) {
      event.preventDefault(); get("undo").click();
    }
    if (event.key === "Enter" && !get("save").disabled) get("save").click();
  });
  window.addEventListener("message", event => {
    if (event.source !== window.parent || !event.data || event.data.type !== "streamlit:render") return;
    args = event.data.args; disabled = Boolean(event.data.disabled);
    // Initialize a new canvas in the requested mode. Later renders must
    // not overwrite the user's selection for an unfinished drawing.
    if (!initialized) {
      sensitivity.value = ["sensitive", "non_sensitive"].includes(args.initial_sensitivity)
        ? args.initial_sensitivity : "auto";
      initialized = true;
    }
    const theme = event.data.theme || {};
    document.body.style.setProperty("--text", args.text_color || theme.textColor || "#e8eaf0");
    document.body.style.setProperty("--surface", args.surface_color || theme.secondaryBackgroundColor || "#1e2130");
    document.body.style.setProperty("--background", args.background_color || theme.backgroundColor || "#0f1117");
    tool.disabled = disabled; sensitivity.disabled = disabled;
    svg.setAttribute("viewBox", `0 0 ${args.image_width} ${args.image_height}`);
    get("image").setAttribute("width", args.image_width);
    get("image").setAttribute("height", args.image_height);
    get("image").setAttribute("href", args.image_url);
    help(); paint(); fitFrame();
  });
  new ResizeObserver(fitFrame).observe(get("root"));
  send("streamlit:componentReady", {apiVersion: 1});
  fitFrame();
})();
