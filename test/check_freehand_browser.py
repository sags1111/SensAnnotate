"""Chrome regression checks against the actual local drawing component.

Run from any directory with Python, Pillow, websocket-client, and Chrome installed:
    python test/check_freehand_browser.py
No models or Streamlit server are started. HTTP and DevTools bind to loopback.
"""
import base64
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
import threading
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import urllib.request
import websocket
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parents[1]
chrome = shutil.which('google-chrome') or shutil.which('chromium')
if chrome is None:
    raise SystemExit('Chrome or Chromium is required for the browser check.')
sys.path.insert(0, str(root))
from manual_drawing import make_manual_region

with tempfile.TemporaryDirectory(prefix='annotation-browser-') as directory:
    profile = Path(directory)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(root / 'drawing_canvas')))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log = (profile / 'chrome.log').open('w')
    process = subprocess.Popen([chrome, '--headless', '--no-sandbox', '--disable-dev-shm-usage', '--no-first-run', '--remote-debugging-port=0', f'--user-data-dir={profile}', '--window-size=1000,900', 'about:blank'], stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 15
        while not (profile / 'DevToolsActivePort').exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError((profile / 'chrome.log').read_text()[-2000:])
            time.sleep(.1)
        port = (profile / 'DevToolsActivePort').read_text().splitlines()[0]
        pages = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json'))
        connection = websocket.create_connection(next(page for page in pages if page['type'] == 'page')['webSocketDebuggerUrl'], suppress_origin=True, timeout=10)
        sequence = 0
        def call(method, params=None):
            global sequence
            sequence += 1
            connection.send(json.dumps({'id': sequence, 'method': method, 'params': params or {}}))
            while True:
                data = json.loads(connection.recv())
                if data.get('id') == sequence:
                    if 'error' in data: raise RuntimeError(data['error'])
                    return data.get('result', {})
        def js(expression):
            result = call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
            if 'exceptionDetails' in result: raise RuntimeError(result['exceptionDetails'])
            return result.get('result', {}).get('value')
        call('Page.enable')
        call('Page.navigate', {'url': f'http://127.0.0.1:{server.server_port}/index.html'})
        for _ in range(50):
            if js('document.readyState === "complete" && document.getElementById("surface") !== null'): break
            time.sleep(.05)
        image = Image.new('RGB', (900, 600), '#394957')
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 100, 780, 500), fill='#596b79')
        draw.text((30, 30), 'Live drawing component check', fill='white')
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        args = dict(image_url='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), image_width=900, image_height=600, sensitive_color='#ff8000', non_sensitive_color='#2ec4a6')
        js('window.saved=[];window.errors=[];window.addEventListener("error",e=>errors.push(e.message));window.addEventListener("message",e=>{if(e.data.type==="streamlit:setComponentValue")saved.push(e.data.value)});')
        js('window.postMessage(' + json.dumps({'type':'streamlit:render', 'args':args}) + ',"*")')
        js('new Promise(resolve=>setTimeout(resolve,100))')
        def mouse(kind, x, y, buttons=0):
            screen = js(f'(()=>{{const p=new DOMPoint({x},{y}).matrixTransform(document.getElementById("surface").getScreenCTM());return {{x:p.x,y:p.y}}}})()')
            call('Input.dispatchMouseEvent', dict(type=kind, x=screen['x'], y=screen['y'], button='left' if kind != 'mouseMoved' else 'none', buttons=buttons, clickCount=1))
            js('new Promise(resolve=>requestAnimationFrame(()=>resolve()))')
        def click(x,y):
            mouse('mousePressed',x,y,1); mouse('mouseReleased',x,y)
        def select(id, value):
            js(f'document.getElementById({json.dumps(id)}).value={json.dumps(value)};document.getElementById({json.dumps(id)}).dispatchEvent(new Event("change"));')
        def save():
            js('document.getElementById("save").click()')
            js('new Promise(resolve=>setTimeout(resolve,30))')
            return js('saved[saved.length-1]')
        failures = []
        assert js('document.getElementById("sensitivity").value') == 'auto'
        select('tool', 'freehand')
        select('sensitivity', 'non_sensitive')
        def stroke():
            mouse('mousePressed',180,120,1)
            for x,y in [(400,90),(700,150),(750,380),(420,500),(140,370),(160,180),(180,120)]:
                mouse('mouseMoved',x,y,1)
        stroke()
        # Natural completion: user reaches the start, then moves a little before releasing.
        mouse('mouseMoved',200,130,1)
        mouse('mouseReleased',200,130)
        try:
            result=save()
            make_manual_region(result,1800,1200)
            print('PASS natural closed freehand saves')
        except (ValueError,TypeError) as exc:
            failures.append('closed stroke: '+str(exc))
        js('document.getElementById("clear").click()')
        mouse('mousePressed',180,120,1)
        for x,y in [(500,100),(720,300),(450,450)]: mouse('mouseMoved',x,y,1)
        js('document.getElementById("surface").dispatchEvent(new PointerEvent("pointercancel",{pointerId:1}))')
        if not js('document.getElementById("fill").getAttribute("points")'):
            failures.append('pointer cancellation discards the whole visible draft')
        elif js('document.getElementById("save").disabled'):
            failures.append('pointer cancellation leaves Save disabled')
        else:
            make_manual_region(save(),1800,1200)
            print('PASS cancelled stroke remains editable and saveable')
        mouse('mouseReleased',450,450)
        js('document.getElementById("clear").click()')
        mouse('mousePressed',180,120,1)
        for x,y in [(500,100),(720,300),(450,450)]: mouse('mouseMoved',x,y,1)
        js('document.getElementById("surface").releasePointerCapture(1)')
        mouse('mouseReleased',450,450)
        if js('document.getElementById("save").disabled'):
            failures.append('lost pointer capture leaves drawing stuck and Save disabled')
        else:
            make_manual_region(save(),1800,1200)
            print('PASS lost capture does not lock drawing')
        mouse('mouseReleased',450,450)
        assert js('errors') == []
        print('FAILURES:',failures)
        assert not failures, failures
        connection.close()
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
        server.shutdown(); server.server_close()
