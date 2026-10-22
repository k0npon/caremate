"""CareMate perception service (AWS): OpenCV 5 target verification over HTTP.

The robot keeps low-level control local and sends each fresh head-camera
frame here; the service answers "is <target> in this image, where, and how
sure (status, not probability)?".  Same code path as the local backend
(caremate.perception_v0_7.PerceptionBackend), so results are identical.

Endpoints
  GET  /                  small web page: upload an image, pick a target, see the overlay
  GET  /v1/health         liveness + model/version info
  GET  /v1/references     names of objects this service can verify
  POST /v1/perceive       multipart: image (PNG/JPEG), target, optional roi_polygon (JSON
                          list of [x,y]), optional camera_pose (JSON), overlay=0/1
                          -> PerceptionResult JSON (+ base64 PNG overlay if overlay=1)
  POST /v1/references     (staff, admin token) enrol a new object from one photo of a printed
                          face: name, face, image, optional bbox [x0,y0,x1,y1], dry_run=0/1.
                          Acceptance checks in caremate.enrol_v0_7 (texture, self-match under a
                          tilted smaller view, look-alike report).  Takes effect immediately for
                          every robot using this service.
  DELETE /v1/references/{name}   remove an enrolled (not built-in) object
  GET  /v1/samples        demo head-camera frames from the simulation (for judges / visitors)
  GET  /samples/{id}.png  one sample frame (raw, so it can also be downloaded and uploaded)
  POST /v1/samples/{id}/perceive   run /v1/perceive on a sample with its stored ROI (no upload)

Security / responsible delivery
  * If CAREMATE_API_TOKEN is set, /v1/perceive requires "Authorization: Bearer <token>".
  * Robot camera frames are processed in memory and never stored; only one JSON log line per
    request (target, status, latency, image hash) goes to stdout -> CloudWatch/docker logs.
  * Enrolment is disabled unless CAREMATE_ADMIN_TOKEN is set; enrolled reference photos (product
    packaging, not people) are kept in CAREMATE_REF_STORE so they survive restarts.
  * Upload size and image size are capped; requests are serialised (one model instance).

Run locally:  uvicorn caremate.service.app:app --host 0.0.0.0 --port 8000
Env: CAREMATE_ALIKED (model path), CAREMATE_REFS (reference dir), CAREMATE_API_TOKEN,
     CAREMATE_STANDIN_SIFT=1 (tests only; NOT OpenCV 5 ALIKED).
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from caremate import enrol_v0_7 as enrol
from caremate.perception_v0_7 import PerceptionBackend, draw

ROOT = Path(__file__).resolve().parents[2]
ALIKED = Path(os.environ.get("CAREMATE_ALIKED", ROOT / "models/vision/aliked-n16rot-top1k-640.onnx"))
REFS = Path(os.environ.get("CAREMATE_REFS", ROOT / "references/v0_7"))
TOKEN = os.environ.get("CAREMATE_API_TOKEN", "")
ADMIN_TOKEN = os.environ.get("CAREMATE_ADMIN_TOKEN", "")
REF_STORE = Path(os.environ.get("CAREMATE_REF_STORE", "/data/references"))
SAMPLES = Path(os.environ.get("CAREMATE_SAMPLES", ROOT / "samples/v0_7"))
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 2048

app = FastAPI(title="CareMate perception service", version="0.7")
_lock = threading.Lock()
_backend: PerceptionBackend | None = None
_info: dict = {}
_samples: dict = {}      # id -> metadata (samples.json); images are read from SAMPLES on demand


def backend() -> PerceptionBackend:
    global _backend
    if _backend is None:
        ext = None
        if os.environ.get("CAREMATE_STANDIN_SIFT") == "1":
            from caremate.agent_loop_v0_7 import SiftStandIn
            ext = SiftStandIn()
        t0 = time.perf_counter()
        _backend = PerceptionBackend(ALIKED, REFS, extractor=ext)
        _info["builtin"] = sorted(_backend.refs)
        if REF_STORE.is_dir():            # previously enrolled objects
            for f in sorted(REF_STORE.glob("*.png")):
                name, _, face = f.stem.partition("__")
                im = cv2.imread(str(f), cv2.IMREAD_COLOR)
                if im is not None and enrol.NAME_RE.match(name):
                    _backend.add_reference(name, face or "front", im)
        warm = _backend.warmup()
        _info.update(method=_backend.method, opencv=cv2.__version__,
                     references=sorted(_backend.refs), load_s=round(time.perf_counter() - t0, 2),
                     warmup_s=round(warm, 2), cpu_count=os.cpu_count())
    return _backend


def _load_samples():
    f = SAMPLES / "samples.json"
    if f.is_file():
        for s in json.loads(f.read_text()):
            if (SAMPLES / f"{s['id']}.png").is_file():
                _samples[s["id"]] = s
    _info["samples"] = sorted(_samples)


@app.on_event("startup")
def _startup():
    with _lock:
        backend()
    _load_samples()


def _log(**kw):
    print(json.dumps({"ts": round(time.time(), 3), **kw}), flush=True)


@app.get("/v1/health")
def health():
    return {"status": "ok", **_info}


@app.get("/v1/references")
def references():
    be = backend()
    return {"references": {n: sorted(f) for n, f in be.refs.items()},
            "builtin": _info.get("builtin", []),
            "enrolled": sorted(set(be.refs) - set(_info.get("builtin", []))),
            "enrolment_enabled": bool(ADMIN_TOKEN)}


def _admin(authorization: str | None):
    if not ADMIN_TOKEN:
        raise HTTPException(403, "enrolment disabled on this deployment (CAREMATE_ADMIN_TOKEN not set)")
    if authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "admin token required")


@app.post("/v1/references")
async def enrol_reference(image: UploadFile = File(...), name: str = Form(...), face: str = Form("front"),
                          bbox: str | None = Form(None), dry_run: int = Form(0),
                          authorization: str | None = Header(None)):
    if not dry_run:                   # checking only (nothing stored) is open; storing needs staff token
        _admin(authorization)
    name, face = name.strip().lower(), face.strip().lower()
    if not enrol.NAME_RE.match(name) or not enrol.FACE_RE.match(face):
        raise HTTPException(400, "name: lowercase letters/digits/_ (2-41 chars); face: 1-16 letters/digits")
    if name in _info.get("builtin", []):
        raise HTTPException(409, f"'{name}' is a built-in reference")
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 8 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    try:
        box = json.loads(bbox) if bbox else None
        if box is not None and len(box) != 4:
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(400, "bbox must be JSON [x0, y0, x1, y1]")
    ref = enrol.normalise(img, box)
    be = backend()
    with _lock:
        rep = enrol.check(be, name, face, ref)
        if rep["accepted"] and not dry_run:
            REF_STORE.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(REF_STORE / f"{name}__{face}.png"), ref)
            be.add_reference(name, face, ref)
            be.warmup()
            rep["stored"] = True
    _log(event="enrol", name=name, face=face, accepted=rep["accepted"], dry_run=bool(dry_run),
         texture=rep.get("texture_score"), self_match=rep.get("self_match_inliers"))
    ok, png = cv2.imencode(".png", ref)
    rep["reference_png_base64"] = base64.b64encode(png.tobytes()).decode() if ok else None
    return JSONResponse(rep, status_code=200 if rep["accepted"] else 422)


@app.delete("/v1/references/{name}")
def delete_reference(name: str, authorization: str | None = Header(None)):
    _admin(authorization)
    if name in _info.get("builtin", []):
        raise HTTPException(409, "built-in references cannot be deleted")
    be = backend()
    if name not in be.refs:
        raise HTTPException(404, "unknown reference")
    with _lock:
        be.remove_reference(name)
        for f in REF_STORE.glob(f"{name}__*.png"):
            f.unlink()
    _log(event="delete_reference", name=name)
    return {"deleted": name}


@app.post("/v1/perceive")
async def perceive(image: UploadFile = File(...), target: str = Form(...),
                   roi_polygon: str | None = Form(None), camera_pose: str | None = Form(None),
                   overlay: int = Form(0), origin: str | None = Form(None),
                   frame_wh: str | None = Form(None), authorization: str | None = Header(None)):
    """origin/frame_wh (optional JSON [x, y] / [w, h]): the image is a crop of a larger
    frame starting at origin; results are returned in full-frame coordinates."""
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "missing or wrong bearer token")
    t_recv = time.perf_counter()
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 8 MB")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "could not decode image")
    if max(bgr.shape[:2]) > MAX_SIDE:
        raise HTTPException(400, f"image side > {MAX_SIDE}px")
    try:
        roi = np.asarray(json.loads(roi_polygon), np.float32) if roi_polygon else None
        pose = json.loads(camera_pose) if camera_pose else None
        org = [int(v) for v in json.loads(origin)] if origin else [0, 0]
        fwh = [int(v) for v in json.loads(frame_wh)] if frame_wh else None
        if len(org) != 2 or (fwh is not None and len(fwh) != 2):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(400, "roi_polygon / camera_pose / origin / frame_wh must be JSON")
    if roi is not None and (roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3):
        raise HTTPException(400, "roi_polygon must be a list of >= 3 [x, y] points")
    return JSONResponse(_run(bgr, target, roi, pose, org, fwh, overlay, t_recv, len(data)))


def _explain(res) -> str:
    """One plain sentence for the web page (same rules as the robot's decision)."""
    unresolved = [n for n, c in res.competitors.items() if c.get("resolution") == "unresolved"]
    rejected = sorted({n for r in res.rejected_instances for n in r["is_actually"]} |
                      {n for n, c in res.competitors.items() if c.get("resolution") == "region_is_competitor"})
    if res.status == "candidate" and not res.touches_border and not unresolved:
        return f"{res.n_inliers} geometric inliers, fully inside the frame, no unresolved look-alike: the robot may stop here."
    if res.status == "candidate":
        return "Target touches the image border (may be cut off): the robot looks again."
    if res.status == "ambiguous":
        return "; ".join(res.notes) if res.notes else f"Look-alike(s) {unresolved} explain the same region: the robot needs a clearer view."
    if res.status == "not_found" and rejected:
        return f"The matching region is actually {', '.join(rejected)} (a look-alike), so it is rejected: the robot keeps searching."
    if res.status == "error":
        return f"Perception error ({res.error}): the robot does not assume anything."
    return "No geometrically consistent evidence for this object: the robot looks elsewhere, then asks for help."


def _run(bgr, target, roi, pose, org, fwh, overlay, t_recv, nbytes, source="upload") -> dict:
    be = backend()
    if target not in be.refs:
        raise HTTPException(404, f"unknown target; known: {sorted(be.refs)}")
    t0 = time.perf_counter()
    with _lock:                       # one model instance; keep requests serialised
        res = be.perceive(bgr, target, camera_pose=pose, roi_polygon=roi, origin=org, frame_wh=fwh)
    t_inf = time.perf_counter() - t0
    out = res.to_json()
    out["explanation"] = _explain(res)
    out["server"] = {"decode_ms": round(1000 * (t0 - t_recv), 1),
                     "inference_ms": round(1000 * t_inf, 1),
                     "image_wh": [bgr.shape[1], bgr.shape[0]], "bytes": nbytes}
    if overlay:
        ok, png = cv2.imencode(".png", draw(bgr, res, None, roi))
        out["overlay_png_base64"] = base64.b64encode(png.tobytes()).decode() if ok else None
    _log(event="perceive", source=source, target=target, status=res.status, inliers=res.n_inliers,
         image_id=res.image_id, inference_ms=out["server"]["inference_ms"], bytes=nbytes)
    return out


@app.get("/v1/samples")
def samples():
    return {"samples": [{k: s[k] for k in ("id", "title", "caption", "target", "view", "expected")}
                        for s in _samples.values()]}


def _sample(sid: str) -> dict:
    if sid not in _samples:           # only ids from samples.json: no path traversal
        raise HTTPException(404, "unknown sample")
    return _samples[sid]


@app.get("/samples/{sid}.png")
def sample_image(sid: str):
    _sample(sid)
    return FileResponse(SAMPLES / f"{sid}.png", media_type="image/png")


@app.post("/v1/samples/{sid}/perceive")
def sample_perceive(sid: str, authorization: str | None = Header(None)):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "missing or wrong bearer token")
    s = _sample(sid)
    t_recv = time.perf_counter()
    bgr = cv2.imread(str(SAMPLES / f"{sid}.png"), cv2.IMREAD_COLOR)
    roi = np.asarray(s["roi_polygon"], np.float32) if s.get("roi_polygon") else None
    out = _run(bgr, s["target"], roi, None, [0, 0], None, 1, t_recv, 0, source=f"sample:{sid}")
    out["sample"] = {k: s[k] for k in ("id", "title", "caption", "view", "expected")}
    return JSONResponse(out)


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>CareMate perception</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;
background:#fafafa;color:#222}img{max-width:100%;border:1px solid #ccc}pre{background:#eee;
padding:8px;overflow:auto;font-size:12px}label{display:block;margin:8px 0}</style></head><body>
<h2>CareMate &mdash; OpenCV 5 target verification on AWS</h2>
<p>A care robot is asked to fetch an item. Before it acts, this service checks the robot's camera image with
OpenCV 5 (ALIKED features + USAC/MAGSAC homography + a colour check against look-alikes) and answers
<b>candidate / ambiguous / not_found</b>. It never claims a calibrated probability and never authorises
grasping. Images are not stored.</p>
<h3>Try it: frames from the robot's head camera (simulation)</h3>
<p>Each button sends one simulated head-camera frame to OpenCV 5 running on this AWS instance.
Green = accepted target, orange = rejected look-alike; the outlined polygon is the workspace region searched.</p>
<div id=sb></div><p id=ss style="font-weight:600"></p><p id=sx></p><img id=so><pre id=sj></pre>
<hr><h3>Upload your own frame</h3>
<p>Choose the object to verify and upload an image (for example, one of the sample frames above:
<span id=sl></span>).</p>
<form id=f><label>Target <select name=target id=t></select></label>
<label>Image <input type=file name=image accept="image/*" required></label>
<label>API token (if required) <input id=k type=password></label>
<button>Verify</button></form><p id=s></p><img id=o><pre id=j></pre>
<hr><h3>Staff: enrol a new object</h3>
<p>One clear, front-on photo of a <b>printed, flat</b> face (box, carton, book). Plain mugs or bottles are
rejected: they have too little texture for reliable verification. A new object is usable by every robot
immediately, with no retraining.</p>
<p><b>Anyone can try the acceptance check</b> with a photo of a box on their desk (tick "only check": the photo
is checked in memory and not stored). Storing a new object needs the staff token.</p>
<form id=e><label>Name (e.g. blister_pack_a) <input name=name required pattern="[a-z][a-z0-9_]{1,40}"></label>
<label>Face <input name=face value=front></label>
<label>Photo <input type=file name=image accept="image/*" required></label>
<label>Admin token (only to store) <input id=ak type=password></label>
<label><input type=checkbox id=dry checked> only check, do not store</label>
<button>Enrol</button></form><p id=es></p><img id=eo style="max-height:240px"><pre id=ej></pre>
<script>
const show=(d,ok,st,ex,im,js)=>{if(d.overlay_png_base64){im.src='data:image/png;base64,'+d.overlay_png_base64;delete d.overlay_png_base64}
 st.textContent=ok?(d.target+': '+d.status.toUpperCase()+' ('+d.n_inliers+' inliers, '+d.server.inference_ms+' ms on the server)'):'error';
 if(ex)ex.textContent=d.explanation||'';js.textContent=JSON.stringify(d,null,1)};
fetch('/v1/samples').then(r=>r.json()).then(d=>{for(const s of d.samples){
 const b=document.createElement('button');b.textContent=s.title;b.style.cssText='display:block;margin:6px 0;text-align:left';
 b.onclick=async()=>{ss.textContent='running OpenCV 5 on AWS...';sx.textContent=s.caption;so.removeAttribute('src');
  const r=await fetch('/v1/samples/'+s.id+'/perceive',{method:'POST'});const d=await r.json();
  show(d,r.ok,ss,null,so,sj);if(r.ok)sx.textContent=s.caption+' Result: '+d.explanation};sb.appendChild(b);
 const a=document.createElement('a');a.href='/samples/'+s.id+'.png';a.textContent=s.id;a.download='';sl.append(a,' ')}});
fetch('/v1/references').then(r=>r.json()).then(d=>{for(const n of Object.keys(d.references)){
 const o=document.createElement('option');o.textContent=n;t.appendChild(o)}});
e.onsubmit=async ev=>{ev.preventDefault();const fd=new FormData(e);fd.append('dry_run',dry.checked?'1':'0');
 es.textContent='checking...';const eh={};if(ak.value)eh.Authorization='Bearer '+ak.value;
 const r=await fetch('/v1/references',{method:'POST',body:fd,headers:eh});const d=await r.json();
 if(d.reference_png_base64){eo.src='data:image/png;base64,'+d.reference_png_base64;delete d.reference_png_base64}
 es.textContent=r.status==200?('ACCEPTED: '+d.name+' ('+d.note+')'):('REJECTED ('+r.status+'): '+(d.reason||d.detail));
 ej.textContent=JSON.stringify(d,null,1);if(r.status==200&&!dry.checked)location.reload()};
f.onsubmit=async e=>{e.preventDefault();const fd=new FormData(f);fd.append('overlay','1');
 s.textContent='running...';const h={};if(k.value)h.Authorization='Bearer '+k.value;
 const r=await fetch('/v1/perceive',{method:'POST',body:fd,headers:h});const d=await r.json();
 show(d,r.ok,s,null,o,j);if(r.ok)s.textContent+=' \u2014 '+d.explanation};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE
