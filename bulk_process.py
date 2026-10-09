"""
bulk_process.py  -  Command-line bulk pothole detector
Run your trained best.pt on a whole folder of images,
auto-extract GPS from EXIF, reverse-geocode road names,
and save every detection into potholes.db.

Usage
-----
  python bulk_process.py --images ./photos --conf 0.45
  python bulk_process.py --images ./photos --conf 0.45 --lat 18.5204 --lon 73.8567
  python bulk_process.py --images ./photos --conf 0.45 --address "MG Road, Pune"
"""

import os, sys, json, time, sqlite3, datetime, argparse
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from geopy.geocoders import Nominatim
    GEO_AVAILABLE = True
    _geo = Nominatim(user_agent="pothole_bulk_script")
except ImportError:
    GEO_AVAILABLE = False
    _geo = None

SUPPORTED_EXT = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}

def _dms_to_decimal(dms, ref):
    try:
        d, m, s = dms
        def f(v):
            if hasattr(v,"numerator"): return v.numerator/v.denominator if v.denominator else 0.0
            if isinstance(v,tuple): return v[0]/v[1] if v[1] else 0.0
            return float(v)
        dec = f(d) + f(m)/60 + f(s)/3600
        if ref in ("S","W"): dec = -dec
        return dec
    except: return None

def extract_gps(path):
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(path)
        exif_raw = img._getexif()
        if not exif_raw: return None, None
        exif = {TAGS.get(k,k):v for k,v in exif_raw.items()}
        gps_raw = exif.get("GPSInfo")
        if not gps_raw: return None, None
        gps = {GPSTAGS.get(k,k):v for k,v in gps_raw.items()}
        lat = _dms_to_decimal(gps.get("GPSLatitude"),  gps.get("GPSLatitudeRef","N"))
        lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef","E"))
        return lat, lon
    except: return None, None

def reverse_geocode(lat, lon):
    if not GEO_AVAILABLE: return "",""
    try:
        loc = _geo.reverse(f"{lat},{lon}", language="en", timeout=5)
        if not loc: return "",""
        raw = loc.raw.get("address",{})
        road = raw.get("road") or raw.get("pedestrian") or raw.get("street") or ""
        time.sleep(1.1)
        return loc.address, road
    except: return "",""

def severity(s):
    return "High" if s>=0.80 else ("Medium" if s>=0.60 else "Low")

def overall_severity(scores):
    return "None" if not scores else severity(float(np.mean(scores)))

def init_db(db_path):
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
        address TEXT, road_name TEXT, num_potholes INTEGER,
        avg_conf REAL, max_area_pct REAL, severity TEXT,
        status TEXT DEFAULT 'Reported', detections TEXT)""")
    for col,dfn in [("road_name","TEXT"),("status","TEXT DEFAULT 'Reported'")]:
        try: con.execute(f"ALTER TABLE reports ADD COLUMN {col} {dfn}")
        except: pass
    con.commit()
    return con

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images",      required=True)
    p.add_argument("--model",       default="best.pt")
    p.add_argument("--db",          default="potholes.db")
    p.add_argument("--conf",        type=float, default=0.45)
    p.add_argument("--lat",         type=float, default=None)
    p.add_argument("--lon",         type=float, default=None)
    p.add_argument("--address",     default="")
    p.add_argument("--skip-no-gps", action="store_true")
    a = p.parse_args()

    if not os.path.isdir(a.images): sys.exit(f"Folder not found: {a.images}")
    if not os.path.isfile(a.model): sys.exit(f"Model not found: {a.model}")

    print(f"Loading model: {a.model}")
    model = YOLO(a.model)
    class_names = model.names
    con = init_db(a.db)

    files = [os.path.join(a.images,fn) for fn in sorted(os.listdir(a.images))
             if os.path.splitext(fn)[1].lower() in SUPPORTED_EXT]
    if not files: sys.exit(f"No images in {a.images}")

    print(f"\nFound {len(files)} image(s)\n")
    print(f"{'#':>4}  {'File':<36} {'GPS':>8}  {'Count':>5}  Severity  Road")
    print("-"*80)
    saved=no_ph=skip=err=0

    for idx,fpath in enumerate(files,1):
        fname = os.path.basename(fpath)
        try:
            lat, lon = extract_gps(fpath)
            gps_src  = "EXIF"
            if lat is None:
                if a.skip_no_gps: skip+=1; print(f"{idx:>4}  {fname:<36} {'skipped':>8}"); continue
                if a.lat is None: skip+=1; print(f"{idx:>4}  {fname:<36} {'no-GPS':>8}  (set --lat --lon)"); continue
                lat, lon, gps_src = a.lat, a.lon, "fallback"

            img_bgr = cv2.imread(fpath)
            if img_bgr is None: raise ValueError("Could not decode")
            h,w = img_bgr.shape[:2]

            res = model.predict(img_bgr, conf=a.conf, verbose=False)
            dets=[]
            for b in res[0].boxes:
                s = b.conf[0].item()
                if s < a.conf: continue
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                dets.append({"class":class_names[int(b.cls[0].item())],
                             "conf":round(s,3),"severity":severity(s),
                             "bbox":[round(v,1) for v in [x1,y1,x2,y2]],
                             "area_px":round((x2-x1)*(y2-y1)),
                             "area_pct":round((x2-x1)*(y2-y1)/(h*w)*100,3)})
            if not dets:
                no_ph+=1; print(f"{idx:>4}  {fname:<36} {gps_src:>8}  {0:>5}"); continue

            addr,road = ("","")
            if gps_src=="EXIF": addr,road = reverse_geocode(lat,lon)
            elif a.address: road=a.address; addr=a.address

            confs=[d["conf"] for d in dets]; areas=[d["area_pct"] for d in dets]
            sev=overall_severity(confs)
            con.execute("""INSERT INTO reports
                (ts,lat,lon,address,road_name,num_potholes,avg_conf,max_area_pct,severity,status,detections)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.datetime.now().isoformat("T","seconds"),lat,lon,addr,road,len(dets),
                 round(float(np.mean(confs)),4),round(float(max(areas)),4),sev,"Reported",json.dumps(dets)))
            con.commit(); saved+=1
            print(f"{idx:>4}  {fname:<36} {gps_src:>8}  {len(dets):>5}  {sev:<8}  {(road or addr[:20] or '-')[:22]}")
        except Exception as ex:
            err+=1; print(f"{idx:>4}  {fname:<36} ERROR  {str(ex)[:40]}")

    con.close()
    print("\n"+"="*80)
    print(f"DONE  Saved:{saved}  NoPotholes:{no_ph}  Skipped:{skip}  Errors:{err}")
    print(f"DB: {os.path.abspath(a.db)}")

if __name__=="__main__":
    main()
