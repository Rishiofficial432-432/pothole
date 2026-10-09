"""Wipes ALL saved reports, uploaded photos and sample images so you can start a clean live demo.
Run once:  python reset_demo.py"""
import os, shutil, sqlite3
BASE = os.path.dirname(os.path.abspath(__file__))
if input("Delete ALL reports, uploads and sample_images? Type yes: ").strip().lower() != "yes":
    raise SystemExit("Cancelled.")
db = os.path.join(BASE, "potholes.db")
if os.path.exists(db):
    con = sqlite3.connect(db)
    con.execute("DROP TABLE IF EXISTS reports")
    con.commit()
    con.execute("VACUUM")
    con.close()
for d in ("uploads", "sample_images"):
    p = os.path.join(BASE, d)
    if os.path.isdir(p):
        shutil.rmtree(p)
os.makedirs(os.path.join(BASE, "uploads"), exist_ok=True)
print("Clean. Reports, uploads and sample images removed.")
