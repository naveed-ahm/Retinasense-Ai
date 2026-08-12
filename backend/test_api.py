import urllib.request, json

BASE = "http://127.0.0.1:8000/api"

# 1. Health
r = urllib.request.urlopen(f"{BASE}/health", timeout=5)
print("1. Health:", json.loads(r.read()))

# 2. Login
req = urllib.request.Request(
    f"{BASE}/auth/login",
    data=json.dumps({"email": "dr.rajan@apollo.org", "password": "demo1234"}).encode(),
    headers={"Content-Type": "application/json"},
)
r = urllib.request.urlopen(req, timeout=5)
login = json.loads(r.read())
token = login["token"]
print(f'2. Login: {login["user"]["name"]} ({login["user"]["email"]})')

hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# 3. Dashboard
req = urllib.request.Request(f"{BASE}/dashboard/stats", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
dash = json.loads(r.read())
print(f'3. Dashboard: {dash["kpis"][0]["title"]}={dash["kpis"][0]["value"]}')

# 4. Patients list
req = urllib.request.Request(f"{BASE}/patients", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
pats = json.loads(r.read())
print(f'4. Patients: {pats["filtered"]} total')

# 5. Get single patient
pid = pats["patients"][0]["id"]
req = urllib.request.Request(f"{BASE}/patients/{pid}", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
p = json.loads(r.read())
print(f'5. Patient {pid}: {p["patient"]["name"]}, {len(p["scan_history"])} scans')

# 6. Analysis latest
req = urllib.request.Request(f"{BASE}/analysis/latest", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
a = json.loads(r.read())
print(f'6. Analysis: {a["diagnosis"]["primary"]} ({a["diagnosis"]["confidence"]}%)')

# 7. Reports latest
req = urllib.request.Request(f"{BASE}/reports/latest", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
rep = json.loads(r.read())
print(f'7. Reports: {rep["report_id"]} for {rep["patient_info"]["patient_name"]}')

# 8. Analytics
req = urllib.request.Request(f"{BASE}/analytics/summary", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
an = json.loads(r.read())
print(f'8. Analytics: {len(an["category_stats"])} stat categories')

# 9. Settings
req = urllib.request.Request(f"{BASE}/settings/profile", headers=hdr)
r = urllib.request.urlopen(req, timeout=5)
s = json.loads(r.read())
print(f'9. Settings: {s["first_name"]} {s["last_name"]}')

print("\nALL ENDPOINTS VERIFIED OK")
print(f"Swagger UI: http://127.0.0.1:8000/docs")
