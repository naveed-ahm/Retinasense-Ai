# RetinaSense API verification script (Step 2 / Step 12 API testing)
import io
import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
TOKEN = None


def req(method, path, body=None, headers=None, raw_body=None, ctype=None, timeout=180):
    url = BASE + path
    h = dict(headers or {})
    if TOKEN and "Authorization" not in h:
        h["Authorization"] = f"Bearer {TOKEN}"
    if raw_body is not None:
        data = raw_body
        if ctype:
            h["Content-Type"] = ctype
    else:
        data = json.dumps(body).encode() if body is not None else None
        if body is not None:
            h["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def multipart(fields, file_field, filename, file_bytes):
    boundary = "----RetinaSenseBoundary"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: application/octet-stream\r\n\r\n".encode() + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    if not cond:
        sys.exit(1)


def login():
    global TOKEN
    code, body = req("POST", "/api/auth/login", {"email": "dr.rajan@apollo.org", "password": "demo1234"})
    check("POST /api/auth/login returns 200", code == 200, f"(got {code})")
    TOKEN = json.loads(body)["token"]


def main():
    print("=== RetinaSense API Verification ===\n")

    # 1. Root + health
    code, body = req("GET", "/")
    check("GET / returns 200 + JSON", code == 200 and json.loads(body).get("name") == "RetinaSense AI", f"(got {code})")
    code, body = req("GET", "/api/health")
    health = json.loads(body)
    check("GET /api/health returns ok", code == 200 and health.get("status") == "ok" and health.get("model_loaded"), f"(got {code})")

    # 2. Auth
    login()
    code, _ = req("POST", "/api/auth/login", {"email": "dr.rajan@apollo.org", "password": "WRONG"})
    check("Login with wrong password -> 401", code == 401, f"(got {code})")

    # 3. Authenticated endpoints
    code, body = req("GET", "/api/dashboard/stats")
    check("GET /api/dashboard/stats returns 200", code == 200, f"(got {code})")
    code, _ = req("GET", "/api/dashboard/stats", headers={"Authorization": "Bearer invalid.token.here"})
    check("Dashboard with invalid token -> 401", code == 401, f"(got {code})")

    code, _ = req("GET", "/api/patients")
    check("GET /api/patients returns 200", code == 200, f"(got {code})")
    code, _ = req("GET", "/api/analytics/summary")
    check("GET /api/analytics/summary returns 200", code == 200, f"(got {code})")

    # 4. Scan upload — valid image
    img = open("D:/Projects/Proj/datasets/test/AMD/amd_13_0.png", "rb").read()
    data, ctype = multipart(
        {"patient_id": "PT-API1", "patient_name": "API Test", "patient_gender": "Female"},
        "image", "scan.png", img,
    )
    code, body = req("POST", "/api/scans", raw_body=data, ctype=ctype)
    check("POST /api/scans with valid image -> 200 + scan_id", code == 200 and "SC-" in body.decode(), f"(got {code})")
    scan_id = json.loads(body)["scan_id"]
    code, body = req("GET", f"/api/scans/{scan_id}/analysis")
    analysis = json.loads(body)
    check("GET /api/scans/{id}/analysis returns diagnosis JSON", code == 200 and analysis["diagnosis"]["primary"], f"(got {code})")
    code, body = req("GET", f"/api/scans/{scan_id}/report")
    check("GET /api/scans/{id}/report returns report JSON", code == 200 and json.loads(body)["report_id"], f"(got {code})")

    # 5. Scan upload — invalid file content (text bytes renamed .png)
    data, ctype = multipart({"patient_id": "PT-API1", "patient_name": "API Test"}, "image", "fake.png", b"not an image at all")
    code, body = req("POST", "/api/scans", raw_body=data, ctype=ctype)
    check("Invalid image content -> 400", code == 400, f"(got {code}) body={body.decode()[:60]}")

    # 6. Scan upload — wrong extension
    data, ctype = multipart({"patient_id": "PT-API1", "patient_name": "API Test"}, "image", "malware.exe", img)
    code, body = req("POST", "/api/scans", raw_body=data, ctype=ctype)
    check("Wrong extension (.exe) -> 400", code == 400, f"(got {code})")

    # 7. Scan upload — missing image -> simulate (200)
    data, ctype = multipart({"patient_id": "PT-API1", "patient_name": "API Test"}, "image", "scan.png", b"")
    code, body = req("POST", "/api/scans", raw_body=data, ctype=ctype)
    check("Empty file -> 400", code == 400, f"(got {code})")

    # 8. Scan upload — large file (11 MB)
    big = img + b"0" * (11 * 1024 * 1024)
    data, ctype = multipart({"patient_id": "PT-API1", "patient_name": "API Test"}, "image", "big.png", big)
    code, body = req("POST", "/api/scans", raw_body=data, ctype=ctype)
    check("Large file (11MB) -> 413", code == 413, f"(got {code})")

    # 9. Scan upload — unauthenticated
    code, _ = req("POST", "/api/scans", raw_body=data, ctype=ctype, headers={"Authorization": ""})
    check("Scan without auth -> 401/403", code in (401, 403), f"(got {code})")

    # 10. Predict endpoint
    data, ctype = multipart({}, "image", "scan.png", img)
    code, body = req("POST", "/api/predict", raw_body=data, ctype=ctype)
    pred = json.loads(body) if code == 200 else {}
    check("POST /api/predict -> 200 + prediction", code == 200 and pred.get("primary_diagnosis"), f"(got {code})")
    check("Predict returns image_url", bool(pred.get("image_url")), f"(got {pred.get('image_url')})")

    # 11. Upload endpoint
    code, body = req("POST", "/api/upload", raw_body=data, ctype=ctype)
    up = json.loads(body) if code == 200 else {}
    check("POST /api/upload -> 200 + url", code == 200 and up.get("image_url"), f"(got {code})")

    # 12. Image serving (authenticated)
    code, body = req("GET", up["image_url"])
    check("GET /api/images/{name} -> 200 image", code == 200 and len(body) > 1000, f"(got {code})")
    code, _ = req("GET", "/api/images/../app/config.py")
    check("Path traversal blocked -> 404/400", code in (404, 400, 401, 403), f"(got {code})")

    # 13. Scans list
    code, body = req("GET", "/api/scans?page=1&page_size=5")
    scans = json.loads(body) if code == 200 else {}
    check("GET /api/scans returns list + total", code == 200 and isinstance(scans.get("scans"), list), f"(got {code})")

    # 14. Patients search + pagination
    code, body = req("GET", "/api/patients?search=Eleanor&page=1&page_size=5")
    check("GET /api/patients?search= -> 200", code == 200, f"(got {code})")

    print("\n=== ALL API CHECKS PASSED ===")


if __name__ == "__main__":
    main()
