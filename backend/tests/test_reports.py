import pytest


@pytest.mark.asyncio
async def test_latest_report_json(client, auth_headers):
    r = await client.get("/api/reports/latest", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["report_id"].startswith("RS-")
    assert body["diagnosis"]["severity"] in {"Moderate", "Severe", "Critical", "Mild"}
    assert isinstance(body["diagnosis"]["risk_score"], float)
    assert body["patient_info"]["patient_name"]


@pytest.mark.asyncio
async def test_latest_report_pdf(client, auth_headers):
    r = await client.get("/api/reports/latest/pdf", headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert len(r.content) > 1000
    assert "Content-Disposition" in r.headers


@pytest.mark.asyncio
async def test_reports_require_auth(client):
    assert (await client.get("/api/reports/latest")).status_code == 401
    assert (await client.get("/api/reports/latest/pdf")).status_code == 401


@pytest.mark.asyncio
async def test_scan_specific_report_pdf(client, auth_headers):
    created = await client.post(
        "/api/scans",
        headers=auth_headers,
        data={"patient_id": "P-99", "patient_name": "Per Scan Patient"},
    )
    assert created.status_code == 200
    scan_id = created.json()["scan_id"]

    jr = await client.get(f"/api/scans/{scan_id}/report", headers=auth_headers)
    assert jr.status_code == 200

    pr = await client.get(f"/api/scans/{scan_id}/report/pdf", headers=auth_headers)
    assert pr.status_code == 200
    assert pr.headers["content-type"] == "application/pdf"
    assert pr.content[:5] == b"%PDF-"

    bad = await client.get("/api/scans/SC-NOPE/report/pdf", headers=auth_headers)
    assert bad.status_code == 404


@pytest.mark.asyncio
async def test_patient_report_pdf(client, auth_headers):
    # Create patient first (no seed data)
    cp = await client.post("/api/patients", json={
        "name": "Eleanor Vasquez", "age": 62,
        "condition": "Diabetic Retinopathy", "status": "Active",
    }, headers=auth_headers)
    assert cp.status_code == 201
    pid = cp.json()["id"]

    created = await client.post(
        "/api/scans",
        headers=auth_headers,
        data={"patient_id": pid, "patient_name": "Eleanor Vasquez"},
    )
    assert created.status_code == 200

    r = await client.get(f"/api/patients/{pid}/report/pdf", headers=auth_headers)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"

    bad = await client.get("/api/patients/P-NOPE/report/pdf", headers=auth_headers)
    assert bad.status_code == 404
