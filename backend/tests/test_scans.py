import pytest
from sqlalchemy import select, func
from app.models.prediction import Prediction
from app.models.report import Report


async def _create_scan(client, auth_headers, patient_id="", name="History Patient"):
    data = {"patient_name": name}
    if patient_id:
        data["patient_id"] = patient_id
    return await client.post(
        "/api/scans",
        headers=auth_headers,
        data=data,
    )


@pytest.mark.asyncio
async def test_scans_require_auth(client):
    assert (await client.get("/api/scans")).status_code == 401
    assert (await client.get("/api/scans/SC-X/status")).status_code == 401
    assert (await client.get("/api/scans/SC-X/analysis")).status_code == 401
    assert (await client.get("/api/scans/SC-X/report")).status_code == 401


@pytest.mark.asyncio
async def test_scan_persists_prediction_and_report(client, auth_headers, db_session):
    created = await _create_scan(client, auth_headers)
    scan_id = created.json()["scan_id"]

    pred = await db_session.scalar(select(Prediction).where(Prediction.scan_id == scan_id))
    assert pred is not None
    assert pred.primary_diagnosis == "Diabetic Retinopathy"
    assert pred.confidence > 0

    rep = await db_session.scalar(select(Report).where(Report.scan_id == scan_id))
    assert rep is not None
    assert rep.report_data["report_id"] == rep.report_id
    assert rep.prediction_id == pred.id


@pytest.mark.asyncio
async def test_scan_list_search_filter_paginate(client, auth_headers):
    r1 = await _create_scan(client, auth_headers, name="Zeta Patient")
    r2 = await _create_scan(client, auth_headers, name="Alpha Patient")

    r = await client.get("/api/scans", headers=auth_headers)
    body = r.json()
    assert body["total"] >= 2
    assert body["filtered"] == body["total"]
    assert "page" in body and "page_size" in body

    r2 = await client.get("/api/scans", params={"search": "Alpha"}, headers=auth_headers)
    assert r2.json()["filtered"] == 1
    assert r2.json()["scans"][0]["patient_name"] == "Alpha Patient"

    r3 = await client.get("/api/scans", params={"search": "Zeta"}, headers=auth_headers)
    assert r3.json()["filtered"] == 1

    r4 = await client.get("/api/scans", params={"page": -1, "page_size": 500}, headers=auth_headers)
    assert r4.status_code == 200
    assert r4.json()["page"] == 1
    assert r4.json()["page_size"] == 100

    r5 = await client.get("/api/scans", params={"sort_by": "confidence", "sort_dir": "asc"}, headers=auth_headers)
    assert r5.status_code == 200


@pytest.mark.asyncio
async def test_patient_detail_uses_real_scan_history(client, auth_headers):
    created = await client.post(
        "/api/patients",
        headers=auth_headers,
        json={"name": "History Patient", "age": 60},
    )
    pid = created.json()["id"]
    await _create_scan(client, auth_headers, patient_id=pid, name="History Patient")
    r = await client.get(f"/api/patients/{pid}", headers=auth_headers)
    assert r.status_code == 200
    history = r.json()["scan_history"]
    assert any(h["scan_id"] for h in history)
    assert history[0]["result"] == "Diabetic Retinopathy"


@pytest.mark.asyncio
async def test_delete_scan_and_report(client, auth_headers, db_session):
    created = await _create_scan(client, auth_headers)
    scan_id = created.json()["scan_id"]

    assert (await client.delete(f"/api/scans/{scan_id}/report", headers=auth_headers)).status_code == 204
    assert await db_session.scalar(select(func.count()).select_from(Report).where(Report.scan_id == scan_id)) == 0

    assert (await client.delete(f"/api/scans/{scan_id}", headers=auth_headers)).status_code == 204
    assert await db_session.scalar(select(func.count()).select_from(Prediction).where(Prediction.scan_id == scan_id)) == 0
    assert (await client.get(f"/api/scans/{scan_id}/status", headers=auth_headers)).status_code == 404
