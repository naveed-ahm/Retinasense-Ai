import pytest

class TestDashboard:
    async def test_dashboard_requires_auth(self, client):
        r = await client.get("/api/dashboard/stats")
        assert r.status_code == 401

    async def test_dashboard_structure(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "kpis" in data
        assert "scan_activity" in data
        assert "disease_distribution" in data
        assert "ai_performance" in data
        assert "recent_activity" in data
        assert isinstance(data["kpis"], list)
        assert isinstance(data["scan_activity"], list)
        assert isinstance(data["disease_distribution"], list)
        assert isinstance(data["ai_performance"], list)
        assert isinstance(data["recent_activity"], list)

    async def test_dashboard_kpis(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers=auth_headers)
        data = r.json()
        for kpi in data["kpis"]:
            assert "title" in kpi
            assert "value" in kpi
            assert "change" in kpi

    async def test_dashboard_activity(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers=auth_headers)
        data = r.json()
        for item in data["scan_activity"]:
            assert "day" in item
            assert "scans" in item
            assert "analyzed" in item

    async def test_dashboard_uses_real_data(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers=auth_headers)
        assert r.json()["kpis"][0]["value"] == "0"
        assert r.json()["disease_distribution"] == []

        await client.post("/api/scans", headers=auth_headers,
                          data={"patient_id": "P-200", "patient_name": "Real Data"})
        r2 = await client.get("/api/dashboard/stats", headers=auth_headers)
        data = r2.json()
        assert data["kpis"][0]["value"] == "1"
        assert any(d["name"] == "Diabetic Retinopathy" for d in data["disease_distribution"])
        assert len(data["recent_activity"]) >= 1

    async def test_analytics_uses_real_data(self, client, auth_headers):
        r = await client.get("/api/analytics/summary", headers=auth_headers)
        assert r.status_code == 200
        assert r.json()["summary"][0]["value"] == "0"

        await client.post("/api/scans", headers=auth_headers,
                          data={"patient_id": "P-200", "patient_name": "Real Data"})
        r2 = await client.get("/api/analytics/summary", headers=auth_headers)
        data = r2.json()
        assert data["summary"][0]["value"] == "1"
        assert any(item["label"] == "Diabetic Retinopathy" for item in data["category_stats"][0]["items"])
        assert len(data["scan_volume"]) == 7
