import pytest

from app.utils.rate_limit import check_rate_limit, clear_rate_limit


class TestLatestAnalysis:
    async def test_latest_analysis_empty_404(self, client, auth_headers):
        r = await client.get("/api/analysis/latest", headers=auth_headers)
        assert r.status_code == 404

    async def test_latest_analysis_structure(self, client, auth_headers):
        await client.post("/api/scans", headers=auth_headers,
                          data={"patient_id": "P-300", "patient_name": "Latest Patient"})
        r = await client.get("/api/analysis/latest", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "patient" in data and "diagnosis" in data
        assert data["patient"]["full_name"] == "Latest Patient"
        assert "probabilities" in data
        assert isinstance(data["diagnosis"]["confidence"], float)


class TestScanAnalysisEndpoints:
    async def test_scan_analysis_and_status(self, client, auth_headers):
        created = await client.post("/api/scans", headers=auth_headers,
                                    data={"patient_id": "P-301", "patient_name": "Scan Status"})
        scan_id = created.json()["scan_id"]

        status = await client.get(f"/api/scans/{scan_id}/status", headers=auth_headers)
        assert status.status_code == 200

        analysis = await client.get(f"/api/scans/{scan_id}/analysis", headers=auth_headers)
        assert analysis.status_code == 200
        data = analysis.json()
        assert data["diagnosis"]["primary"]
        assert isinstance(data["findings"], list)
        assert isinstance(data["recommendations"], list)

    async def test_scan_analysis_missing_404(self, client, auth_headers):
        assert (await client.get("/api/scans/SC-NOPE/analysis", headers=auth_headers)).status_code == 404
        assert (await client.get("/api/scans/SC-NOPE/status", headers=auth_headers)).status_code == 404


class TestGZipCompression:
    async def test_large_response_is_gzipped(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers={**auth_headers, "Accept-Encoding": "gzip"})
        assert r.status_code == 200
        assert r.headers.get("content-encoding") == "gzip"


class TestRateLimiterUnit:
    def test_check_and_clear(self):
        clear_rate_limit("unit:test")
        allowed = [check_rate_limit("unit:test", 3, 60) for _ in range(3)]
        assert allowed == [True, True, True]
        assert check_rate_limit("unit:test", 3, 60) is False
        clear_rate_limit("unit:test")
        assert check_rate_limit("unit:test", 3, 60) is True
        clear_rate_limit("unit:test")

    def test_keys_are_independent(self):
        a, b = "unit:key-a", "unit:key-b"
        clear_rate_limit(a)
        clear_rate_limit(b)
        assert check_rate_limit(a, 1, 60) is True
        assert check_rate_limit(a, 1, 60) is False
        assert check_rate_limit(b, 1, 60) is True
        clear_rate_limit(a)
        clear_rate_limit(b)
