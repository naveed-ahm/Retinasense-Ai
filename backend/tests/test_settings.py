import pytest


class TestProfile:
    async def test_get_profile(self, client, auth_headers):
        r = await client.get("/api/settings/profile", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["email"] == "dr.rajan@apollo.org"
        assert data["first_name"] == "Anil"
        assert data["last_name"] == "Rajan"
        assert data["initials"] == "AR"

    async def test_update_profile(self, client, auth_headers):
        r = await client.put("/api/settings/profile", headers=auth_headers, json={
            "first_name": "Anil", "last_name": "Rajan",
            "email": "dr.rajan@apollo.org", "phone": "+91-5555",
            "specialty": "Retina", "institution": "Apollo Hospitals",
        })
        assert r.status_code == 200
        assert r.json()["message"]

        r2 = await client.get("/api/settings/profile", headers=auth_headers)
        assert r2.json()["phone"] == "+91-5555"
        assert r2.json()["specialty"] == "Retina"

    async def test_profile_requires_auth(self, client):
        assert (await client.get("/api/settings/profile")).status_code == 401
        assert (await client.put("/api/settings/profile", json={})).status_code == 401


class TestNotifications:
    async def test_get_notifications(self, client, auth_headers):
        r = await client.get("/api/settings/notifications", headers=auth_headers)
        assert r.status_code == 200
        for key in ("critical_alerts", "report_ready", "weekly_digest", "model_updates"):
            assert isinstance(r.json()[key], bool)

    async def test_update_notifications(self, client, auth_headers):
        r = await client.put("/api/settings/notifications", headers=auth_headers, json={
            "critical_alerts": True, "report_ready": False,
            "weekly_digest": True, "model_updates": False,
        })
        assert r.status_code == 200
        r2 = await client.get("/api/settings/notifications", headers=auth_headers)
        assert r2.json()["critical_alerts"] is True
        assert r2.json()["report_ready"] is False


class TestTheme:
    async def test_update_theme(self, client, auth_headers):
        r = await client.put("/api/settings/theme", headers=auth_headers, json={"dark_mode": True})
        assert r.status_code == 200
        assert r.json()["message"]
