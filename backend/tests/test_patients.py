import pytest

class TestPatients:
    async def test_list_patients_empty(self, client, auth_headers):
        r = await client.get("/api/patients", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert "patients" in data
        assert data["total"] == 0
        assert data["filtered"] == 0

    async def test_create_patient(self, client, auth_headers):
        r = await client.post("/api/patients", json={
            "name": "Jane Doe", "age": 55,
            "condition": "Healthy", "status": "Active",
        }, headers=auth_headers)
        assert r.status_code == 201
        p = r.json()
        assert p["name"] == "Jane Doe"
        assert p["age"] == 55
        assert p["condition"] == "Healthy"

    async def test_list_patients_after_create(self, client, auth_headers):
        await client.post("/api/patients", json={
            "name": "John Smith", "age": 60,
            "condition": "Glaucoma", "status": "Active",
        }, headers=auth_headers)
        r = await client.get("/api/patients", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["filtered"] >= 1

    async def test_search_patients(self, client, auth_headers):
        await client.post("/api/patients", json={
            "name": "Alice Wonderland", "age": 45,
        }, headers=auth_headers)
        r = await client.get("/api/patients?search=Alice", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["filtered"] >= 1
        assert any("Alice" in p["name"] for p in data["patients"])

    async def test_get_patient(self, client, auth_headers):
        create = await client.post("/api/patients", json={
            "name": "Bob Builder", "age": 50,
        }, headers=auth_headers)
        pid = create.json()["id"]
        r = await client.get(f"/api/patients/{pid}", headers=auth_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["patient"]["name"] == "Bob Builder"

    async def test_get_patient_not_found(self, client, auth_headers):
        r = await client.get("/api/patients/P-9999", headers=auth_headers)
        assert r.status_code == 404

    async def test_delete_patient(self, client, auth_headers):
        create = await client.post("/api/patients", json={
            "name": "Delete Me", "age": 40,
        }, headers=auth_headers)
        pid = create.json()["id"]
        r = await client.delete(f"/api/patients/{pid}", headers=auth_headers)
        assert r.status_code == 204
        r2 = await client.get(f"/api/patients/{pid}", headers=auth_headers)
        assert r2.status_code == 404

    async def test_unauthorized_access(self, client):
        r = await client.get("/api/patients")
        assert r.status_code == 401
