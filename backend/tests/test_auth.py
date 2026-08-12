import pytest

class TestLogin:
    async def test_login_success(self, client, seed_user):
        r = await client.post("/api/auth/login", json={
            "email": "dr.rajan@apollo.org",
            "password": "demo1234",
        })
        assert r.status_code == 200
        data = r.json()
        assert "token" in data
        assert data["user"]["name"] == "Anil Rajan"
        assert data["user"]["email"] == "dr.rajan@apollo.org"
        assert data["user"]["initials"] == "AR"

    async def test_login_wrong_password(self, client, seed_user):
        r = await client.post("/api/auth/login", json={
            "email": "dr.rajan@apollo.org",
            "password": "wrong",
        })
        assert r.status_code == 401

    async def test_login_nonexistent_user(self, client):
        r = await client.post("/api/auth/login", json={
            "email": "nobody@test.com",
            "password": "anything",
        })
        assert r.status_code == 401

class TestForgotPassword:
    async def test_forgot_password(self, client):
        r = await client.post("/api/auth/forgot-password", json={"email": "test@test.com"})
        assert r.status_code == 200
        assert "message" in r.json()

class TestRequestAccess:
    async def test_request_access(self, client):
        r = await client.post("/api/auth/request-access", json={
            "name": "Dr. Test", "email": "test@test.com",
            "institution": "Test Hospital",
        })
        assert r.status_code == 200
        assert "message" in r.json()
