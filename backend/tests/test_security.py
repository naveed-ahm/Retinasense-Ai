import pytest

from app.utils.security import hash_password, verify_password, create_access_token, decode_token


class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "demo1234"
        h = hash_password(pw)
        assert h != pw
        assert verify_password(pw, h)
        assert not verify_password("wrong", h)

    def test_unique_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token({"sub": "test@test.com"})
        payload = decode_token(token)
        assert payload is not None
        assert payload["sub"] == "test@test.com"
        assert "iat" in payload and "iss" in payload

    def test_decode_invalid(self):
        assert decode_token("invalid-token") is None
        assert decode_token("") is None


class TestLoginSecurity:
    async def test_disabled_account_cannot_login(self, client, db_session):
        from app.models.user import User
        from sqlalchemy import select
        user = User(
            email="disabled@apollo.org",
            password_hash=hash_password("demo1234"),
            first_name="X", last_name="Y",
            is_active=False,
        )
        db_session.add(user)
        await db_session.commit()

        r = await client.post("/api/auth/login",
                              json={"email": "disabled@apollo.org", "password": "demo1234"})
        assert r.status_code == 403

    async def test_login_rate_limited(self, client):
        payload = {"email": "nobody@nowhere.com", "password": "wrongpass"}
        for _ in range(5):
            await client.post("/api/auth/login", json=payload)
        r = await client.post("/api/auth/login", json=payload)
        assert r.status_code == 429


class TestSecurityHeaders:
    async def test_headers_present(self, client, auth_headers):
        r = await client.get("/api/dashboard/stats", headers=auth_headers)
        assert r.headers.get("x-content-type-options") == "nosniff"
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("referrer-policy") == "no-referrer"


class TestSettingsSecurity:
    async def test_duplicate_email_conflict(self, client, auth_headers, db_session):
        from app.models.user import User
        from sqlalchemy import select
        from app.utils.security import hash_password
        db_session.add(User(
            email="taken@apollo.org",
            password_hash=hash_password("demo1234"),
            first_name="Other", last_name="User",
        ))
        await db_session.commit()

        r = await client.put("/api/settings/profile", headers=auth_headers, json={
            "first_name": "Anil", "last_name": "Rajan",
            "email": "taken@apollo.org", "phone": "", "specialty": "", "institution": "",
        })
        assert r.status_code == 409

    async def test_weak_password_rejected(self, client, auth_headers):
        r = await client.put("/api/settings/password", headers=auth_headers, json={
            "current_password": "demo1234", "new_password": "short",
        })
        assert r.status_code == 400

    async def test_password_change(self, client, auth_headers):
        r = await client.put("/api/settings/password", headers=auth_headers, json={
            "current_password": "demo1234", "new_password": "newpassword123",
        })
        assert r.status_code == 200
        await client.put("/api/settings/password", headers=auth_headers, json={
            "current_password": "newpassword123", "new_password": "demo1234",
        })


class TestUploadValidation:
    async def test_oversized_upload_rejected(self, client, auth_headers):
        # 11 MB of zeros
        big = b"\x00" * (11 * 1024 * 1024)
        r = await client.post(
            "/api/upload",
            headers=auth_headers,
            files={"image": ("big.png", big, "image/png")},
        )
        assert r.status_code == 413

    async def test_invalid_image_rejected(self, client, auth_headers):
        r = await client.post(
            "/api/upload",
            headers=auth_headers,
            files={"image": ("fake.png", b"not-an-image", "image/png")},
        )
        assert r.status_code == 400

    async def test_bad_extension_rejected(self, client, auth_headers):
        r = await client.post(
            "/api/upload",
            headers=auth_headers,
            files={"image": ("evil.exe", b"\x89PNG\r\n\x1a\nfake", "application/octet-stream")},
        )
        assert r.status_code == 400
