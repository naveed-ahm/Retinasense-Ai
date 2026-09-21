from pathlib import Path

import pytest

from app.config import settings
from app.models.scan import Scan


async def _scan_with_image(db_session, image_path: Path, scan_id: str = "SC-MATLAB"):
    db_session.add(Scan(
        scan_id=scan_id,
        patient_id="P-MATLAB",
        patient_name="MATLAB Test",
        image_path=str(image_path),
    ))
    await db_session.commit()


@pytest.mark.asyncio
async def test_matlab_unavailable_is_a_nonfatal_optional_response(client, auth_headers, db_session, tmp_path, monkeypatch):
    image = tmp_path / "fundus.png"
    image.write_bytes(b"image-placeholder")
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    await _scan_with_image(db_session, image)

    from app.routers import matlab
    monkeypatch.setattr(matlab.matlab_runner, "is_matlab_available", lambda: False)

    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-MATLAB"})

    assert response.status_code == 200
    assert response.json() == {"matlab_available": False, "matlab_status": "MATLAB analysis unavailable"}


@pytest.mark.asyncio
async def test_matlab_addon_returns_separate_quality_enhancement_and_features(client, auth_headers, db_session, tmp_path, monkeypatch):
    image = tmp_path / "fundus.png"
    enhanced = tmp_path / "fundus_matlab.png"
    image.write_bytes(b"image-placeholder")
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    await _scan_with_image(db_session, image)

    from app.routers import matlab
    monkeypatch.setattr(matlab.matlab_runner, "is_matlab_available", lambda: True)
    monkeypatch.setattr(matlab.matlab_runner, "run_quality_analysis", lambda _: {
        "quality_score": 86, "brightness_score": 82, "contrast_score": 90,
        "sharpness_score": 84, "fov_score": 88, "status": "GOOD",
    })
    monkeypatch.setattr(matlab.matlab_runner, "run_enhancement", lambda _, output: {"enhancement_path": str(enhanced)})
    monkeypatch.setattr(matlab.matlab_runner, "run_retinal_features", lambda _: {"mean_intensity": 0.52, "contrast": 0.31})

    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-MATLAB"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matlab_available"] is True
    assert payload["quality"]["quality_score"] == 86
    assert payload["enhanced_image"] == "fundus_matlab.png"
    assert payload["features"] == {"mean_intensity": 0.52, "contrast": 0.31}


@pytest.mark.asyncio
async def test_matlab_addon_rejects_scan_paths_outside_uploads(client, auth_headers, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image-placeholder")
    await _scan_with_image(db_session, outside)

    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-MATLAB"})

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid uploaded image reference"


@pytest.mark.asyncio
async def test_matlab_feature_failure_does_not_hide_quality_or_enhancement(client, auth_headers, db_session, tmp_path, monkeypatch):
    image = tmp_path / "fundus.png"
    enhanced = tmp_path / "fundus_matlab.png"
    image.write_bytes(b"image-placeholder")
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    await _scan_with_image(db_session, image)

    from app.routers import matlab
    monkeypatch.setattr(matlab.matlab_runner, "is_matlab_available", lambda: True)
    monkeypatch.setattr(matlab.matlab_runner, "run_quality_analysis", lambda _: {"quality_score": 80})
    monkeypatch.setattr(matlab.matlab_runner, "run_enhancement", lambda _, output: {"enhancement_path": str(enhanced)})
    monkeypatch.setattr(matlab.matlab_runner, "run_retinal_features", lambda _: (_ for _ in ()).throw(RuntimeError("optional")))

    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-MATLAB"})

    assert response.status_code == 200
    assert response.json()["matlab_available"] is True
    assert response.json()["quality"] == {"quality_score": 80}
    assert "features" not in response.json()


@pytest.mark.asyncio
async def test_matlab_addon_requires_authentication(client):
    response = await client.post("/api/matlab/analyze", data={"scan_id": "SC-MATLAB"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_matlab_addon_missing_scan_returns_404(client, auth_headers):
    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-NONEXISTENT"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Scan not found"


@pytest.mark.asyncio
async def test_matlab_addon_preserves_original_image(client, auth_headers, db_session, tmp_path, monkeypatch):
    import hashlib

    original_bytes = b"original-fundus-image-data-bytes"
    image = tmp_path / "fundus_scan.png"
    image.write_bytes(original_bytes)
    orig_hash = hashlib.sha256(original_bytes).hexdigest()

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path))
    await _scan_with_image(db_session, image, scan_id="SC-HASH-TEST")

    from app.routers import matlab
    monkeypatch.setattr(matlab.matlab_runner, "is_matlab_available", lambda: True)
    monkeypatch.setattr(matlab.matlab_runner, "run_quality_analysis", lambda _: {"quality_score": 85})

    def fake_enhancement(input_path, output_path):
        out_file = Path(output_path)
        out_file.write_bytes(b"enhanced-fundus-data")
        return {"enhancement_path": str(out_file)}

    monkeypatch.setattr(matlab.matlab_runner, "run_enhancement", fake_enhancement)

    response = await client.post("/api/matlab/analyze", headers=auth_headers, data={"scan_id": "SC-HASH-TEST"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["matlab_available"] is True
    assert payload["enhanced_image"] != image.name

    # Confirm original image content and hash are 100% byte-identical
    assert image.read_bytes() == original_bytes
    assert hashlib.sha256(image.read_bytes()).hexdigest() == orig_hash


def test_matlab_runner_availability_check():
    try:
        from matlab_service import matlab_runner
    except ImportError:
        from backend.matlab_service import matlab_runner

    available = matlab_runner.is_matlab_available()
    assert isinstance(available, bool)


def test_matlab_runner_offline_safe(monkeypatch):
    try:
        from matlab_service import matlab_runner
    except ImportError:
        from backend.matlab_service import matlab_runner

    monkeypatch.setattr(matlab_runner, "_get_engine", lambda: (_ for _ in ()).throw(matlab_runner.MatlabUnavailable("Engine unavailable")))
    assert matlab_runner.is_matlab_available() is False


