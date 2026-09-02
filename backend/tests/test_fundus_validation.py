import cv2
import numpy as np
import pytest
from sqlalchemy import select

from app.models.prediction import Prediction
from app.models.report import Report
from app.models.scan import Scan
from app.services.fundus_validator import is_fundus_image


def _fundus_like_image() -> bytes:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    cv2.circle(image, (128, 128), 110, (35, 120, 220), -1)
    cv2.line(image, (128, 128), (45, 55), (20, 55, 115), 4)
    cv2.line(image, (128, 128), (205, 80), (20, 55, 115), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_fundus_validator_accepts_retinal_colour_field():
    image = cv2.imdecode(np.frombuffer(_fundus_like_image(), np.uint8), cv2.IMREAD_COLOR)
    assert is_fundus_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def test_fundus_validator_rejects_non_fundus_image():
    image = np.full((256, 256, 3), (220, 120, 20), dtype=np.uint8)
    assert not is_fundus_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def test_fundus_validator_rejects_orange_landscape():
    image = np.zeros((256, 384, 3), dtype=np.uint8)
    image[:140] = (115, 65, 150)  # sunset sky (BGR)
    image[140:210] = (25, 90, 210)  # reflected orange water
    image[210:] = (15, 20, 30)  # dark foreground
    cv2.rectangle(image, (300, 60), (383, 220), (10, 15, 10), -1)  # tree line
    assert not is_fundus_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


@pytest.mark.asyncio
async def test_non_fundus_scan_skips_model_prediction_and_report(client, auth_headers, db_session, monkeypatch):
    from app.services import analysis_service

    monkeypatch.setattr(
        analysis_service,
        "_get_model",
        lambda: (_ for _ in ()).throw(AssertionError("disease model must not be called")),
    )
    image = np.full((256, 256, 3), (220, 120, 20), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = await client.post(
        "/api/scans",
        headers=auth_headers,
        data={"patient_name": "Validation Patient"},
        files={"image": ("not-fundus.png", encoded.tobytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Invalid Image\n\n"
        "Please upload a valid retinal fundus image for disease screening."
    )
    assert (await db_session.scalar(select(Scan))) is None
    assert (await db_session.scalar(select(Prediction))) is None
    assert (await db_session.scalar(select(Report))) is None
