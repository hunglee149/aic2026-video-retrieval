import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient

@pytest.fixture
def mock_hf_download():
    with patch("huggingface_hub.hf_hub_download") as mock:
        mock.return_value = "/mock/cached/path"
        yield mock

def test_resolve_path_success(mock_hf_download):
    with patch.dict(os.environ, {"AIC_HF_REPO_ID": "manhha2502/fullhd", "AIC_HF_REVISION": "main"}):
        sys.modules.pop("aic.ui.app", None)
        from aic.ui.app import resolve_path
        res = resolve_path("local/clip_faiss.index")
        assert res == Path("/mock/cached/path")
        mock_hf_download.assert_any_call(
            repo_id="manhha2502/fullhd",
            filename="local/clip_faiss.index",
            revision="main",
            repo_type="dataset",
            cache_dir=None
        )

def test_get_frame_mapping_direct():
    import json
    from aic.ui import app
    
    mock_data = [
        {"video_id": "L21_V001", "frame_idx": 100, "keyframe_num": 5}
    ]
    
    with patch("builtins.open", MagicMock()):
        with patch("json.load") as mock_json_load:
            mock_json_load.return_value = mock_data
            with patch.object(Path, "exists", return_value=True):
                app._frame_to_n = None
                mapping = app._get_frame_mapping()
                assert mapping == {("L21_V001", 100): 5}

def test_video_endpoints():
    from aic.ui.app import app as fastapi_app
    client = TestClient(fastapi_app)
    
    # Kiểm tra tính hoạt động của TestClient và app thông qua healthz ping
    try:
        ping_res = client.get("/healthz")
        ping_ok = ping_res.status_code == 200 and ping_res.json() == {"ok": True}
    except Exception as e:
        pytest.fail(f"Lỗi ở thư viện / bộ kiểm thử (TestClient ping thất bại): {e}")
    
    assert ping_ok, "Lỗi ở thư viện / bộ kiểm thử: không ping được FastAPI app qua /healthz"
    
    mock_meta = {
        "L21_V001": {"path": "videos/Videos_L21_a/video/L21_V001.mp4", "fps": 30.0}
    }
    
    with patch("aic.ui.app.load_video_metadata", return_value=mock_meta):
        res_info = client.get("/api/video_info/L21_V001")
        assert res_info.status_code == 200
        assert res_info.json() == {"fps": 30.0}
        
        res_video = client.get("/api/video/L21_V001", follow_redirects=False)
        assert res_video.status_code == 307
        assert "huggingface.co/datasets/manhha2502/fullhd" in res_video.headers["location"]
