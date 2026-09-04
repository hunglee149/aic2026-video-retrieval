import pytest
from aic.ui.app import validate_ws_update, merge_shared_state
import aic.ui.app as app_module


def test_validate_ws_update_valid():
    valid_data = {
        "type": "update",
        "manifest": [{"query_id": "q1", "task": "kis"}],
        "selections": [{"queryId": "q1", "video_id": "V001", "frames": [10, 20]}],
        "queryCache": {"q1": {"candidates": []}},
    }
    is_valid, msg = validate_ws_update(valid_data)
    assert is_valid is True
    assert msg == ""


def test_validate_ws_update_invalid():
    # Not a dict
    assert validate_ws_update("string")[0] is False
    assert validate_ws_update([1, 2, 3])[0] is False

    # Wrong type
    assert validate_ws_update({"type": "unknown"})[0] is False

    # Bad manifest
    assert validate_ws_update({"type": "update", "manifest": "not_a_list"})[0] is False
    assert validate_ws_update({"type": "update", "manifest": [{}]})[0] is False  # missing query_id

    # Bad selections
    assert validate_ws_update({"type": "update", "selections": "not_a_list"})[0] is False
    assert validate_ws_update({"type": "update", "selections": [{"frames": [1]}]})[0] is False  # missing video_id
    assert validate_ws_update({"type": "update", "selections": [{"video_id": "V1"}]})[0] is False  # missing frames
    assert validate_ws_update({"type": "update", "selections": [{"video_id": "V1", "frames": "not_list"}]})[0] is False

    # Bad queryCache
    assert validate_ws_update({"type": "update", "queryCache": "not_dict"})[0] is False


def test_merge_shared_state_multi_user_safety():
    # Reset app state
    app_module.shared_manifest = [{"query_id": "q1", "task": "kis"}]
    app_module.shared_selections = [
        {"queryId": "q1", "video_id": "V001", "frames": [100], "answer": ""}
    ]
    app_module.shared_query_cache = {"q1": {"candidates": ["c1"]}}

    # User B sends update for query q2 with their own selection
    incoming_manifest = [{"query_id": "q1", "task": "kis"}, {"query_id": "q2", "task": "qa"}]
    incoming_selections = [
        {"queryId": "q2", "video_id": "V002", "frames": [200], "answer": "cat"}
    ]
    incoming_cache = {"q2": {"candidates": ["c2"]}}

    merge_shared_state(incoming_manifest, incoming_selections, incoming_cache)

    # Manifest should have both q1 and q2
    qids = {m["query_id"] for m in app_module.shared_manifest}
    assert qids == {"q1", "q2"}

    # Selections should KEEP q1 selection from User A and ADD q2 selection from User B!
    assert len(app_module.shared_selections) == 2
    sel_queries = {s["queryId"] for s in app_module.shared_selections}
    assert sel_queries == {"q1", "q2"}

    # Query cache should have both q1 and q2
    assert "q1" in app_module.shared_query_cache
    assert "q2" in app_module.shared_query_cache


def test_merge_shared_state_empty_does_not_wipe():
    # User C joins with empty state
    app_module.shared_selections = [
        {"queryId": "q1", "video_id": "V001", "frames": [100], "answer": ""}
    ]
    merge_shared_state(incoming_manifest=[], incoming_selections=[], incoming_query_cache={})

    # Selection for q1 must NOT be wiped by an empty incoming array
    assert len(app_module.shared_selections) == 1
    assert app_module.shared_selections[0]["video_id"] == "V001"


def test_validate_ws_delete_query():
    # Valid delete_query
    is_valid, msg = validate_ws_update({"type": "delete_query", "query_id": "q1"})
    assert is_valid is True
    assert msg == ""

    # Invalid delete_query (missing query_id)
    is_valid, msg = validate_ws_update({"type": "delete_query"})
    assert is_valid is False
    assert "query_id" in msg

    # Invalid delete_query (empty query_id)
    is_valid, msg = validate_ws_update({"type": "delete_query", "query_id": ""})
    assert is_valid is False


def test_delete_shared_query():
    app_module.shared_manifest = [
        {"query_id": "q1", "task": "kis"},
        {"query_id": "q2", "task": "qa"}
    ]
    app_module.shared_selections = [
        {"queryId": "q1", "video_id": "V1", "frames": [1]},
        {"queryId": "q2", "video_id": "V2", "frames": [2]}
    ]
    app_module.shared_query_cache = {
        "q1": {"candidates": ["c1"]},
        "q2": {"candidates": ["c2"]}
    }

    app_module.delete_shared_query("q1")

    # q1 must be removed completely
    assert [m["query_id"] for m in app_module.shared_manifest] == ["q2"]
    assert [s["queryId"] for s in app_module.shared_selections] == ["q2"]
    assert "q1" not in app_module.shared_query_cache
    assert "q2" in app_module.shared_query_cache

