import os
import sys
from io import StringIO
from pathlib import Path

import pytest
import yaml

from scalescope.model_spec import kv_heads_per_tp_rank
from scalescope.run_scalescope import main as scalescope_main

PROJECT_ROOT = Path(__file__).parent.parent
BASE_MODEL = PROJECT_ROOT / "model_zoo/hot_interconnects_2025/gpt_moe_config_1__32_experts.yaml"


@pytest.mark.parametrize(
    "num_kv_heads, tp_size, expected",
    [
        (128, 16, 8),
        (16, 16, 1),
        (8, 16, 1),
        (1, 16, 1),
        (16, 8, 2),
    ],
)
def test_kv_heads_per_tp_rank(num_kv_heads, tp_size, expected):
    assert kv_heads_per_tp_rank(num_kv_heads, tp_size) == expected


def _run_decode_with_kv_heads(tmp_path: Path, num_kv_heads: int) -> dict:
    """Run the Passage 32-expert decode scenario (TP_Attn=16) with a GQA model."""
    model = yaml.safe_load(BASE_MODEL.read_text())
    model["num_kv_heads"] = num_kv_heads
    model_path = tmp_path / f"gqa_{num_kv_heads}.yaml"
    model_path.write_text(yaml.safe_dump(model))
    sweep_path = tmp_path / "empty_sweep.yaml"
    sweep_path.write_text("{}\n")
    output_name = f"decode_kv{num_kv_heads}.yaml"

    args = [
        "scalescope-cli",
        "--hardware-lib",
        str(PROJECT_ROOT / "hardware_specs/hardware_library.yaml"),
        "--physical-topology",
        str(
            PROJECT_ROOT
            / "topology_zoo/physical/hot_interconnects_2025/passage_physical_32768.yaml"
        ),
        "--logical-topology",
        str(
            PROJECT_ROOT
            / "topology_zoo/logical/hot_interconnects_2025/passage_logical_config_1__32_experts.yaml"
        ),
        "--workload-spec",
        str(
            PROJECT_ROOT
            / "inference_decode_specs/hot_interconnects_2025/inference_decode_spec.yaml"
        ),
        "--model-spec",
        str(model_path),
        "--sweep-config",
        str(sweep_path),
        "--pass-mode",
        "inference_decode",
        "--output-file",
        output_name,
        "--raw-output",
    ]

    old_argv, old_cwd, old_stdout = sys.argv, os.getcwd(), sys.stdout
    sys.argv, sys.stdout = args, StringIO()
    try:
        os.chdir(tmp_path)
        try:
            scalescope_main()
        except SystemExit as e:
            assert e.code in (0, None)
    finally:
        sys.argv, sys.stdout = old_argv, old_stdout
        os.chdir(old_cwd)

    return yaml.safe_load((tmp_path / output_name).read_text())[0]


def test_decode_kv_cache_not_zero_when_tp_exceeds_kv_heads(tmp_path):
    # TP_Attn is 16 in this topology. With 8 KV heads, each rank still holds one
    # (replicated) KV head, so the KV cache must match the 16 KV head case, not vanish.
    res_8 = _run_decode_with_kv_heads(tmp_path, num_kv_heads=8)
    res_16 = _run_decode_with_kv_heads(tmp_path, num_kv_heads=16)

    kv_footprint_8 = res_8["memory_footprint"]["attn"]["kv_cache_bytes_per_xpu_per_layer"]
    kv_footprint_16 = res_16["memory_footprint"]["attn"]["kv_cache_bytes_per_xpu_per_layer"]
    assert kv_footprint_8 > 0
    assert kv_footprint_8 == kv_footprint_16

    kv_traffic_8 = res_8["memory_traffic_per_batch_per_xpu"]["forward"]["attn"]["by_component"][
        "kv_cache"
    ]
    kv_traffic_16 = res_16["memory_traffic_per_batch_per_xpu"]["forward"]["attn"]["by_component"][
        "kv_cache"
    ]
    assert kv_traffic_8["bytes_read_per_layer"] > 0
    assert kv_traffic_8["bytes_read_per_layer"] == kv_traffic_16["bytes_read_per_layer"]
    assert kv_traffic_8["bytes_written_per_layer"] == kv_traffic_16["bytes_written_per_layer"]
