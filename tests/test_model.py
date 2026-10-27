import pytest

torch = pytest.importorskip("torch")

from aiori.model import ForecastGRU, count_parameters  # noqa: E402


@pytest.mark.parametrize("features, params", [(2, 13836), (4, 14220)])
def test_output_shape_and_parameter_count(features, params):
    m = ForecastGRU(features)
    assert m(torch.zeros(5, 12, features)).shape == (5, 12)
    assert count_parameters(m) == params


def test_state_dict_has_only_parameters_no_buffers():
    m = ForecastGRU(2)
    assert list(m.buffers()) == []
    assert sorted(m.state_dict()) == sorted(["gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0",
                                             "gru.bias_hh_l0", "head.weight", "head.bias"])
    assert sum(t.numel() * t.element_size() for t in m.state_dict().values()) == 55344
