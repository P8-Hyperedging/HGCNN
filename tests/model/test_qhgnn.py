import torch
import pytest

from model.QualityHGNN.layers import QHGNN_conv
from model.QualityHGNN.QHGNN import QHGNN


def test_qhgnn_conv_forward():
    """Test that QHGNN_conv produces output of the correct shape."""
    n_nodes = 5
    in_ft = 4
    out_ft = 3

    layer = QHGNN_conv(in_ft, out_ft)
    x = torch.rand(n_nodes, in_ft)
    G = torch.rand(n_nodes, n_nodes)
    Q = torch.ones(n_nodes, n_nodes)

    out = layer(x, G, Q)
    assert out.shape == (n_nodes, out_ft), f"Expected shape ({n_nodes}, {out_ft}), got {out.shape}"


def test_qhgnn_forward():
    """Test that QHGNN produces output of the correct shape."""
    n_nodes = 6
    in_ch = 8
    n_hid = 16
    n_class = 3

    model = QHGNN(in_ch=in_ch, n_class=n_class, n_hid=n_hid, dropout=0.0)
    x = torch.rand(n_nodes, in_ch)
    G = torch.rand(n_nodes, n_nodes)
    Q = torch.ones(n_nodes, n_nodes)

    out = model(x, G, Q)
    assert out.shape == (n_nodes, n_class), f"Expected shape ({n_nodes}, {n_class}), got {out.shape}"
