"""
Tests for GraclusPool, build_block_diagonal_rect, GraphEncoder,
GraphAEEncoder, GraphTransposeDecoder, and GraphAutoencoder.
"""
import pytest
import torch
import torch.nn as nn

from models.graph_unet import GraclusPool, build_block_diagonal_rect
from graphae import GraphEncoder, GraphAEEncoder, GraphTransposeDecoder, GraphAutoencoder  # type: ignore[import]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ring_adj(N: int) -> torch.Tensor:
    """Sparse binary adjacency for an N-node ring graph."""
    rows, cols = [], []
    for i in range(N):
        j = (i + 1) % N
        rows += [i, j]
        cols += [j, i]
    idx = torch.tensor([rows, cols], dtype=torch.long)
    vals = torch.ones(len(rows))
    return torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()


def grid_adj(rows: int, cols: int) -> torch.Tensor:
    """Sparse binary adjacency for a rows×cols grid graph."""
    N = rows * cols
    src, dst = [], []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nb = nr * cols + nc
                    src += [node, nb]
                    dst += [nb, node]
    idx = torch.tensor([src, dst], dtype=torch.long)
    vals = torch.ones(len(src))
    return torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce()


# ---------------------------------------------------------------------------
# build_block_diagonal_rect
# ---------------------------------------------------------------------------

class TestBuildBlockDiagonalRect:
    def test_shape(self):
        N, K, B = 6, 4, 3
        rows = [0, 1, 2, 3, 4, 5]
        cols = [0, 0, 1, 1, 2, 3]
        vals = torch.ones(6)
        P = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), vals, (N, K)
        ).coalesce()
        P_block = build_block_diagonal_rect(P, B)
        assert P_block.size() == (B * N, B * K)

    def test_values_no_cross_block(self):
        """Entries should only exist within diagonal blocks."""
        N, K, B = 4, 3, 2
        rows = [0, 1, 2, 3]
        cols = [0, 0, 1, 2]
        vals = torch.ones(4)
        P = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), vals, (N, K)
        ).coalesce()
        P_block = build_block_diagonal_rect(P, B).to_dense()
        # Block 0: rows 0..N-1, cols 0..K-1
        # Block 1: rows N..2N-1, cols K..2K-1
        # Cross-block areas must be zero
        assert P_block[:N, K:].sum() == 0
        assert P_block[N:, :K].sum() == 0

    def test_block_values_match_original(self):
        N, K, B = 4, 3, 2
        rows = [0, 1, 2, 3]
        cols = [0, 0, 1, 2]
        vals = torch.tensor([0.5, 0.5, 1.0, 1.0])
        P = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), vals, (N, K)
        ).coalesce()
        P_dense = P.to_dense()
        P_block = build_block_diagonal_rect(P, B).to_dense()
        # Both blocks should match P_dense
        assert torch.allclose(P_block[:N, :K], P_dense)
        assert torch.allclose(P_block[N:, K:], P_dense)


# ---------------------------------------------------------------------------
# GraclusPool._greedy_matching
# ---------------------------------------------------------------------------

class TestGraclusMatching:
    def test_all_nodes_assigned(self):
        N = 8
        adj = ring_adj(N)
        assign, K = GraclusPool._greedy_matching(adj)
        assert assign.shape == (N,)
        assert (assign >= 0).all()
        assert K >= 1

    def test_coarsening_reduces_size(self):
        N = 10
        adj = ring_adj(N)
        _, K = GraclusPool._greedy_matching(adj)
        assert K < N

    def test_cluster_ids_contiguous(self):
        """Cluster IDs must be 0..K-1 with no gaps."""
        N = 12
        adj = grid_adj(3, 4)
        assign, K = GraclusPool._greedy_matching(adj)
        assert set(assign.tolist()) == set(range(K))

    def test_singleton_graph_no_edges(self):
        """A graph with no edges: every node is its own cluster."""
        N = 5
        # Self-loops only
        idx = torch.arange(N).unsqueeze(0).repeat(2, 1)
        adj = torch.sparse_coo_tensor(idx, torch.ones(N), (N, N)).coalesce()
        assign, K = GraclusPool._greedy_matching(adj)
        assert K == N


# ---------------------------------------------------------------------------
# GraclusPool.forward
# ---------------------------------------------------------------------------

class TestGraclusPoolForward:
    def setup_method(self):
        self.pool = GraclusPool()

    def test_output_shapes(self):
        N, H, B = 8, 16, 3
        adj = ring_adj(N)
        h = torch.randn(B * N, H)
        h_coarse, assign, adj_coarse, K = self.pool(h, adj, batch_size=B)
        assert h_coarse.shape == (B * K, H)
        assert assign.shape == (N,)
        assert adj_coarse.is_sparse
        assert adj_coarse.size() == (K, K)
        assert K < N

    def test_weighted_mean_pool(self):
        """Mean-pool: coarse node value should equal mean of its cluster members."""
        N, H, B = 4, 1, 1
        # Path graph: 0-1-2-3; greedy will pair (0,1) and (2,3)
        rows = [0, 1, 1, 2, 2, 3]
        cols = [1, 0, 2, 1, 3, 2]
        adj = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), torch.ones(6), (N, N)
        ).coalesce()
        # All-ones feature: every coarse node should have value 1.0
        h = torch.ones(N, H)
        h_coarse, assign, _, K = self.pool(h, adj, batch_size=1)
        assert torch.allclose(h_coarse, torch.ones(K, H), atol=1e-5)

    def test_cache_returns_same_assign(self):
        """Same graph → same assignment every time."""
        N, H, B = 6, 8, 1
        adj = ring_adj(N)
        h1 = torch.randn(N, H)
        h2 = torch.randn(N, H)
        _, assign1, _, K1 = self.pool(h1, adj, batch_size=B)
        _, assign2, _, K2 = self.pool(h2, adj, batch_size=B)
        assert K1 == K2
        assert torch.equal(assign1, assign2)

    def test_batch_independence(self):
        """Batched forward should be equivalent to concatenating B single-graph results."""
        N, H, B = 6, 4, 3
        adj = ring_adj(N)
        h = torch.randn(B * N, H)

        # Run batched
        h_coarse_batch, assign, _, K = self.pool(h, adj, batch_size=B)

        # Run single-graph B times and concatenate
        singles = []
        for b in range(B):
            h_b = h[b * N:(b + 1) * N]
            pool_b = GraclusPool()
            h_b_coarse, _, _, K_b = pool_b(h_b, adj, batch_size=1)
            assert K_b == K
            singles.append(h_b_coarse)
        h_coarse_manual = torch.cat(singles, dim=0)
        assert torch.allclose(h_coarse_batch, h_coarse_manual, atol=1e-5)

    def test_no_information_loss_constant_features(self):
        """With constant features, coarse features must equal that constant."""
        N, H, B = 10, 4, 2
        adj = grid_adj(2, 5)
        const_val = 3.7
        h = torch.full((B * N, H), const_val)
        h_coarse, _, _, _ = self.pool(h, adj, batch_size=B)
        assert torch.allclose(h_coarse, torch.full_like(h_coarse, const_val), atol=1e-5)

    def test_coarse_adj_is_symmetric(self):
        """Coarse adjacency should be symmetric (undirected graph)."""
        N, H, B = 8, 4, 1
        adj = ring_adj(N)
        h = torch.randn(N, H)
        _, _, adj_coarse, K = self.pool(h, adj, batch_size=B)
        dense = adj_coarse.to_dense()
        assert torch.equal(dense, dense.t())


# ---------------------------------------------------------------------------
# GraphEncoder
# ---------------------------------------------------------------------------

class TestGraphEncoder:
    def _make_encoder(self, **kwargs):
        defaults = dict(in_dim=1, hidden_dim=16, latent_dim=8, depth=2,
                        blocks_per_stage=1, dropout=0.0)
        defaults.update(kwargs)
        return GraphEncoder(**defaults)

    def test_output_shape(self):
        N, B = 12, 3
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc = self._make_encoder()
        z, mu, logvar = enc(x, adj, pos, batch_size=B)
        assert z.shape == (B, 8)
        assert mu is None
        assert logvar is None

    def test_stochastic_output_shape(self):
        N, B = 12, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc = self._make_encoder(use_stochastic=True)
        z, mu, logvar = enc(x, adj, pos, batch_size=B)
        assert z.shape == (B, 8)
        assert mu.shape == (B, 8)
        assert logvar.shape == (B, 8)

    def test_deterministic_eval_mode(self):
        """Non-stochastic encoder should be deterministic."""
        N, B = 10, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc = self._make_encoder().eval()
        with torch.no_grad():
            z1, _, _ = enc(x, adj, pos, batch_size=B)
            z2, _, _ = enc(x, adj, pos, batch_size=B)
        assert torch.equal(z1, z2)

    def test_gradient_flows(self):
        N, B = 10, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1, requires_grad=True)
        enc = self._make_encoder()
        z, _, _ = enc(x, adj, pos, batch_size=B)
        z.sum().backward()
        assert x.grad is not None
        assert x.grad.norm() > 0

    def test_depth_reduces_nodes(self):
        """Each pooling stage must produce strictly fewer nodes."""
        N, B = 16, 1
        adj = grid_adj(4, 4)
        pos = torch.randn(N, 3)
        x = torch.randn(N, 1)
        enc = self._make_encoder(depth=3).eval()
        # Just ensure forward completes without error
        with torch.no_grad():
            z, _, _ = enc(x, adj, pos, batch_size=B)
        assert z.shape == (1, 8)


# ---------------------------------------------------------------------------
# GraphAEEncoder
# ---------------------------------------------------------------------------

class TestGraphAEEncoder:
    def _make_encoder(self, N=12, **kwargs):
        defaults = dict(in_dim=1, hidden_dim=16, latent_dim=8, depth=2,
                        blocks_per_stage=1, dropout=0.0)
        defaults.update(kwargs)
        return GraphAEEncoder(**defaults)

    def test_output_shapes(self):
        N, B, depth = 12, 2, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc = self._make_encoder(depth=depth)
        z, pool_indices = enc(x, adj, pos, batch_size=B)
        assert z.shape == (B, 8)
        assert len(pool_indices) == depth

    def test_pool_indices_format(self):
        """Each entry must be (assign: LongTensor, K: int, adj_fine: sparse, adj_coarse: sparse)."""
        N, B, depth = 16, 2, 3
        adj = grid_adj(4, 4)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc = self._make_encoder(depth=depth)
        _, pool_indices = enc(x, adj, pos, batch_size=B)
        for i, entry in enumerate(pool_indices):
            assign, K, adj_fine, adj_coarse = entry
            assert assign.dtype == torch.long, f"stage {i}: assign must be long"
            assert isinstance(K, int), f"stage {i}: K must be int"
            assert adj_fine.is_sparse, f"stage {i}: adj_fine must be sparse"
            assert adj_coarse.is_sparse, f"stage {i}: adj_coarse must be sparse"
            assert assign.shape == (adj_fine.size(0),), f"stage {i}: assign length must equal N_fine"
            assert adj_coarse.size() == (K, K), f"stage {i}: adj_coarse must be K×K"

    def test_coarsening_monotone(self):
        """Each pooling stage must strictly reduce the number of nodes."""
        N, B, depth = 16, 1, 3
        adj = grid_adj(4, 4)
        pos = torch.randn(N, 3)
        x = torch.randn(N, 1)
        enc = self._make_encoder(depth=depth)
        _, pool_indices = enc(x, adj, pos, batch_size=B)
        sizes = [pool_indices[0][2].size(0)]  # adj_fine of first stage = original
        for assign, K, adj_fine, adj_coarse in pool_indices:
            sizes.append(K)
        for i in range(1, len(sizes)):
            assert sizes[i] < sizes[i - 1], f"stage {i}: K={sizes[i]} not < {sizes[i-1]}"


# ---------------------------------------------------------------------------
# GraphTransposeDecoder
# ---------------------------------------------------------------------------

class TestGraphTransposeDecoder:
    def _make_pair(self, N=12, B=2, depth=2):
        enc = GraphAEEncoder(in_dim=1, hidden_dim=16, latent_dim=8, depth=depth,
                             blocks_per_stage=1, dropout=0.0)
        dec = GraphTransposeDecoder(latent_dim=8, hidden_dim=16, out_dim=1, n_nodes=N,
                                    depth=depth, blocks_per_stage=1, dropout=0.0)
        return enc, dec

    def test_output_shape(self):
        N, B, depth = 12, 2, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc, dec = self._make_pair(N=N, B=B, depth=depth)
        z, pool_indices = enc(x, adj, pos, batch_size=B)
        rec = dec(z, adj, pos, pool_indices, batch_size=B)
        assert rec.shape == (B * N, 1)

    def test_wrong_depth_raises(self):
        N, B, depth = 10, 1, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(N, 1)
        enc, dec = self._make_pair(N=N, B=B, depth=depth)
        _, pool_indices = enc(x, adj, pos, batch_size=B)
        with pytest.raises(ValueError):
            dec(torch.randn(B, 8), adj, pos, pool_indices[:-1], batch_size=B)

    def test_gradient_flows(self):
        N, B, depth = 10, 2, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        enc, dec = self._make_pair(N=N, B=B, depth=depth)
        z, pool_indices = enc(x, adj, pos, batch_size=B)
        rec = dec(z, adj, pos, pool_indices, batch_size=B)
        rec.sum().backward()
        # Check that decoder parameters received gradients
        has_grad = any(p.grad is not None and p.grad.norm() > 0
                       for p in dec.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# GraphAutoencoder (end-to-end)
# ---------------------------------------------------------------------------

class TestGraphAutoencoder:
    def _make_model(self, N, depth=2, **kwargs):
        defaults = dict(in_dim=1, hidden_dim=16, latent_dim=8, depth=depth,
                        blocks_per_stage=1, dropout=0.0)
        defaults.update(kwargs)
        return GraphAutoencoder(n_nodes=N, **defaults)

    def test_forward_shapes(self):
        N, B = 12, 3
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        model = self._make_model(N)
        rec, z = model(x, adj, pos, batch_size=B)
        assert rec.shape == (B * N, 1)
        assert z.shape == (B, 8)

    def test_encode_decode_roundtrip_shape(self):
        N, B = 10, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        model = self._make_model(N)
        z, pool_indices = model.encode(x, adj, pos, batch_size=B)
        rec = model.decode(z, adj, pos, pool_indices, batch_size=B)
        assert rec.shape == x.shape

    def test_loss_decreases(self):
        """MSE loss should decrease over a few gradient steps on a fixed batch."""
        N, B = 12, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        model = self._make_model(N)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(10):
            opt.zero_grad()
            rec, _ = model(x, adj, pos, batch_size=B)
            loss = nn.functional.mse_loss(rec, x)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease over 10 steps"

    def test_different_batches_give_different_z(self):
        """Two different inputs should produce different latent codes."""
        N, B = 10, 2
        adj = ring_adj(N)
        pos = torch.randn(N, 3)
        model = self._make_model(N).eval()
        x1 = torch.randn(B * N, 1)
        x2 = torch.randn(B * N, 1)
        with torch.no_grad():
            _, z1 = model(x1, adj, pos, batch_size=B)
            _, z2 = model(x2, adj, pos, batch_size=B)
        assert not torch.allclose(z1, z2)

    def test_deeper_model(self):
        N, B, depth = 20, 2, 3
        adj = grid_adj(4, 5)
        pos = torch.randn(N, 3)
        x = torch.randn(B * N, 1)
        model = self._make_model(N, depth=depth)
        rec, z = model(x, adj, pos, batch_size=B)
        assert rec.shape == (B * N, 1)
        assert z.shape == (B, 8)
