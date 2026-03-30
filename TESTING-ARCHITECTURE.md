# Testing Architecture Proposal

`edmkit` の `tests/` を、保守しやすく回帰に強い構成へ整理するための具体案。
`TESTING-TODO.md` が課題リストなのに対し、この文書は「どう整理するか」の設計案を定義する。

## Sources

この案は主に次の公式ガイドに合わせている。

- pytest Good Integration Practices  
  https://docs.pytest.org/en/stable/explanation/goodpractices.html
- pytest parametrization  
  https://docs.pytest.org/en/stable/how-to/parametrize.html
- Hypothesis settings profiles  
  https://hypothesis.readthedocs.io/en/latest/tutorial/settings.html
- Hypothesis flaky failures  
  https://hypothesis.readthedocs.io/en/latest/tutorial/flaky.html

要点:

- テストは `src/` 外の `tests/` に置き、役割ごとに明確に分ける
- 重複テストは `parametrize` で整理する
- Hypothesis は invariant を見る用途に絞る
- flaky な property test は禁止し、外部状態や unmanaged randomness を排除する
- suite-wide な Hypothesis profile は `conftest.py` で管理する

## Design Goals

1. どのテストが何を守るかを、ファイル名だけである程度分かるようにする
2. unit / property / trend / integration / smoke の責務を混ぜない
3. fixture と input generator の置き場をルール化し、ローカル helper の乱立を止める
4. assertion の強さをパターンごとに揃える
5. 新しい module を追加しても同じ型でテストを書けるようにする

## Required Test Categories

各 public module について、原則として次の分類で考える。

### 1. Unit

- 対象: validation, shape, dtype, exact examples, helper functions
- 役割: 最も安く、最も強く、失敗原因が最も明確なテスト
- 例:
- `lagged_embed` の index semantics
- `weights()` や `pearson_correlation()` の deterministic behavior
- `metrics`, `splits`, `util` の shape / error contract

### 2. Property

- 対象: 実装に依らない invariant
- 役割: examples では抜けやすい一般性を押さえる
- 例:
- permutation invariance
- mask と explicit filtering の一致
- batch と single-call の一致
- prediction range bounds

### 3. Trend

- 対象: exact 値ではなく方向や順位で見るべき現象
- 役割: EDM 的な妥当性確認
- 例:
- forecast horizon 増加で性能低下
- noise 増加で性能低下
- CCM の正方向が逆方向より強く収束

### 4. Integration

- 対象: 複数 module をつないだ wiring
- 役割: 埋め込みから予測までの接続確認
- 例:
- `lagged_embed -> simplex_projection`
- `lagged_embed -> smap`
- `lagged_embed -> with_simplex_projection`

### 5. Smoke

- 対象: packaging / import / minimal API availability
- 役割: release path で壊れていないことだけを確認
- 例:
- import 成功
- 代表 API が 1 回だけ成功する

## File Layout Proposal

現状の flat layout は維持しつつ、ファイル名にレイヤを出す。

```text
tests/
  conftest.py
  smoke_test.py

  test_embedding_unit.py
  test_embedding_scan_unit.py
  test_embedding_scan_property.py

  test_simplex_projection_unit.py
  test_simplex_projection_property.py
  test_simplex_projection_mask_property.py
  test_simplex_projection_tensor_integration.py
  test_simplex_projection_trend.py

  test_smap_unit.py
  test_smap_property.py
  test_smap_mask_property.py
  test_smap_trend.py

  test_ccm_unit.py
  test_ccm_property.py
  test_ccm_trend.py
  test_ccm_integration.py

  test_metrics_unit.py
  test_splits_unit.py
  test_util_unit.py
  test_util_property.py

  test_generate_unit.py
  test_generate_trend.py

  test_pipeline_integration.py
```

ポイント:

- `module + layer` を基本単位にする
- `mask`, `tensor`, `integration` は layer として意味が立つときだけ suffix で分ける
- `test_embed.py` のような短縮名は使わない
- `test_e2e.py` は `test_pipeline_integration.py` のように役割名へ変更する

## Naming Rules

### File names

原則:

- `test_<module>_<layer>.py`
- 例外:
- `smoke_test.py` は現状維持
- `conftest.py` は shared fixture / profile 用

許可する layer 名:

- `unit`
- `property`
- `trend`
- `integration`

追加 suffix:

- `mask`
- `tensor`
- `scan`

例:

- `test_simplex_projection_mask_property.py`
- `test_embedding_scan_unit.py`
- `test_ccm_trend.py`

### Test class names

原則:

- `Test<API><Aspect>`

例:

- `TestLaggedEmbedExamples`
- `TestLaggedEmbedValidation`
- `TestSimplexProjectionMaskEquivalence`
- `TestSMapTrend`

避ける:

- `TestAnalyticalSolutions`
- `TestMathematicalProperties`
- `TestConvergenceTrends`

これらは file を開かないと対象 API が分からないため。

### Test function names

原則:

- `test_<behavior>_<expected_outcome>`
- 例:
- `test_masked_call_matches_explicit_subset`
- `test_theta_zero_matches_ols`
- `test_select_raises_on_all_nan_scores`

## Input Definition And Sharing Rules

### 1. Inline literal

使う場面:

- 3-10 要素程度の hand-computable example
- その test にしか意味がない小さな入力

例:

- `lagged_embed` の exact indexing
- `to_xy` の幾何チェック

### 2. Local helper or local strategy

使う場面:

- その 1 ファイルだけで複数回使う
- 他 file へ再利用する意味が弱い

例:

- `weighted_lstsq_reference`
- `simplex_inputs`

原則:

- file 冒頭に置く
- docstring に「何の契約を見る入力か」を書く

### 3. Shared fixture in `conftest.py`

使う場面:

- 複数 file が同じドメイン系列を使う
- 系列自体に意味がある

例:

- `logistic_map`
- `lorenz_series`
- `causal_pair`

原則:

- fixture 名は系列名そのものにする
- docstring に「どのアルゴリズムのどの性質を見る系列か」を書く
- 未使用 fixture は置かない

### 4. Shared strategy helper

現時点では `conftest.py` に置かず、必要になったら `tests/strategies.py` を新設する。

作る条件:

- 同じ strategy を 3 file 以上で使う
- その strategy が単なる shape helper ではなく、意味のある invariant space を定義している

## Proposed Shared Inputs For Edmkit

`tests/conftest.py` に残すもの:

- `logistic_map`
- `bounded_linear_series`
- `lorenz_series`
- `causal_pair`
- `independent_pair`

削除候補:

- `sine_wave`
- `bidirectional_pair`

追加候補:

- `periodic_series`
  - 用途: `scan/select`, embedding delay, periodic predictability
- `constant_target`
  - 用途: simplex / smap / metrics
- `short_series`
  - 用途: validation / boundary

ただし `constant_target` や `short_series` は inline で十分なら fixture 化しない。

## Assertion Policy

### Unit

- exact に言えるものは `assert_array_equal` または厳密比較
- validation は `pytest.raises` で型とメッセージまで確認

### Property

- 1 test 1 invariant
- `shape` だけで終わらせない
- 期待すべき relation を明示する

良い例:

- `masked == explicit_subset`
- `batched == stack(single)`
- `prediction in [min(Y), max(Y)]`

悪い例:

- `isfinite`
- `not all NaN`
- `shape is correct`

これらは integration/smoke でのみ許容する。

### Trend

- exact 値ではなく、方向や順位で見る
- `corr > 0` は原則弱すぎる
- 代わりに次を使う:
- Spearman の符号
- 大小比較
- 正方向 > 逆方向

### Integration

- shape と finite を主に見る
- 数値品質の assertion を入れる場合は弱くしすぎない
- ただし主契約は wiring に留める

### Smoke

- import
- 代表 API の最小成功呼び出し
- `tests` helper や dev dependency に依存しない

## Hypothesis Policy

Hypothesis を使うのは次に限定する。

- shape preservation
- permutation invariance
- mask equivalence
- batch/unbatch consistency
- metric symmetry / bounds

避ける:

- ただの shape smoke
- 生成が偏った prefix-mask only property
- `assume(...)` だらけで例を大量に捨てる設計

原則:

- randomness は strategy が管理する
- test 本体で unmanaged RNG に依存しない
- 既知の壊れやすいケースは `@example` で固定する

## Template Patterns

### Template A: Unit exact example

```python
class TestLaggedEmbedExamples:
    def test_known_values_match_expected_matrix(self):
        x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        actual = lagged_embed(x, tau=2, e=2)
        expected = np.array([[2, 0], [3, 1], [4, 2], [5, 3]], dtype=float)
        np.testing.assert_array_equal(actual, expected)
```

### Template B: Validation contract

```python
class TestSMapValidation:
    def test_negative_theta_raises_value_error(self):
        X = np.zeros((5, 2))
        Y = np.zeros(5)
        Q = np.zeros((2, 2))
        with pytest.raises(ValueError, match="non-negative"):
            smap(X, Y, Q, theta=-1.0)
```

### Template C: Property invariant

```python
class TestSimplexProjectionMaskEquivalence:
    @given(...)
    def test_masked_call_matches_explicit_subset(self, ...):
        pred_masked = simplex_projection(X, Y, Q, mask=mask)
        pred_subset = simplex_projection(X[mask], Y[mask], Q)
        np.testing.assert_allclose(pred_masked, pred_subset, atol=1e-12)
```

### Template D: Trend test

```python
class TestCCMTrend:
    @pytest.mark.slow
    def test_forward_direction_converges_more_than_reverse(self, causal_pair):
        forward = ...
        reverse = ...
        assert forward[-1] > reverse[-1]
        rho, _ = spearmanr(lib_sizes, forward - reverse)
        assert rho > 0
```

### Template E: Integration test

```python
def test_simplex_pipeline_runs_end_to_end(periodic_series):
    embedding = lagged_embed(periodic_series, tau=1, e=3)
    pred = simplex_projection(embedding[:50], periodic_series[3:53], embedding[50:])
    assert pred.shape == (len(embedding) - 50,)
    assert np.all(np.isfinite(pred))
```

### Template F: Smoke test

```python
def test_basic_import_and_call():
    x = np.sin(np.linspace(0, 2 * np.pi, 40))
    emb = lagged_embed(x, tau=1, e=2)
    pred = simplex_projection(emb[:20], x[1:21], emb[20:])
    assert np.all(np.isfinite(pred))
```

## Concrete Migration Proposal

### Current -> Proposed

- `tests/test_embedding.py`
  - keep concept, rename to `tests/test_embedding_unit.py`

- `tests/test_embed.py`
  - split into:
  - `tests/test_embedding_scan_unit.py`
  - `tests/test_embedding_scan_property.py`
  - optionally `tests/test_embedding_scan_integration.py`

- `tests/test_simplex_projection.py`
  - split into:
  - `tests/test_simplex_projection_unit.py`
  - `tests/test_simplex_projection_property.py`
  - `tests/test_simplex_projection_trend.py`

- `tests/test_simplex_projection_mask.py`
  - rename to `tests/test_simplex_projection_mask_property.py`

- `tests/test_smap.py`
  - split into:
  - `tests/test_smap_unit.py`
  - `tests/test_smap_property.py`
  - `tests/test_smap_trend.py`

- `tests/test_smap_mask.py`
  - rename to `tests/test_smap_mask_property.py`

- `tests/test_ccm.py`
  - split into:
  - `tests/test_ccm_unit.py`
  - `tests/test_ccm_integration.py`
  - `tests/test_ccm_trend.py`

- `tests/test_util.py`
  - split into:
  - `tests/test_util_unit.py`
  - `tests/test_util_property.py`

- `tests/test_generate.py`
  - split into:
  - `tests/test_generate_unit.py`
  - `tests/test_generate_trend.py`

- `tests/test_e2e.py`
  - replace with `tests/test_pipeline_integration.py`

### Suggested Phase Order

1. rename and split `test_embed.py`, `test_e2e.py`
2. clean `conftest.py` and shared inputs
3. split `simplex_projection`, `smap`, `ccm` into unit/property/trend
4. split `util`, `generate`
5. trim `smoke_test.py`

## Config Recommendations

`pytest` 側の追加候補:

```toml
[tool.pytest.ini_options]
addopts = "-v --tb=short --import-mode=importlib"
```

理由:

- pytest の公式 docs は新規 project に `importlib` mode を推奨している
- test module 名の衝突や `sys.path` 由来の surprise を減らせる

ただし、repo の import 事情に影響するので、導入は別 PR で確認する。

## Decision Summary

この repo では、今後の基準を次で固定するのがよい。

1. file は `module + layer` で切る
2. shared input は `conftest.py`、shared strategy は必要になってから `tests/strategies.py`
3. unit は exact / validation、property は invariant、trend は方向比較、integration は wiring、smoke は import/minimal call
4. `shape/isfinite` 止まりのテストは integration/smoke に限定する
5. fixture と helper は「再利用される意味」がある場合だけ共有する
