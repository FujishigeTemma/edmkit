# TESTING.md — edmkit テスト設計書

## 1. 背景と目的

### 1.1 現状の課題

現在のテストスイートは pyEDM の出力を「正解」として比較する方式を採用している。この方式には以下の問題がある。

1. **実行速度**: pyEDM はネイティブライブラリ（cppEDM）のラッパーであり、テスト実行時のインストール・起動コストが大きい
2. **柔軟性の欠如**: pyEDM の API や内部挙動に依存しているため、edmkit 独自の拡張（バッチ処理、カスタムサンプラー等）のテストが困難
3. **可読性**: pyEDM 固有のパラメータ変換（tau の符号反転、ライブラリ範囲の文字列指定等）がテストコードの理解を妨げている
4. **脆弱性**: pyEDM のバージョンアップにより、テストが意図せず壊れるリスクがある

### 1.2 目標

- pyEDM への依存を排除し、edmkit 独自のテストスイートを構築する
- 数理的性質に基づいた検証により、実装の正しさをアルゴリズムの本質から保証する
- テストの実行速度を高速に保ち、開発サイクルを阻害しない
- テストコードが各アルゴリズムの仕様書としても機能する可読性を実現する

---

## 2. テスト設計の基本方針

### 2.1 数理計算ライブラリにおけるテストの原則

数理計算ライブラリのテストでは、外部実装との比較（cross-validation against reference implementation）は初期開発のサニティチェックとしては有用だが、長期的なテスト戦略としては以下のアプローチが望ましい。

| テスト手法 | 説明 | 適用箇所 |
|---|---|---|
| **解析解との比較** | 理論的に答えが導出できるケースでの厳密な検証 | 線形系、既知の不動点、自明なケース |
| **数学的不変量の検証** | アルゴリズムが満たすべき性質の確認 | 対称性、等冪性、境界条件 |
| **収束性テスト** | パラメータ変化に伴う漸近的振る舞いの検証 | ライブラリサイズ増加、ノイズ低減時 |
| **退化ケース（degenerate case）** | 極端な入力での正しい挙動の検証 | 定数時系列、次元1、サンプル数最小 |
| **摂動テスト** | 微小な入力変化に対する出力の安定性の検証 | 数値安定性、条件数 |
| **自己無撞着性テスト** | 同一結果を別経路で計算し一致を確認 | バッチ vs 個別、異なるコードパス |
| **エッジケース・エラーハンドリング** | 不正入力に対する適切なエラーの検証 | 型チェック、次元チェック、値域チェック |

### 2.2 テスト分類

テストを以下の3層に分類する。

```
Layer 1: Unit Tests（単体テスト）
  各関数の入出力を独立に検証。高速に実行可能。

Layer 2: Integration Tests（結合テスト）
  複数モジュールの連携を検証（例: embedding → simplex_projection）。

Layer 3: Smoke Tests（疎通テスト）
  エンドツーエンドのパイプラインが動作することの確認。
```

---

## 3. モジュール別テスト設計

### 3.1 `embedding.py` — `lagged_embed()`

#### 3.1.1 解析解テスト

| テスト名 | 内容 | 期待値 |
|---|---|---|
| `test_identity_embedding` | `e=1, tau=1` ではリシェイプのみ | `x.reshape(-1, 1)` と一致 |
| `test_known_values` | 小さな配列で手計算と照合 | 各行が `[x[i+(e-1)*tau], ..., x[i]]` |
| `test_output_shape` | 任意の `(N, tau, e)` に対して出力形状を検証 | `(N - (e-1)*tau, e)` |
| `test_linear_sequence` | 等差数列の入力で各行の差分が `tau` 刻み | 各行内の差が均一 |

#### 3.1.2 数学的性質テスト

| テスト名 | 内容 |
|---|---|
| `test_no_information_loss` | 出力の全要素が入力に含まれている |
| `test_tau1_e1_preserves_values` | `tau=1, e=1` で元の値がそのまま保持される |
| `test_sliding_window_consistency` | 連続する行が1ステップずつスライドしている |

#### 3.1.3 エラーハンドリング

| テスト名 | 条件 | 期待 |
|---|---|---|
| `test_rejects_2d_input` | 2D配列を渡す | `ValueError` |
| `test_rejects_zero_tau` | `tau=0` | `ValueError` |
| `test_rejects_zero_e` | `e=0` | `ValueError` |
| `test_rejects_insufficient_length` | `(e-1)*tau >= len(x)` | `ValueError` |

---

### 3.2 `util.py`

#### 3.2.1 `pairwise_distance` / `pairwise_distance_np`

**解析解テスト**

| テスト名 | 内容 |
|---|---|
| `test_known_2d_distances` | 手計算可能な2-3点の二乗距離 |
| `test_identity_distance_zero_diagonal` | `A=B` のとき対角要素が0 |
| `test_single_point` | 1点同士の距離は差の二乗和 |

**数学的性質テスト**

| テスト名 | 検証する性質 |
|---|---|
| `test_symmetry` | `D(A, B) = D(B, A)^T` |
| `test_non_negativity` | 全要素 ≥ 0 |
| `test_self_distance_diagonal_zero` | `D(A, A)` の対角は0 |
| `test_triangle_inequality` | `sqrt(D[i,k]) ≤ sqrt(D[i,j]) + sqrt(D[j,k])` |
| `test_translation_invariance` | `D(A+c, B+c) = D(A, B)` |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 |

**NumPy / Tensor 一致テスト**

| テスト名 | 内容 |
|---|---|
| `test_numpy_tensor_agreement` | 同一入力に対して `pairwise_distance` と `pairwise_distance_np` が同一結果 |

#### 3.2.2 `dtw`

**解析解テスト**

| テスト名 | 内容 | 期待値 |
|---|---|---|
| `test_identical_sequences` | 同一系列 | 0 |
| `test_single_element` | 1要素同士 | ユークリッド距離 |
| `test_known_small_case` | 2-3要素で手計算 | 既知のDP結果 |

**数学的性質テスト**

| テスト名 | 検証する性質 |
|---|---|
| `test_non_negativity` | DTW距離 ≥ 0 |
| `test_identity_of_indiscernibles` | D=0 ⟺ 同一系列 |
| `test_symmetry` | `dtw(A, B) = dtw(B, A)` |

#### 3.2.3 `autocorrelation`

**解析解テスト**

| テスト名 | 内容 | 期待値 |
|---|---|---|
| `test_lag_zero_is_one` | ラグ0の自己相関 | 1.0 |
| `test_white_noise` | 長い白色ノイズ | ラグ>0 で ≈ 0 |
| `test_sine_wave` | 正弦波で周期的パターン | 半周期でラグ ≈ -1 |
| `test_constant_signal` | 定数信号 | 全ラグで NaN または 0/0 の適切な処理 |

#### 3.2.4 `pad`

| テスト名 | 内容 |
|---|---|
| `test_same_dimensions_no_padding` | 同一サイズの配列はパディング不要 |
| `test_mixed_dimensions_zero_fill` | 異なるサイズの配列がゼロ埋めされる |
| `test_output_shape` | 出力形状が `(B, L, max_D)` |

---

### 3.3 `simplex_projection.py` — `simplex_projection()`

#### 3.3.1 解析解テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_identity_prediction` | クエリ点がライブラリ点と一致する場合 | 最近傍距離=0 → 重み集中 → 完全予測 |
| `test_linear_system_exact` | 線形力学系 `x[t+1] = a*x[t]` の予測 | 線形系はE=1で完全に再構成可能 |
| `test_constant_target` | Y が定数の場合 | 重みによらず予測値 = 定数 |

#### 3.3.2 数学的性質テスト

| テスト名 | 検証する性質 |
|---|---|
| `test_prediction_in_target_range` | 予測値が Y の値域 [min(Y), max(Y)] に収まる（凸結合性） |
| `test_neighbor_count_equals_e_plus_1` | k = E + 1 近傍を使用（E次元単体の頂点数） |
| `test_prediction_improves_with_optimal_e` | 最適な埋め込み次元で予測精度が向上する |
| `test_permutation_invariance_of_library` | ライブラリ点の順序入れ替えで結果不変 |

#### 3.3.3 収束性テスト

| テスト名 | 内容 |
|---|---|
| `test_accuracy_improves_with_library_size` | ライブラリサイズ増加に伴い RMSE が非増加傾向（カオス系で検証） |
| `test_noise_sensitivity` | ノイズレベル増加に伴い精度が単調劣化 |

#### 3.3.4 自己無撞着性テスト

| テスト名 | 内容 |
|---|---|
| `test_batch_vs_individual` | 3Dバッチ入力と2D個別入力の結果一致 |
| `test_numpy_vs_tensor` | `use_tensor=True/False` で同一結果（許容誤差内） |

#### 3.3.5 退化ケース

| テスト名 | 内容 |
|---|---|
| `test_single_query_point` | クエリ点1つ |
| `test_minimum_library_size` | ライブラリサイズ = E + 1（最小近傍数） |

---

### 3.4 `smap.py` — `smap()`

#### 3.4.1 解析解テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_theta_zero_equals_ols` | `theta=0` でOLS（最小二乗法）と一致 | `theta=0` は全点等重み → グローバル線形回帰 |
| `test_linear_system_recovery` | `Y = a*X + b` で `theta=0` のとき完全復元 | 線形関係はOLSで完全にフィット |
| `test_identity_prediction` | クエリ点がライブラリ点に一致 | 局所線形でも完全予測可能 |
| `test_constant_target` | Y が定数の場合 | 切片項のみで予測 |

#### 3.4.2 数学的性質テスト

| テスト名 | 検証する性質 |
|---|---|
| `test_theta_increases_locality` | `theta` 増加に伴い、遠方点の影響が減少 |
| `test_regularization_effect` | `alpha` が大きいほど係数が縮小（リッジ回帰の性質） |
| `test_intercept_not_regularized` | 切片項は正則化されない（バイアス不変性） |
| `test_permutation_invariance_of_library` | ライブラリ順序に依存しない |

#### 3.4.3 theta に関する比較テスト

| テスト名 | 内容 |
|---|---|
| `test_theta_zero_vs_nonzero_on_linear` | 線形系ではどの `theta` でも同精度（全点が同一超平面上） |
| `test_theta_nonzero_better_on_nonlinear` | 非線形系では `theta > 0` が `theta = 0` より高精度 |

#### 3.4.4 自己無撞着性テスト

| テスト名 | 内容 |
|---|---|
| `test_batch_vs_individual` | バッチ処理と個別処理の一致 |

#### 3.4.5 数値安定性テスト

| テスト名 | 内容 |
|---|---|
| `test_ill_conditioned_library` | ほぼ共線的なライブラリ点でも正則化により安定 |
| `test_large_theta_stability` | `theta` が大きくても NaN/Inf が発生しない |

---

### 3.5 `ccm.py`

#### 3.5.1 `pearson_correlation`

**解析解テスト**

| テスト名 | 内容 | 期待値 |
|---|---|---|
| `test_perfect_positive` | `Y = aX + b (a > 0)` | 1.0 |
| `test_perfect_negative` | `Y = -aX + b` | -1.0 |
| `test_uncorrelated` | 直交する正弦波 | ≈ 0 |
| `test_self_correlation` | `X` と `X` | 1.0 |

**数学的性質テスト**

| テスト名 | 検証する性質 |
|---|---|
| `test_range` | -1 ≤ corr ≤ 1 |
| `test_symmetry` | `corr(X, Y) = corr(Y, X)` |
| `test_invariance_to_linear_transform` | `corr(aX+b, cY+d) = sign(ac) * corr(X, Y)` |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 |

#### 3.5.2 `bootstrap` / `ccm`

**収束性テスト（CCM の本質的な検証）**

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_convergence_with_known_causality` | X→Y の因果がある系で、ライブラリサイズ増加に伴い相関が収束（増加傾向） | CCM の定義的性質: 因果があれば収束する |
| `test_no_convergence_without_causality` | 独立な系列では収束しない | 偽陽性がないことの確認 |

**構成要素のテスト**

| テスト名 | 内容 |
|---|---|
| `test_custom_sampler` | カスタムサンプラーが正しく呼ばれる |
| `test_custom_aggregator` | カスタムアグリゲータ（例: 中央値）が適用される |
| `test_reproducibility_with_fixed_seed` | 同一シードで同一結果 |

**自己無撞着性テスト**

| テスト名 | 内容 |
|---|---|
| `test_ccm_equals_aggregated_bootstrap` | `ccm()` の出力が `bootstrap()` + aggregator と一致 |
| `test_with_simplex_convenience` | `ccm.with_simplex_projection()` が手動構築と同一結果 |
| `test_with_smap_convenience` | `ccm.with_smap()` が手動構築と同一結果 |

---

### 3.6 `generate/` — データ生成器

各生成器に共通するテスト項目:

| テスト名 | 内容 |
|---|---|
| `test_output_shape` | 出力の形状が `(t_max/dt, D)` |
| `test_time_array` | 時間配列が等間隔で `[0, t_max)` |
| `test_deterministic` | 同一パラメータで同一出力 |
| `test_no_nan_or_inf` | 出力に NaN/Inf がない |

#### Lorenz 固有テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_attractor_boundedness` | 軌道が有界 | Lorenz アトラクタは有界（散逸系） |
| `test_sensitive_dependence` | 初期値の微小変化で軌道が発散 | カオスの定義的性質 |

#### Mackey-Glass 固有テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_positivity` | 正の初期値からは常に正 | 生物学的モデルの物理的制約 |
| `test_chaos_for_large_tau` | `tau > 17` で不規則な振る舞い | 既知のカオス閾値 |

#### Double Pendulum 固有テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_state_dimension` | 出力が4次元 `(θ1, θ2, ω1, ω2)` | 状態空間の次元 |
| `test_to_xy_conversion` | 角度→直交座標変換が幾何学的に正しい | 三角関数による計算 |

---

## 4. テストデータ戦略

### 4.1 合成データの分類

テストで使用するデータを目的に応じて分類する。

```
Deterministic (解析解が存在)
├── 定数系列:       x[t] = c
├── 線形系:         x[t+1] = a*x[t] + b
├── 正弦波:         x[t] = sin(ωt)
└── 等差数列:       x[t] = t * d

Chaotic (統計的性質で検証)
├── Logistic map:   x[t+1] = r*x[t]*(1-x[t])
├── Lorenz:         3次元連続力学系
├── Mackey-Glass:   遅延微分方程式
└── Double Pendulum: ハミルトン系

Stochastic (統計的性質で検証)
├── 白色ノイズ:     x[t] ~ N(0, σ²)
├── AR(1):          x[t] = φ*x[t-1] + ε[t]
└── 因果ペア:       Y[t] = f(X[t-k]) + ε[t]
```

### 4.2 フィクスチャ設計

テストフィクスチャは以下の原則で設計する。

1. **最小サイズの原則**: テストの意図を検証できる最小のデータサイズを使用する。単体テストでは N ≤ 50 を目安とする
2. **決定論的生成**: 乱数を使用する場合は固定シードを使い、再現性を保証する
3. **独立性**: 各テスト関数は他のテストに依存しない
4. **conftest.py での共有**: 複数テストファイルで使うフィクスチャは `conftest.py` に集約する

```python
# tests/conftest.py の設計イメージ

@pytest.fixture
def linear_series():
    """x[t+1] = 0.5 * x[t], x[0] = 1.0, N=50"""
    ...

@pytest.fixture
def sine_wave():
    """x[t] = sin(2π * t / 20), N=100"""
    ...

@pytest.fixture
def logistic_map():
    """x[t+1] = 3.8 * x[t] * (1 - x[t]), x[0] = 0.4, N=200"""
    ...

@pytest.fixture
def causal_pair():
    """X→Y の因果がある2変量系列, N=500"""
    ...

@pytest.fixture
def independent_pair():
    """独立な2変量系列, N=500"""
    ...
```

---

## 5. 許容誤差（Tolerance）の設計

### 5.1 基本方針

数値計算のテストでは、浮動小数点演算の性質を考慮して適切な許容誤差を設定する。

| カテゴリ | 許容誤差 | 使用場面 |
|---|---|---|
| **厳密一致** | `atol=1e-15` | 解析解が存在し、数値誤差が丸め誤差のみの場合 |
| **高精度** | `atol=1e-10` | 逆行列計算など条件数に依存する計算 |
| **統計的** | 傾向の検証（単調性、符号） | 収束性テスト、カオス系での予測精度 |

### 5.2 比較関数の使い分け

```python
# 厳密一致（丸め誤差のみ）
np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)

# 条件数依存の計算
np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

# 統計的傾向
assert np.corrcoef(lib_sizes, correlations)[0, 1] > 0.5  # 正の相関

# 単調性
assert all(a <= b for a, b in zip(values[:-1], values[1:]))
```

---

## 6. テスト実行戦略

### 6.1 pytest マーカー

```python
# 高速テスト（< 1秒）: デフォルトで実行
def test_fast_example(): ...

# 低速テスト（> 5秒）: -m slow で実行
@pytest.mark.slow
def test_convergence_large_dataset(): ...

# GPU 依存テスト
@pytest.mark.gpu
def test_tensor_computation(): ...
```

### 6.2 実行コマンド

```bash
# 通常の開発サイクル（高速テストのみ）
uv run pytest tests/ -m "not slow and not gpu"

# 全テスト
uv run pytest tests/

# 特定モジュール
uv run pytest tests/test_simplex_projection.py -v
```

### 6.3 パラメタライズの活用

同一ロジックを複数データソースで検証する場合は `@pytest.mark.parametrize` を使う。

```python
@pytest.mark.parametrize("e", [1, 2, 3, 5])
@pytest.mark.parametrize("tau", [1, 2, 4])
def test_embedding_output_shape(sine_wave, e, tau):
    result = lagged_embed(sine_wave, tau=tau, e=e)
    assert result.shape == (len(sine_wave) - (e - 1) * tau, e)
```

---

## 7. テストファイル構成

移行後のテストファイル構成:

```
tests/
├── conftest.py                    # 共有フィクスチャ
├── test_embedding.py              # lagged_embed のテスト
├── test_util.py                   # ユーティリティ関数のテスト
│   ├── pairwise_distance
│   ├── dtw
│   ├── autocorrelation
│   └── pad
├── test_simplex_projection.py     # simplex_projection のテスト
├── test_smap.py                   # smap のテスト
├── test_ccm.py                    # ccm / bootstrap / pearson_correlation のテスト
├── test_generate.py               # データ生成器のテスト
└── smoke_test.py                  # エンドツーエンド疎通テスト
```

---

## 8. 移行計画

### Phase 1: 基盤整備

- `conftest.py` を作成し、共有フィクスチャを定義
- pytest マーカーを `pyproject.toml` に登録

### Phase 2: 新テストの実装

各モジュールについて、本設計書のテスト項目を実装する。優先順位は以下の通り:

1. `test_embedding.py` — 最も単純で依存が少ない
2. `test_util.py` — 他モジュールの基盤
3. `test_simplex_projection.py` — コアアルゴリズム
4. `test_smap.py` — コアアルゴリズム
5. `test_ccm.py` — 最も複雑
6. `test_generate.py` — 独立したモジュール

### Phase 3: pyEDM 依存の除去

- 全テストが pyEDM なしで動作することを確認
- `pyproject.toml` の dev dependencies から pyEDM を削除
- CI/CD パイプラインの更新

---

## 9. pyEDM 比較テストの保持について

pyEDM との比較テストは、独立したテストファイル（例: `tests/reference/test_pyedm_comparison.py`）として保持することを推奨する。ただし以下の条件を設ける:

- 通常の `uv run pytest tests/` では実行されない（マーカーまたはディレクトリで分離）
- pyEDM がインストールされている場合のみ実行可能
- リリース前の最終検証やアルゴリズム変更時のリグレッションチェックとして使用

```python
pyedm = pytest.importorskip("pyEDM")

@pytest.mark.reference
def test_simplex_matches_pyedm():
    """リファレンス実装との整合性確認（通常テストには含まない）"""
    ...
```
