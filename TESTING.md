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

---

## Appendix A. テスト項目の理論的妥当性の検証

本節では、セクション3で設計した各テスト項目が EDM の原著論文の理論から正当化できるかを検証する。
各項目について、理論的根拠の強さを以下の3段階で評価する。

- **Strong**: 原著論文または数学的定義から直接導かれる性質。テストとして強く推奨。
- **Moderate**: 理論的に妥当だが、条件付きでのみ成立する、または実装依存の要素がある。条件の明示が必要。
- **Weak**: 直感的には正しそうだが、理論的保証がないか、反例が存在する。テスト設計の見直しが必要。

### 参考文献

以下の原著論文およびリソースを参照した。

1. Sugihara, G. & May, R.M. (1990). "Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series." *Nature*, 344, 734–741. — Simplex projection の原論文
2. Sugihara, G. (1994). "Nonlinear forecasting for the classification of natural time series." *Phil. Trans. R. Soc. Lond. A*, 348, 477–495. — S-Map の原論文
3. Sugihara, G., May, R., Ye, H., et al. (2012). "Detecting Causality in Complex Ecosystems." *Science*, 338, 496–500. — CCM の原論文
4. Takens, F. (1981). "Detecting strange attractors in turbulence." *Lecture Notes in Mathematics*, 898, 366–381. — 埋め込み定理
5. [EDM Algorithms in Depth - Sugihara Lab](https://sugiharalab.github.io/EDM_Documentation/algorithms_in_depth/) — pyEDM/rEDM 公式ドキュメント
6. [Explaining empirical dynamic modelling using verbal, graphical and mathematical approaches](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094587/) — EDM の解説論文 (2024)
7. Cenci, S., Sugihara, G., & Saavedra, S. (2019). "Regularized S-map for inference and forecasting with noisy ecological time series." *Methods in Ecology and Evolution*, 10, 650–660. — 正則化 S-Map

---

### A.1 `simplex_projection` のテスト項目の検証

#### A.1.1 `test_identity_prediction` — **Strong**

> クエリ点がライブラリ点と一致する場合、完全予測が可能

**理論的根拠**: Sugihara & May (1990) の重み付けスキームは `w_i = exp(-d_i / d_min)` である。クエリ点がライブラリ点の1つに正確に一致する場合、`d_min = 0` となる。edmkit の実装（`simplex_projection.py:126`）では `d_min` を `1e-6` にクランプしているため、最近傍の重みは `exp(-0/1e-6) = exp(0) = 1` となり、他の近傍の重みは `exp(-d_i/1e-6) ≈ 0`（`d_i > 0` の場合）となる。したがって予測値は最近傍の Y 値とほぼ一致する。

**注意**: 厳密な一致ではなく、クランプによる近似値となる。許容誤差 `atol=1e-10` 程度が妥当。

#### A.1.2 `test_linear_system_exact` — **Moderate**

> 線形力学系 `x[t+1] = a*x[t]` の予測

**理論的根拠**: Takens の埋め込み定理 [4] により、決定論的力学系は十分な埋め込み次元で状態空間を再構成できる。線形系は1次元（E=1）で完全に再構成可能であり、simplex projection は E+1=2 近傍の重み付き平均で予測する。

**しかし**: Simplex projection は加重平均（凸結合に近い操作）であり、線形回帰ではない。線形系であっても、近傍点の Y 値の加重平均が真の値と一致する保証は、近傍の配置に依存する。例えば、クエリ点が2つの近傍のちょうど中間にあれば線形補間に近づくが、一般にはそうならない。

**修正案**: テストの期待値を「完全予測」ではなく「高精度な予測（RMSE が小さい）」に緩和する。または、ライブラリが十分に密であることを前提条件として明示する。

#### A.1.3 `test_constant_target` — **Strong**

> Y が定数の場合、重みによらず予測値 = 定数

**理論的根拠**: 予測式 `ŷ = Σ(w_i * Y_i) / Σ(w_i)` で `Y_i = c` (定数) のとき、`ŷ = c * Σ(w_i) / Σ(w_i) = c`。これは重みの値に依存せず、純粋に代数的に成立する。

#### A.1.4 `test_prediction_in_target_range` — **Strong（条件付き）**

> 予測値が Y の値域 [min(Y), max(Y)] に収まる

**理論的根拠**: Simplex projection の予測は `ŷ = Σ(w_i * Y_{N_i}) / Σ(w_i)` で、重み `w_i = exp(-d_i/d_min)` は常に正（指数関数の性質）であり、正規化後は和が1になる。これは Y の **E+1 個の近傍の値** の凸結合であるため、予測値はこれら近傍の値の範囲内に収まる。

**重要な注意**: この性質は「全ての Y の値域」ではなく「**選ばれた E+1 個の近傍の Y 値の範囲**」内に収まるという主張が正確である。一般には `min(Y_neighbors) ≤ ŷ ≤ max(Y_neighbors)` であり、`min(Y) ≤ ŷ ≤ max(Y)` はその帰結として成立する。

**修正案**: テスト名と説明を「凸結合であるため、近傍の Y 値の範囲内に予測が収まる」に修正する。

#### A.1.5 `test_neighbor_count_equals_e_plus_1` — **Strong**

> k = E + 1 近傍を使用

**理論的根拠**: Sugihara & May (1990) [1] において、E 次元空間における最小の単体（simplex）を構成するために E+1 個の頂点が必要であることが述べられている。実装でも `k = X.shape[1] + 1`（`simplex_projection.py:118`）で確認できる。これはアルゴリズムの定義そのものである。

#### A.1.6 `test_prediction_improves_with_optimal_e` — **Moderate**

> 最適な埋め込み次元で予測精度が向上する

**理論的根拠**: Sugihara & May (1990) [1] の主要な結果の一つは、埋め込み次元 E を変化させたときの予測精度の変化がカオスとノイズの識別に使えるというものである。カオス系では最適な E が存在し、それを超えると「次元の呪い」により精度が低下する。

**ただし**: 「最適な E」の値はデータと系に依存し、精度が E に対して単調に改善する保証はない。E が小さすぎれば状態空間の再構成が不十分、E が大きすぎれば近傍の距離が増大して予測が劣化する。テストとしては「ある E で精度がピークに達する」ことの確認が適切であり、「E を増やせば常に改善」ではない。

**修正案**: テストを「E の増加に伴い精度が一度改善してから悪化する（逆U字型）」に修正するか、データとパラメータを固定して既知の最適 E で精度改善を確認する。

#### A.1.7 `test_permutation_invariance_of_library` — **Moderate**

> ライブラリ点の順序入れ替えで結果不変

**理論的根拠**: Simplex projection のアルゴリズムは距離に基づく近傍探索であり、数学的には点集合の順序に依存しない。

**ただし**: KDTree の実装では、等距離の近傍が存在する場合にタイブレークが挿入順序に依存する可能性がある。edmkit の実装は `scipy.spatial.KDTree` を使用しており、タイの処理は実装依存である。

**修正案**: テストデータとして等距離点が存在しないケースを選ぶか、タイが発生しないことを前提条件として明記する。

#### A.1.8 `test_accuracy_improves_with_library_size` — **Weak**

> ライブラリサイズ増加に伴い RMSE が非増加傾向

**理論的根拠**: 直感的には、ライブラリが大きいほどアトラクタの被覆が密になり、近傍がより近くなるため予測精度が向上する。EDM の文献 [1,5] でもこの傾向は示唆されている。

**しかし**: Simplex projection 単体では、ライブラリサイズに対する予測精度の単調改善は**理論的に保証されていない**。理由:
- ランダムサンプリングの場合、特定のサンプルでは近傍の構成が悪化する可能性がある
- ライブラリが大きくても、ノイズの多いデータが追加されれば精度は悪化しうる
- 単調性の保証があるのは CCM における収束性の文脈であり、それも期待値（多数のサンプル平均）の話である

**修正案**: 「厳密な単調非増加」ではなく「全体的な傾向として改善」をテストする。具体的には相関係数が正であることを検証するか、最小ライブラリと最大ライブラリの比較に留める。または、このテストを CCM の収束性テストに統合する。

#### A.1.9 `test_noise_sensitivity` — **Moderate**

> ノイズレベル増加に伴い精度が単調劣化

**理論的根拠**: Sugihara & May (1990) [1] の主要な結果の一つは、カオス系ではノイズの増加とともに予測精度が低下するというものである。ノイズが埋め込み空間の近傍関係を乱すため、これは直感的にも理論的にも妥当。

**ただし**: 厳密な単調性は保証されない。特に小さなノイズの違いでは、ランダム性により逆転が起こりうる。テストとしては「十分に異なるノイズレベル間の比較」が適切。

---

### A.2 `smap` のテスト項目の検証

#### A.2.1 `test_theta_zero_equals_ols` — **Strong**

> `theta=0` で OLS と一致

**理論的根拠**: Sugihara (1994) [2] において、S-Map の重み付けは `w = exp(-θ * d / D)` と定義されている。`θ = 0` のとき `w = exp(0) = 1`（全点等重み）となり、これは切片付きの通常最小二乗法（OLS）に帰着する。これは S-Map の定義から直接導かれる性質であり、rEDM のドキュメント [5] でも "the S-map reduces to a type of autoregressive model" と明記されている。

edmkit の実装（`smap.py:140-141`）でも `theta == 0` のとき `weights = np.ones_like(D)` としており、定義と一致する。

#### A.2.2 `test_linear_system_recovery` — **Strong**

> `Y = a*X + b` で `theta=0` のとき完全復元

**理論的根拠**: `theta=0` が OLS と等価であること（A.2.1）から、真のデータ生成過程が `Y = a*X + b`（ノイズなし）の場合、OLS は真のパラメータ `(b, a)` を完全に復元する。これは線形回帰の基本的な性質であり、数値誤差のみ。

**注意**: edmkit の実装では Tikhonov 正則化（`alpha=1e-10`）が適用されるため、係数に微小なバイアスが生じる。テストでは `alpha` の影響を考慮した許容誤差（`atol=1e-8` 程度）が必要。あるいは `alpha=0` を明示的に指定する。

#### A.2.3 `test_identity_prediction` — **Moderate**

> クエリ点がライブラリ点に一致する場合、完全予測

**理論的根拠**: クエリ点がライブラリに含まれている場合でも、S-Map は全ライブラリ点を使った重み付き線形回帰であるため、回帰直線（超平面）がそのデータ点を正確に通る保証はない。これは simplex projection とは異なる。

**ただし**: `theta` が十分大きい場合、クエリ点に近いライブラリ点の重みが支配的になり、局所的にはほぼ完全にフィットする可能性がある。また、`theta=0`（OLS）で E+1 個以下のデータ点の場合（自由度がパラメータ数以上）、完全フィットとなる。

**修正案**: このテストの条件を明確化する。「`theta` が大きく、かつクエリ点がライブラリ点の一つである」または「ライブラリサイズが E+1 以下」のケースに限定する。一般的には S-Map で identity prediction は保証されないことを注記する。

#### A.2.4 `test_constant_target` — **Strong**

> Y が定数の場合、切片項のみで予測

**理論的根拠**: Y が定数 c の場合、重み付き線形回帰の解は切片 = c、他の係数 = 0 となる（正則化の影響が十分小さい場合）。予測値は `[1, q] @ [c, 0, ..., 0]^T = c` となる。

#### A.2.5 `test_theta_increases_locality` — **Strong**

> `theta` 増加に伴い、遠方点の影響が減少

**理論的根拠**: S-Map の重み `w = exp(-θ * d / D)` は `θ` の増加に対して、`d > 0` の点の重みが指数関数的に減衰する。Sugihara (1994) [2] でこれが S-Map の核心的な設計意図であり、EDM の解説論文 [6] でも「as theta increases, predictions become more sensitive to the nonlinear behavior of a system by drawing more heavily on nearby observations」と記述されている。

テストとしては、特定のクエリ点に対して、異なる `theta` で重みを計算し、遠方点の重みが `theta` の増加に伴い減少することを確認する。これはアルゴリズムの定義から直接検証可能。

#### A.2.6 `test_regularization_effect` — **Moderate**

> `alpha` が大きいほど係数が縮小

**理論的根拠**: edmkit の実装は Tikhonov 正則化（リッジ回帰）を使用している。リッジ回帰では正則化パラメータ λ の増加に伴い係数のノルムが縮小することは、凸最適化の標準的な結果として知られている。

**ただし**: Cenci et al. (2019) [7] による正則化 S-Map はより一般的な正則化を提案しており、edmkit の Tikhonov 正則化は Sugihara (1994) の原著にはない拡張である。テストとしては妥当だが、「EDM の理論的性質」ではなく「実装の正則化の性質」のテストであることを区別すべき。

#### A.2.7 `test_intercept_not_regularized` — **Moderate**

> 切片項は正則化されない

**理論的根拠**: これはリッジ回帰のベストプラクティスとして一般的に採用される設計判断であり、EDM 固有の理論ではない。切片を正則化しないことで、予測のバイアスを防ぐ。edmkit の実装（`smap.py:157`、`eye[0, 0] = 0`）でこの設計判断が確認できる。

テストとしては実装の正しさの検証として妥当だが、理論的根拠は「リッジ回帰の標準的慣行」である。

#### A.2.8 `test_theta_zero_vs_nonzero_on_linear` — **Strong**

> 線形系ではどの `theta` でも同精度

**理論的根拠**: Sugihara (1994) [2] および rEDM チュートリアル [5] で明確に述べられている。線形系では状態空間の全ての点が同一の線形写像に従うため、局所化（`theta > 0`）の効果はなく、グローバル線形回帰と等価になるべきである。

**ただし**: 正則化の影響により、`theta > 0` でデータ点の有効数が減少する（局所化により事実上のサンプルサイズが小さくなる）ため、`theta` が極端に大きい場合は正則化の影響が相対的に大きくなり、精度が若干低下する可能性がある。テストでは中程度の `theta` 値で検証するのが安全。

**修正案**: 「厳密に同精度」ではなく「精度の差が小さい」とする。または正則化による微小な差を許容する `atol` を設定する。

#### A.2.9 `test_theta_nonzero_better_on_nonlinear` — **Strong**

> 非線形系では `theta > 0` が `theta = 0` より高精度

**理論的根拠**: これは S-Map の存在意義そのものであり、Sugihara (1994) [2] の主要な主張である。非線形（状態依存的）力学系では、状態空間の各位置で異なる局所線形写像が必要であり、`theta > 0` による局所化がこれを実現する。rEDM チュートリアル [5] でも「if forecast skill increases for θ > 0, then the results are suggestive of nonlinear dynamics」と述べている。

**ただし**: 最適な `theta` の値はデータに依存する。`theta` が大きすぎると過学習に近い挙動を示す可能性がある。テストでは、Logistic map や Lorenz のような既知の非線形系で、`theta=0` と中程度の `theta`（例: 2-4）を比較するのが適切。

#### A.2.10 `test_permutation_invariance_of_library` — **Strong**

> ライブラリ順序に依存しない

**理論的根拠**: S-Map は全ライブラリ点を使った重み付き線形回帰であり、数学的に点の順序に不変である（行列演算 `X^T W X` は行の並び順に依存しない）。simplex projection とは異なり、KDTree を使用しないため、タイブレークの問題も発生しない。

---

### A.3 `ccm` のテスト項目の検証

#### A.3.1 `test_convergence_with_known_causality` — **Strong**

> X→Y の因果がある系で、ライブラリサイズ増加に伴い相関が収束（増加傾向）

**理論的根拠**: これは CCM の定義的性質そのものである。Sugihara et al. (2012) [3] において、"convergent cross mapping" の名称自体がこの収束性に由来する。原論文では、X が Y に因果的影響を与える場合、Y のアトラクタから X を cross-map で復元する精度がライブラリサイズの増加に伴い向上（収束）することが示されている。

メカニズムとしては、ライブラリが大きいほど再構成されたアトラクタの被覆が密になり、近傍推定の精度が向上するため、cross-map の精度が向上する。

**注意**: 収束は**期待値**に対する性質であり、個々のブートストラップサンプルでは非単調な振る舞いが生じうる。テストでは `n_samples` を十分に取った上で平均相関を検証すべき。また、収束は「厳密な単調増加」ではなく「漸近的な改善傾向」であることに注意。

**テスト設計の推奨**: ライブラリサイズの最小値と最大値での相関を比較し、最大値の方が有意に高いことを検証する形式が安定的。

#### A.3.2 `test_no_convergence_without_causality` — **Moderate**

> 独立な系列では収束しない

**理論的根拠**: Sugihara et al. (2012) [3] では、因果関係がない場合に cross-map の精度がライブラリサイズに対して収束しないことが示されている。独立な系列間では、Y のアトラクタに X の情報が含まれないため、cross-map は系統的に改善しない。

**ただし、重要な制約がある**:
1. **季節性・共有外力**: 2つの独立な系列が共通の外力（季節性など）に駆動されている場合、偽の収束が生じうることが Sugihara et al. (2012) 自身で認められている。テストでは「共有外力がない、純粋に独立な系列」を使用する必要がある。
2. **有限サンプル効果**: 短い時系列では、偶然の相関により偽の収束パターンが出現する可能性がある。

**修正案**: テストデータとして、異なる乱数シードで独立に生成した2つの系列を使用する。相関値が系統的に増加しないこと（Spearman 相関が有意でない、または最大ライブラリでの相関が低い）を検証する。「収束しない」の判定基準を明確にする。

#### A.3.3 `test_ccm_equals_aggregated_bootstrap` — **Strong**

> `ccm()` の出力が `bootstrap()` + aggregator と一致

**理論的根拠**: これはアルゴリズムのテストではなく、実装の自己無撞着性のテストである。`ccm.py:233-245` のコードを見ると、`ccm()` は内部で `bootstrap()` を呼び出し、その結果に `aggregator` を適用している。したがってこの一致は実装の定義から保証される。

---

### A.4 テスト項目の修正・追加推奨

上記の検証に基づき、以下の修正と追加を推奨する。

#### A.4.1 修正が必要な項目

| テスト | 現在の記述 | 問題 | 推奨修正 |
|---|---|---|---|
| `test_linear_system_exact` (simplex) | 線形系で完全予測 | Simplex は線形回帰ではなく加重平均であり、完全予測は保証されない | 「高精度な予測」に緩和。RMSE < ε のように閾値ベースにする |
| `test_prediction_in_target_range` (simplex) | Y の値域に収まる | 正確には E+1 近傍の Y 値の凸結合 | 正しいが、テスト記述を「近傍 Y 値の凸結合」と明記する |
| `test_prediction_improves_with_optimal_e` (simplex) | 最適 E で精度向上 | 単調改善ではなく逆U字型 | 「複数の E で最良のものが E=1 より優れる」に修正 |
| `test_accuracy_improves_with_library_size` (simplex) | 非増加傾向 | simplex 単体では単調改善の理論的保証なし | CCM の収束性テストに統合するか、傾向の統計検定に変更 |
| `test_identity_prediction` (smap) | 完全予測 | S-Map は全点回帰であり、通過保証なし | 条件を限定（theta が大きい場合、またはライブラリ ≤ E+1）するか削除 |
| `test_theta_zero_vs_nonzero_on_linear` (smap) | 同精度 | 正則化の影響で微差が出る | 許容誤差を設けるか、正則化なし (`alpha=0`) で検証 |
| `test_no_convergence_without_causality` (ccm) | 収束しない | 偽陽性のリスクはゼロではない | 「最大ライブラリでの相関が低い」に緩和 |

#### A.4.2 追加を推奨する項目

| テスト | 内容 | 根拠 |
|---|---|---|
| `test_simplex_forecast_decay` (simplex) | 予測ステップ Tp の増加に伴い精度が低下 | Sugihara & May (1990) の主要結果: カオス系では予測ホライズンの増加に伴い精度が指数関数的に低下する。ただし edmkit は Tp を明示的に扱っていないため、ユーザー側のデータ構成で検証する |
| `test_smap_coefficients_state_dependent` (smap) | `theta > 0` で回帰係数がクエリ点ごとに異なる | Sugihara (1994) の核心的な主張: S-Map の係数は状態空間の位置に依存する |
| `test_smap_weight_formula` (smap) | 重み `exp(-θd/D)` が正しく計算される | 原著論文の式と実装の直接的な照合 |
| `test_ccm_asymmetry` (ccm) | 一方向因果 X→Y で、Y→X の cross-map 精度が X→Y より高い | Sugihara et al. (2012) の Figure 1: 「X causes Y」のとき「Y xmap X」が高精度になる（**方向に注意: 因果の逆方向に cross-map する**）|
| `test_ccm_bidirectional` (ccm) | 双方向因果の系で両方向の cross-map が収束する | Sugihara et al. (2012) の Figure 3B に相当 |

#### A.4.3 edmkit 実装固有の注意点

原著論文の理論とは別に、edmkit の実装には以下の固有の設計判断があり、テストで考慮すべきである。

1. **d_min のクランプ** (`simplex_projection.py:126`): `d_min = np.fmax(distances.min(...), 1e-6)` — 距離が 0 の場合の数値安定性のため。原著にはない実装上の処理。
2. **正則化のスケーリング** (`smap.py:158-159`): 正則化項が `alpha * trace(X^T W X)` でスケールされている。これは Cenci et al. (2019) [7] の手法に近いが、原著 Sugihara (1994) にはない。
3. **S-Map の d_mean のクランプ** (`smap.py:143`): `d_mean = np.maximum(D.mean(...), 1e-6)` — 全距離が 0 の退化ケースへの対応。
4. **CCM のグローバル RNG** (`ccm.py:11`): `rng = np.random.default_rng(42)` — モジュールレベルの固定シードにより再現性を確保しているが、テスト間の独立性に影響する可能性がある。
