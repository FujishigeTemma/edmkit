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

### 2.1 「理論的な主張」と「数値保証」の区別

数理計算ライブラリのテスト設計において、以下の区別が極めて重要である。

- **数値保証（Numerical guarantee）**: アルゴリズムの定義・代数的性質から、特定の入力に対して出力が厳密に（または浮動小数点誤差の範囲で）決定される性質。テストとして直接検証可能。
  - 例: 「Y が定数 c のとき、simplex projection の予測値は c に等しい」（代数的恒等式）
  - 例: 「θ=0 のとき、S-Map は OLS と等価である」（アルゴリズムの定義）

- **理論的主張（Theoretical claim）**: 論文中で「XXであれば、YYであるはず」と主張されている性質。数値テストに落とし込むには、具体的なデータ・パラメータ設定において達成すべき閾値の特定が必要。
  - 例: 「カオス系では最適な E で予測精度が向上する」（Sugihara & May, 1990）
  - 例: 「因果関係がある場合、CCM は収束する」（Sugihara et al., 2012）

理論的主張をテストに変換する際は、**具体的な入力設定と達成すべき数値閾値**を明示する必要がある。「理論的に正しいはず」ではテストにならない。

### 2.2 数理計算ライブラリにおけるテスト手法

| テスト手法 | 説明 | 保証の強さ | 適用箇所 |
|---|---|---|---|
| **解析解との比較** | 理論的に答えが導出できるケースでの厳密な検証 | 数値保証 | 線形系、既知の不動点、自明なケース |
| **代数的不変量の検証** | アルゴリズムの定義から直接導かれる性質 | 数値保証 | 対称性、凸結合性、境界条件 |
| **退化ケース（degenerate case）** | 極端な入力での正しい挙動の検証 | 数値保証 | 定数時系列、次元1、サンプル数最小 |
| **自己無撞着性テスト** | 同一結果を別経路で計算し一致を確認 | 数値保証 | バッチ vs 個別、異なるコードパス |
| **収束傾向テスト** | パラメータ変化に伴う漸近的振る舞いの検証 | 条件付き保証 | 固定データ・パラメータでの傾向確認 |
| **摂動テスト** | 微小な入力変化に対する出力の安定性の検証 | 条件付き保証 | 数値安定性、条件数 |
| **エッジケース・エラーハンドリング** | 不正入力に対する適切なエラーの検証 | 数値保証 | 型チェック、次元チェック、値域チェック |

### 2.3 テスト分類

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

#### 3.1.1 解析解テスト（数値保証）

| テスト名 | 内容 | 期待値 | 許容誤差 |
|---|---|---|---|
| `test_identity_embedding` | `e=1, tau=1` ではリシェイプのみ | `x.reshape(-1, 1)` と一致 | 厳密一致 |
| `test_known_values` | 小さな配列で手計算と照合 | 各行が `[x[i+(e-1)*tau], ..., x[i]]` | 厳密一致 |
| `test_output_shape` | 任意の `(N, tau, e)` に対して出力形状を検証 | `(N - (e-1)*tau, e)` | 厳密一致 |
| `test_linear_sequence` | 等差数列の入力で各行の差分が `tau` 刻み | 各行内の隣接要素の差が `tau * d` | 厳密一致 |

**根拠**: いずれも `lagged_embed` の定義から代数的に導かれる性質であり、浮動小数点演算を含まない（整数インデクシングのみ）ため厳密一致が保証される。

#### 3.1.2 代数的性質テスト（数値保証）

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_no_information_loss` | 出力の全要素が入力に含まれている | スライス操作の定義 |
| `test_tau1_e1_preserves_values` | `tau=1, e=1` で元の値がそのまま保持される | 退化ケース |
| `test_sliding_window_consistency` | 連続する行が1ステップずつスライドしている | スライディングウィンドウの定義 |

#### 3.1.3 エラーハンドリング（数値保証）

| テスト名 | 条件 | 期待 |
|---|---|---|
| `test_rejects_2d_input` | 2D配列を渡す | `ValueError` |
| `test_rejects_zero_tau` | `tau=0` | `ValueError` |
| `test_rejects_zero_e` | `e=0` | `ValueError` |
| `test_rejects_insufficient_length` | `(e-1)*tau >= len(x)` | `ValueError` |

---

### 3.2 `util.py`

#### 3.2.1 `pairwise_distance` / `pairwise_distance_np`

**解析解テスト（数値保証）**

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_known_2d_distances` | 手計算可能な2-3点の二乗距離 | `atol=1e-15` |
| `test_identity_distance_zero_diagonal` | `A=B` のとき対角要素が0 | `atol=1e-15` |
| `test_single_point` | 1点同士の距離は差の二乗和 | `atol=1e-15` |

**代数的性質テスト（数値保証）**

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_non_negativity` | 全要素 ≥ 0 | `clamp(min_=0)` による実装保証 |
| `test_self_distance_diagonal_zero` | `D(A, A)` の対角は0 | 二乗距離の定義 |
| `test_symmetry` | `D(A, B) = D(B, A)^T` | 二乗ユークリッド距離の対称性 |
| `test_triangle_inequality` | `sqrt(D[i,k]) ≤ sqrt(D[i,j]) + sqrt(D[j,k])` | ユークリッド距離の距離の公理 |
| `test_translation_invariance` | `D(A+c, B+c) = D(A, B)` | ユークリッド距離の平行移動不変性 |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 | 実装の自己無撞着性 |

**NumPy / Tensor 一致テスト（数値保証）**

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_numpy_tensor_agreement` | 同一入力に対して `pairwise_distance` と `pairwise_distance_np` が同一結果 | `atol=1e-6`（float32 vs float64） |

#### 3.2.2 `dtw`

**解析解テスト（数値保証）**

| テスト名 | 内容 | 期待値 | 根拠 |
|---|---|---|---|
| `test_identical_sequences` | 同一系列 | 0 | DTW コスト行列の対角パスが最適 |
| `test_single_element` | 1要素同士 | ユークリッド距離 | DPテーブルが1セルのみ |
| `test_known_small_case` | 2-3要素で手計算 | 既知のDP結果 | DP漸化式の手計算で検証可能 |

**代数的性質テスト（数値保証）**

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_non_negativity` | DTW距離 ≥ 0 | コスト関数が非負 |
| `test_identity_of_indiscernibles` | 同一系列で D=0 | 対角パスのコストが0 |
| `test_symmetry` | `dtw(A, B) = dtw(B, A)` | ワーピングパスの可逆性 |

#### 3.2.3 `autocorrelation`

**解析解テスト（数値保証）**

| テスト名 | 内容 | 期待値 | 許容誤差 | 根拠 |
|---|---|---|---|---|
| `test_lag_zero_is_one` | ラグ0の自己相関 | 1.0 | `atol=1e-15` | 自己相関の正規化の定義: `acf[0] = var(x)/var(x) = 1` |
| `test_sine_wave` | `sin(2πt/T)` で周期的パターン | ラグ `T/2` で ≈ -1 | `atol=0.05` | 正弦波の自己相関は `cos(2πτ/T)` |

**退化ケース（数値保証）**

| テスト名 | 内容 | 期待値 | 根拠 |
|---|---|---|---|
| `test_constant_signal` | 定数信号 | NaN または 0/0 の適切な処理 | 分散0での除算 |

**統計的テスト（条件付き保証）**

| テスト名 | 内容 | 期待値 | 根拠 |
|---|---|---|---|
| `test_white_noise` | 長い白色ノイズ（N=10000, seed固定） | ラグ>0 で `\|acf\| < 0.05` | 白色ノイズの自己相関は `1/√N` のオーダー。N=10000 で `1/√N ≈ 0.01` |

**注意**: 白色ノイズのテストでは `N=10000` 程度が必要。`N=100` では `1/√N ≈ 0.1` であり、閾値の設定が難しい。シード固定で決定論的に実行する。

#### 3.2.4 `pad`

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_same_dimensions_no_padding` | 同一サイズの配列はパディング不要 | 定義 |
| `test_mixed_dimensions_zero_fill` | 異なるサイズの配列がゼロ埋めされる | 定義 |
| `test_output_shape` | 出力形状が `(B, L, max_D)` | 定義 |

---

### 3.3 `simplex_projection.py` — `simplex_projection()`

#### 参考文献

- Sugihara, G. & May, R.M. (1990). "Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series." *Nature*, 344, 734–741.
- Takens, F. (1981). "Detecting strange attractors in turbulence." *Lecture Notes in Mathematics*, 898, 366–381.

#### 3.3.1 解析解テスト（数値保証）

| テスト名 | 内容 | 期待値 | 許容誤差 | 根拠 |
|---|---|---|---|---|
| `test_identity_prediction` | クエリ点がライブラリ点と一致する場合 | 対応する Y 値に一致 | `atol=1e-10` | 重み `w = exp(-d/d_min)` で `d_min → 0` のとき最近傍に重みが集中。実装上は `d_min` を `1e-6` にクランプするため、他の近傍（`d_i > 0`）の重みは `exp(-d_i/1e-6) ≈ 0` となり、予測値は最近傍の Y 値にほぼ一致する。 |
| `test_constant_target` | Y が定数 c の場合 | c | `atol=1e-15` | `ŷ = Σ(w_i * c) / Σ(w_i) = c`。重みの値に依存せず代数的に成立する恒等式。 |

#### 3.3.2 代数的性質テスト（数値保証）

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_prediction_in_target_range` | 予測値が Y の値域 `[min(Y), max(Y)]` に収まる | 重み `w_i = exp(-d_i/d_min) > 0` は常に正。正規化後は `Σw_i = 1`（凸結合）。選ばれた E+1 個の近傍の Y 値の凸結合であるため、`min(Y_neighbors) ≤ ŷ ≤ max(Y_neighbors)` が成立し、`[min(Y), max(Y)]` 内に収まることはその帰結。 |
| `test_neighbor_count_equals_e_plus_1` | `k = E + 1` 近傍を使用 | Sugihara & May (1990): E 次元空間における最小の単体（simplex）の頂点数。アルゴリズムの定義そのもの。実装: `k = X.shape[1] + 1`。 |

#### 3.3.3 予測精度テスト（条件付き保証）

以下のテストは理論的主張を具体的なデータ・パラメータ設定で数値的に検証するものである。閾値は固定シード・固定パラメータのもとで経験的に安定した値として設定する。

| テスト名 | 内容 | 設定 | 閾値 | 根拠 |
|---|---|---|---|---|
| `test_linear_system_high_accuracy` | 線形力学系 `x[t+1] = 0.5*x[t]` の予測 | E=1, tau=1, N=50, lib=40 | RMSE < 0.01 | Takens の埋め込み定理により線形系は E=1 で再構成可能。ただし simplex projection は加重平均であり線形回帰ではないため、「完全予測」は保証されない。ライブラリが十分密であれば高精度が期待できる。 |
| `test_optimal_e_improves_over_e1` | 複数の E で最良のものが E=1 より優れる | Logistic map (r=3.8, x0=0.4, N=300), E∈{1,2,3,4,5}, tau=1 | `min_E>1(RMSE) < RMSE_E1` | Sugihara & May (1990) の主要結果。カオス系ではアトラクタの次元に応じた最適な E が存在する。Logistic map の理論的な埋め込み次元は2-3。E に対する精度は逆U字型を描く（小さすぎれば再構成不十分、大きすぎれば次元の呪い）。 |
| `test_noise_degrades_accuracy` | ノイズ増加で精度が劣化 | Logistic map, ノイズ σ ∈ {0, 0.05, 0.2} | `RMSE(σ=0) < RMSE(σ=0.2)` | Sugihara & May (1990): ノイズが埋め込み空間の近傍関係を乱すため精度が低下。十分に異なるノイズレベル間で比較。中間レベルの厳密な単調性は保証しない。 |

**注意（`test_linear_system_high_accuracy` について）**: Simplex projection は加重平均（凸結合に近い操作）であり、線形回帰ではない。線形系であっても近傍点の Y 値の加重平均が真の値と完全に一致する保証は、近傍の配置に依存する。「完全予測」ではなく「高精度な予測（RMSE < ε）」として検証する。

**注意（`test_optimal_e_improves_over_e1` について）**: 「E を増やせば常に精度が改善する」のではなく、「複数の E の中で E=1 より優れるものが存在する」ことを検証する。これは Sugihara & May (1990) の主張（最適な E の存在）を、単調改善の誤った仮定なしにテスト化したものである。

#### 3.3.4 自己無撞着性テスト（数値保証）

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_batch_vs_individual` | 3Dバッチ入力と2D個別入力の結果一致 | `atol=1e-15` |
| `test_numpy_vs_tensor` | `use_tensor=True/False` で同一結果 | `atol=1e-5`（float32 vs float64 の差異） |

#### 3.3.5 退化ケース（数値保証）

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_single_query_point` | クエリ点1つでも正常動作 | 境界条件 |
| `test_minimum_library_size` | ライブラリサイズ = E + 1（最小近傍数） | k = E+1 であるため最小のライブラリ |

#### 3.3.6 順序不変性テスト（条件付き保証）

| テスト名 | 内容 | 許容誤差 | 注意事項 |
|---|---|---|---|
| `test_permutation_invariance_of_library` | ライブラリ点の順序入れ替えで結果不変 | `atol=1e-12` | テストデータとして等距離の近傍が存在しないケースを使用すること。KDTree のタイブレークが挿入順序に依存する可能性があるため、タイが発生するデータでは保証されない。 |

---

### 3.4 `smap.py` — `smap()`

#### 参考文献

- Sugihara, G. (1994). "Nonlinear forecasting for the classification of natural time series." *Phil. Trans. R. Soc. Lond. A*, 348, 477–495.
- Cenci, S., Sugihara, G., & Saavedra, S. (2019). "Regularized S-map for inference and forecasting with noisy ecological time series." *Methods in Ecology and Evolution*, 10, 650–660.

#### 3.4.1 解析解テスト（数値保証）

| テスト名 | 内容 | 期待値 | 許容誤差 | 根拠 |
|---|---|---|---|---|
| `test_theta_zero_equals_ols` | `theta=0` で OLS と一致 | NumPy `lstsq` の解と一致 | `atol=1e-8` | Sugihara (1994): `θ=0` のとき `w = exp(0) = 1`（全点等重み）→ 切片付き OLS に帰着。実装: `smap.py:140-141` で `weights = np.ones_like(D)` を確認。Tikhonov 正則化（`alpha=1e-10`）による微小なバイアスを許容誤差に反映。 |
| `test_linear_system_recovery` | `Y = a*X + b`（ノイズなし）で `theta=0, alpha=0` のとき完全復元 | 係数 `(b, a)` を完全復元 | `atol=1e-10` | OLS の基本的な性質。ノイズなしの線形関係は OLS で完全にフィットする。`alpha=0` を明示的に指定し、正則化バイアスを排除する。 |
| `test_constant_target` | Y が定数 c の場合 | 予測値 = c | `atol=1e-8` | 重み付き線形回帰の解は切片 = c、他の係数 ≈ 0 となり、予測値は c。正則化の微小な影響を許容。 |

#### 3.4.2 代数的性質テスト（数値保証）

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_theta_increases_locality` | `theta` 増加に伴い、遠方点の重みが減少する | S-Map の重み `w = exp(-θ * d / D̄)` の定義。`θ` 増加で `d > 0` の点の重みが指数関数的に減衰する。Sugihara (1994) の核心的な設計意図。テストでは特定のクエリ点に対し、異なる `theta` で重みを直接計算して比較。 |
| `test_permutation_invariance_of_library` | ライブラリ順序に依存しない | S-Map は全ライブラリ点を使った重み付き線形回帰。行列演算 `X^T W X` は行の並び順に依存しない。KDTree 不使用のためタイブレーク問題なし。 |
| `test_smap_coefficients_state_dependent` | `theta > 0` で回帰係数がクエリ点ごとに異なる | Sugihara (1994) の核心的主張: S-Map の係数は状態空間の位置に依存する。`theta=0` では全クエリ点で同一係数（OLS）、`theta > 0` では各クエリ点の局所近傍に基づく異なる係数が得られる。 |

#### 3.4.3 実装固有の性質テスト（数値保証）

以下は edmkit の実装における設計判断の検証であり、EDM の理論そのものではなく正則化手法の性質である。

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_regularization_effect` | `alpha` 増加に伴い係数ノルムが縮小 | Tikhonov 正則化（リッジ回帰）の標準的性質。Cenci et al. (2019) に基づく実装拡張。凸最適化の結果として保証される。 |
| `test_intercept_not_regularized` | 切片項は正則化されない | リッジ回帰の標準的慣行。実装: `smap.py:157` で `eye[0, 0] = 0`。`alpha` を大きくしても切片が不当に縮小しないことを検証。 |

#### 3.4.4 theta に関する比較テスト（条件付き保証）

| テスト名 | 内容 | 設定 | 閾値 | 根拠 |
|---|---|---|---|---|
| `test_theta_zero_vs_nonzero_on_linear` | 線形系ではどの `theta` でも同程度の精度 | `Y = 2X + 1`, N=50, `theta` ∈ {0, 2, 4} | RMSE の差 < 0.01 | Sugihara (1994): 線形系では全点が同一超平面上にあるため、局所化の効果はない。ただし `theta > 0` でデータ点の有効数が減少し、正則化の相対的影響が増すため、微小な精度差は許容する。極端に大きい `theta` は除外する。 |
| `test_theta_nonzero_better_on_nonlinear` | 非線形系では `theta > 0` が `theta = 0` より高精度 | Logistic map (r=3.8, x0=0.4, N=300), E=3, tau=1, `theta` ∈ {0, 2, 4} | `min(RMSE_θ>0) < RMSE_θ=0` | Sugihara (1994) の主要結果: 非線形力学系では局所化が精度を改善する。テストでは「theta>0 の中で最良のものが theta=0 より優れる」ことを検証（特定の theta 値での優位性ではない）。 |

#### 3.4.5 数値安定性テスト（数値保証）

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_ill_conditioned_library` | ほぼ共線的なライブラリ点でも正則化により安定 | Tikhonov 正則化が特異行列を回避。出力に NaN/Inf が含まれないことを検証。 |
| `test_large_theta_stability` | `theta` が大きくても NaN/Inf が発生しない | `d_mean` のクランプ（`smap.py:143`）による数値安定性。`theta=100` 等の極端な値で検証。 |

#### 3.4.6 自己無撞着性テスト（数値保証）

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_batch_vs_individual` | バッチ処理と個別処理の一致 | `atol=1e-12` |

#### 3.4.7 S-Map `identity_prediction` の扱い

アルファ版で検討されていた `test_identity_prediction`（クエリ点がライブラリ点に一致する場合の完全予測）は、S-Map では**一般に保証されない**。S-Map は全ライブラリ点を使った重み付き線形回帰であるため、回帰超平面が特定のデータ点を正確に通る保証はない。以下の限定的なケースでのみ成立する:

- `theta` が十分大きく、かつ正則化が十分小さい場合（局所的にほぼ完全フィット）
- ライブラリサイズが E+1 以下の場合（自由度がパラメータ数以上で完全フィット）

これらの条件は実用上のエッジケースであり、独立したテストとしてではなく、数値安定性テストの一部として扱う。

---

### 3.5 `ccm.py`

#### 参考文献

- Sugihara, G., May, R., Ye, H., et al. (2012). "Detecting Causality in Complex Ecosystems." *Science*, 338, 496–500.

#### 3.5.1 `pearson_correlation`（数値保証）

**解析解テスト**

| テスト名 | 内容 | 期待値 | 許容誤差 |
|---|---|---|---|
| `test_perfect_positive` | `Y = aX + b (a > 0)` | 1.0 | `atol=1e-10` |
| `test_perfect_negative` | `Y = -aX + b` | -1.0 | `atol=1e-10` |
| `test_uncorrelated` | 直交する正弦波 `sin(t)` と `cos(t)` | ≈ 0 | `atol=0.05` |
| `test_self_correlation` | `X` と `X` | 1.0 | `atol=1e-10` |

**注意（`test_uncorrelated`）**: `sin(t)` と `cos(t)` は理論的に直交するが、有限離散サンプルでは完全にゼロにならない。サンプル数が周期の整数倍であれば `atol=1e-10` で検証可能。そうでない場合は `atol=0.05` 程度。

**代数的性質テスト**

| テスト名 | 検証する性質 | 根拠 |
|---|---|---|
| `test_range` | -1 ≤ corr ≤ 1 | Cauchy-Schwarz の不等式 |
| `test_symmetry` | `corr(X, Y) = corr(Y, X)` | 定義の対称性 |
| `test_invariance_to_positive_scale` | `corr(aX+b, cY+d) = corr(X, Y)` (`a,c > 0`) | 相関係数の正アフィン変換不変性 |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 | 実装の自己無撞着性 |

**注意（アフィン変換不変性）**: アルファ版では `corr(aX+b, cY+d) = sign(ac) * corr(X, Y)` としていたが、edmkit の `pearson_correlation` は `(B, L, E')` 形状の入力に対して `mean(axis=-1)` で集約する独自仕様であるため、標準的な Pearson 相関と挙動が異なる場合がある。テストでは `a, c > 0` の場合（符号不変）のみ検証する。

#### 3.5.2 `bootstrap` / `ccm`

**収束性テスト（条件付き保証）**

CCM の本質的な検証。理論的主張を具体的な設定で数値的に検証する。

| テスト名 | 内容 | 設定 | 閾値 | 根拠 |
|---|---|---|---|---|
| `test_convergence_with_known_causality` | X→Y の因果がある系で、ライブラリサイズ増加に伴い相関が改善傾向 | 結合ロジスティック写像 (`rx=3.8, ry=3.5, Bxy=0.02`, N=1000), E=2, tau=1, n_samples=50 | `corr(L_max) > corr(L_min) + 0.1` | Sugihara et al. (2012) の定義的性質。ただし収束は**期待値**に対する性質であり、個々のサンプルでは非単調な振る舞いが生じうる。テストでは `n_samples` を十分に取った上で平均相関を比較。最小と最大のライブラリサイズ間の比較に限定し、厳密な単調増加は要求しない。 |
| `test_no_convergence_without_causality` | 独立な系列では収束的な改善が見られない | 異なるシードの独立ロジスティック写像2本, N=1000, E=2, tau=1, n_samples=50 | `corr(L_max) < 0.3` | Sugihara et al. (2012): 因果がなければ cross-map 精度が系統的に改善しない。「収束しない」の厳密な判定は困難なため、最大ライブラリでの平均相関が低いことで検証。共有外力がない純粋に独立な系列を使用する。 |
| `test_ccm_asymmetry` | 一方向因果 X→Y で、Y xmap X の相関が X xmap Y より高い | 結合ロジスティック写像（一方向, `Bxy=0.02, Byx=0`）, N=1000, 最大ライブラリサイズ | `corr(Y_xmap_X) > corr(X_xmap_Y) + 0.05` | Sugihara et al. (2012) Figure 1: 「X causes Y」のとき「Y xmap X」が高精度になる。因果の方向と cross-map の方向が逆であることに注意。 |

**注意（収束テストの閾値設定）**: 上記の閾値は固定シード・固定パラメータでの経験的な値である。テスト実装時に実際に複数回実行して安定性を確認し、必要に応じて調整すること。`n_samples` が小さすぎるとブートストラップの分散が大きくなり、テストが不安定になる。

**構成要素のテスト（数値保証）**

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_custom_sampler` | カスタムサンプラーが正しく呼ばれる | API 契約 |
| `test_custom_aggregator` | カスタムアグリゲータ（例: 中央値）が適用される | API 契約 |
| `test_reproducibility_with_fixed_seed` | 同一シードで同一結果 | 決定論的実行の保証 |

**自己無撞着性テスト（数値保証）**

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_ccm_equals_aggregated_bootstrap` | `ccm()` の出力が `bootstrap()` + aggregator と一致 | `atol=1e-15` |
| `test_with_simplex_convenience` | `ccm.with_simplex_projection()` が手動構築と同一結果 | `atol=1e-12` |
| `test_with_smap_convenience` | `ccm.with_smap()` が手動構築と同一結果 | `atol=1e-12` |

**`test_ccm_equals_aggregated_bootstrap` の根拠**: `ccm()` は内部で `bootstrap()` を呼び出し、その結果に `aggregator` を適用している（`ccm.py:233-245`）。したがってこの一致は実装の定義から保証される。

---

### 3.6 `generate/` — データ生成器

各生成器に共通するテスト項目（数値保証）:

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_output_shape` | 出力の形状が `(len(t), D)` | 関数の仕様 |
| `test_time_array` | 時間配列が等間隔で `[0, t_max)` | `np.arange` の性質 |
| `test_deterministic` | 同一パラメータで同一出力 | 決定論的 ODE ソルバー |
| `test_no_nan_or_inf` | 出力に NaN/Inf がない | 数値安定性 |

#### Lorenz 固有テスト

| テスト名 | 内容 | 設定 | 閾値 | 根拠 |
|---|---|---|---|---|
| `test_attractor_boundedness` | 軌道が有界 | 標準パラメータ (σ=10, ρ=28, β=8/3), dt=0.01, t_max=50 | 各座標の絶対値 < 100 | Lorenz アトラクタは散逸系であり、標準パラメータでの軌道は `\|x\| < 25, \|y\| < 30, \|z\| < 55` 程度に収まる。余裕を持った閾値として 100 を使用。 |
| `test_sensitive_dependence` | 初期値の微小変化で軌道が発散 | 初期値を `1e-10` だけ変更, t_max=30 | 最終的な差のノルム > 1.0 | カオスの定義的性質（正のリアプノフ指数）。Lorenz 系の最大リアプノフ指数は約 0.9 であり、`e^(0.9*30) ≫ 1` なので十分に発散する。 |

#### Mackey-Glass 固有テスト

| テスト名 | 内容 | 設定 | 閾値 | 根拠 |
|---|---|---|---|---|
| `test_positivity` | 正の初期値からは常に正 | x0=0.9, 標準パラメータ | 全要素 > 0 | Mackey-Glass 方程式 `dx/dt = β*x_τ/(1+x_τ^n) - γ*x` は正の初期値に対して正の解を持つ（生物学的モデルの物理的制約）。 |

**注意（カオス閾値テスト）**: アルファ版では `test_chaos_for_large_tau` (tau > 17 でカオス) を検討していたが、「カオス的な振る舞い」を数値テストで厳密に判定するには、リアプノフ指数の計算など追加の実装が必要であり、データ生成器のテストの範囲を超える。このテストは省略する。

#### Double Pendulum 固有テスト

| テスト名 | 内容 | 根拠 |
|---|---|---|
| `test_state_dimension` | 出力が4次元 `(θ1, θ2, ω1, ω2)` | 状態空間の定義 |
| `test_to_xy_conversion` | `to_xy` で角度→直交座標変換が幾何学的に正しい | `x1 = L1*sin(θ1)`, `y1 = -L1*cos(θ1)` 等。手計算で検証可能な既知の角度（0, π/2, π）で検証する。 |

---

## 4. テストデータ戦略

### 4.1 合成データの分類

テストで使用するデータを目的に応じて分類する。

```
Deterministic (解析解が存在 → 数値保証テスト)
├── 定数系列:       x[t] = c
├── 線形系:         x[t+1] = a*x[t] + b
├── 正弦波:         x[t] = sin(ωt)
└── 等差数列:       x[t] = t * d

Chaotic (固定パラメータでの条件付き保証テスト)
├── Logistic map:   x[t+1] = r*x[t]*(1-x[t])  (r=3.8, x0=0.4)
├── Lorenz:         3次元連続力学系            (σ=10, ρ=28, β=8/3)
├── Mackey-Glass:   遅延微分方程式
└── Double Pendulum: ハミルトン系

Stochastic (シード固定での条件付き保証テスト)
├── 白色ノイズ:     x[t] ~ N(0, σ²)
├── AR(1):          x[t] = φ*x[t-1] + ε[t]
└── 因果ペア:       Y[t] = f(X[t-k]) + ε[t]
```

### 4.2 フィクスチャ設計

テストフィクスチャは以下の原則で設計する。

1. **最小サイズの原則**: テストの意図を検証できる最小のデータサイズを使用する。単体テストでは N ≤ 50 を目安とする。ただし統計的テスト（CCM 収束等）では N ≥ 500 が必要。
2. **決定論的生成**: 乱数を使用する場合は固定シードを使い、再現性を保証する
3. **独立性**: 各テスト関数は他のテストに依存しない
4. **conftest.py での共有**: 複数テストファイルで使うフィクスチャは `conftest.py` に集約する

```python
# tests/conftest.py の設計

@pytest.fixture
def linear_series():
    """x[t+1] = 0.5 * x[t], x[0] = 1.0, N=50"""
    x = np.zeros(50)
    x[0] = 1.0
    for i in range(1, 50):
        x[i] = 0.5 * x[i - 1]
    return x

@pytest.fixture
def sine_wave():
    """x[t] = sin(2π * t / 20), N=100"""
    return np.sin(2 * np.pi * np.arange(100) / 20)

@pytest.fixture
def logistic_map():
    """x[t+1] = 3.8 * x[t] * (1 - x[t]), x[0] = 0.4, N=300"""
    x = np.zeros(300)
    x[0] = 0.4
    for i in range(1, 300):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    return x

@pytest.fixture
def coupled_logistic_causal():
    """X→Y の一方向因果がある結合ロジスティック写像, N=1000
    rx=3.8, ry=3.5, Bxy=0.02, Byx=0 (X→Y のみ)
    """
    N = 1000
    rx, ry, Bxy = 3.8, 3.5, 0.02
    rng = np.random.default_rng(0)
    X = np.zeros(N)
    Y = np.zeros(N)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N):
        X[i] = X[i - 1] * (rx - rx * X[i - 1])
        Y[i] = Y[i - 1] * (ry - ry * Y[i - 1]) + Bxy * X[i - 1]
    return X, Y

@pytest.fixture
def independent_pair():
    """独立な2変量ロジスティック写像, N=1000"""
    rng = np.random.default_rng(0)
    N = 1000
    X = np.zeros(N)
    Y = np.zeros(N)
    X[0], Y[0] = 0.4, 0.2
    for i in range(1, N):
        X[i] = 3.8 * X[i - 1] * (1 - X[i - 1])
        Y[i] = 3.5 * Y[i - 1] * (1 - Y[i - 1])
    return X, Y
```

---

## 5. 許容誤差（Tolerance）の設計

### 5.1 基本方針

数値計算のテストでは、浮動小数点演算の性質を考慮して適切な許容誤差を設定する。

| カテゴリ | 許容誤差 | 使用場面 | 例 |
|---|---|---|---|
| **厳密一致** | `atol=1e-15, rtol=0` | 整数演算または丸め誤差のみの場合 | embedding のインデクシング |
| **高精度** | `atol=1e-10` | 解析解が存在する数値計算 | 定数ターゲットの予測 |
| **実装差異** | `atol=1e-5 〜 1e-6` | float32/float64 の差異、異なるコードパス | Tensor vs NumPy |
| **正則化影響** | `atol=1e-8` | 正則化バイアスが微小に存在 | S-Map の OLS 比較 |
| **統計的傾向** | 傾向の検証（差分、比較） | 収束性テスト、カオス系での予測精度 | CCM 収束 |

### 5.2 比較関数の使い分け

```python
# 厳密一致（丸め誤差のみ）
np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)

# 解析解との比較（条件数依存の計算）
np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

# float32/float64 の差異
np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

# 傾向の比較（条件付き保証）
assert rmse_large_lib < rmse_small_lib  # 最小 vs 最大の直接比較
assert corr_max > corr_min + margin     # マージン付き比較
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

## 10. 参考文献

1. Sugihara, G. & May, R.M. (1990). "Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series." *Nature*, 344, 734–741.
2. Sugihara, G. (1994). "Nonlinear forecasting for the classification of natural time series." *Phil. Trans. R. Soc. Lond. A*, 348, 477–495.
3. Sugihara, G., May, R., Ye, H., et al. (2012). "Detecting Causality in Complex Ecosystems." *Science*, 338, 496–500.
4. Takens, F. (1981). "Detecting strange attractors in turbulence." *Lecture Notes in Mathematics*, 898, 366–381.
5. Cenci, S., Sugihara, G., & Saavedra, S. (2019). "Regularized S-map for inference and forecasting with noisy ecological time series." *Methods in Ecology and Evolution*, 10, 650–660.
6. [EDM Algorithms in Depth - Sugihara Lab](https://sugiharalab.github.io/EDM_Documentation/algorithms_in_depth/)
7. [Explaining empirical dynamic modelling using verbal, graphical and mathematical approaches](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094587/) (2024)

---

## 付録: アルファ版からの主要変更点

### 修正されたテスト項目

| テスト | アルファ版 | 変更 | 理由 |
|---|---|---|---|
| `test_linear_system_exact` (simplex) | 完全予測を期待 | 「RMSE < ε」の閾値ベースに変更 | Simplex は加重平均であり線形回帰ではない。完全予測は数値保証されない |
| `test_prediction_improves_with_optimal_e` (simplex) | 最適 E で精度向上 | 「E>1 の最良が E=1 より優れる」に変更 | E に対する精度は逆U字型であり単調改善ではない |
| `test_accuracy_improves_with_library_size` (simplex) | 非増加傾向 | 削除（CCM 収束テストに統合） | Simplex 単体ではライブラリサイズに対する単調改善の理論的保証がない |
| `test_identity_prediction` (smap) | 完全予測を期待 | 条件を限定した注記に変更（独立テストとしては削除） | S-Map は全点回帰であり特定点の通過は保証されない |
| `test_theta_zero_vs_nonzero_on_linear` (smap) | 同精度を期待 | 許容誤差付きの「同程度の精度」に変更 | 正則化の影響で微差が生じる |
| `test_no_convergence_without_causality` (ccm) | 「収束しない」 | 「最大ライブラリでの相関が低い」に変更 | 「収束しない」の判定が困難。偽陽性リスクも完全には排除できない |
| `test_invariance_to_linear_transform` (pearson) | `sign(ac) * corr(X,Y)` | 正スケールのみ（`a,c > 0`）に限定 | edmkit 独自の `mean(axis=-1)` 集約により負スケールでの挙動が標準と異なる可能性 |

### 追加されたテスト項目

| テスト | 根拠 |
|---|---|
| `test_smap_coefficients_state_dependent` | Sugihara (1994) の核心的主張の直接検証 |
| `test_ccm_asymmetry` | Sugihara et al. (2012) の因果方向の非対称性の検証 |

### 削除されたテスト項目

| テスト | 理由 |
|---|---|
| `test_chaos_for_large_tau` (Mackey-Glass) | カオスの判定には追加実装（リアプノフ指数計算等）が必要であり、データ生成器のテスト範囲を超える |
| `test_ccm_bidirectional` | 双方向因果の設定は一方向因果テスト + 非対称性テストでカバーされる |
| `test_simplex_forecast_decay` | edmkit は Tp を明示的に扱っていないため、ユーザー側のデータ構成に依存するテストとなり、ライブラリのテストとしては不適切 |
