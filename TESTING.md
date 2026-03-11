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

### 2.3 理論的主張と数値保証の区別

EDM の原著論文における理論的主張をテストケースに変換する際、主張の根拠の強さによってテスト設計を明確に区別する。各テスト項目は以下の3カテゴリのいずれかに分類される。

| カテゴリ | 記号 | 定義 | 許容誤差の性質 | 例 |
|---|---|---|---|---|
| **代数的保証** | **(A)** | 精確算術で恒等的に成立する性質。許容誤差は浮動小数点丸め誤差のみ | `atol` は `O(k × eps_mach)` | 定数ターゲットの予測、凸結合の値域 |
| **理論的近似** | **(B)** | 特定の前提条件下で近似的に成立する性質。許容誤差はアルゴリズム的バイアス（正則化等）を含む | `atol` は既知のバイアス + 安全マージン | θ=0 での OLS 近似（α > 0 時） |
| **統計的期待** | **(C)** | 確率的・漸近的に成立する性質。点推定の許容誤差ではなく、傾向・効果量で判定 | 最小効果量の閾値 | 非線形系での θ > 0 の優位性、CCM 収束 |

**重要な設計原則**: カテゴリ (C) のテストであっても、「傾向が存在すること」のみを検証する bare minimum な判定基準ではなく、理論的に期待される**最小効果量**（minimum effect size）を設定する。テストが理論的に期待される効果を十分に検出できることを保証するためである。

---

## 3. モジュール別テスト設計

### 3.1 `embedding.py` — `lagged_embed()`

> Embedding はインデックス操作のみで算術演算を行わないため、全テストで厳密一致（`assert_array_equal`）を使用する。

#### 3.1.1 解析解テスト

| テスト名 | 内容 | 期待値 | 許容誤差 |
|---|---|---|---|
| `test_identity_embedding` | `e=1, tau=1` ではリシェイプのみ | `x.reshape(-1, 1)` と一致 | exact (`array_equal`) |
| `test_known_values` | 小さな配列で手計算と照合 | 各行が `[x[i+(e-1)*tau], ..., x[i]]` | exact |
| `test_output_shape` | 任意の `(N, tau, e)` に対して出力形状を検証 | `(N - (e-1)*tau, e)` | N/A（形状チェック） |
| `test_linear_sequence` | 等差数列の入力で各行の差分が `tau` 刻み | 各行内の差が均一 | exact |

#### 3.1.2 数学的性質テスト

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_no_information_loss` | 出力の全要素が入力に含まれている | exact |
| `test_tau1_e1_preserves_values` | `tau=1, e=1` で元の値がそのまま保持される | exact |
| `test_sliding_window_consistency` | 連続する行が1ステップずつスライドしている | exact |

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

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_known_2d_distances` | 手計算可能な2-3点の二乗距離 | `atol=1e-14` |
| `test_identity_distance_zero_diagonal` | `A=B` のとき対角要素が0 | `atol=1e-14` |
| `test_single_point` | 1点同士の距離は差の二乗和 | `atol=1e-14` |

**数学的性質テスト**

| テスト名 | 検証する性質 | 許容誤差 |
|---|---|---|
| `test_symmetry` | `D(A, B) = D(B, A)^T` | `atol=1e-14` |
| `test_non_negativity` | 全要素 ≥ 0 | exact（`D.clamp(min_=0)` により保証） |
| `test_self_distance_diagonal_zero` | `D(A, A)` の対角は0 | `atol=1e-14` |
| `test_triangle_inequality` | `sqrt(D[i,k]) ≤ sqrt(D[i,j]) + sqrt(D[j,k])` | `atol=1e-10` |
| `test_translation_invariance` | `D(A+c, B+c) = D(A, B)` | `atol=1e-12` |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 | `atol=1e-14` |

**NumPy / Tensor 一致テスト**

| テスト名 | 内容 | 許容誤差 |
|---|---|---|
| `test_numpy_tensor_agreement` | 同一入力に対して `pairwise_distance` と `pairwise_distance_np` が同一結果 | `atol=1e-4`（float32 vs float64） |

> **実装上の注意**: `A_sq + B_sq - 2*A@B^T` の展開公式は、距離が点の大きさに比して小さい場合に桁落ちが発生する。float32（tinygrad）では `~1e-3` 以下、float64（NumPy）では `~1e-8` 以下の真の距離で影響が出る。`D.clamp(min_=0)` / `np.clip(D, 0, None)` によって負値は防がれるが、小距離の精度は低下する。

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

| テスト名 | カテゴリ | 内容 | 根拠 | 許容誤差 | 前提条件 |
|---|---|---|---|---|---|
| `test_identity_prediction` | **(A)** | クエリ点がライブラリ点と一致する場合、最近傍のY値を予測 | d_min クランプにより最近傍の重みが支配的 → 他の重みが float64 で 0 にアンダーフロー | `atol=1e-15` | ライブラリ点間の最小距離 > 0.01（等間隔点 N≥10 で容易に達成） |
| `test_linear_system_high_accuracy` | **(B)** | 有界な線形力学系で高精度な予測 | Takens 定理により E=1 で再構成可能。加重平均は線形補間に近似するが、指数重みのバイアスにより厳密一致はしない（Simplex は線形回帰ではなく凸結合） | `rho > 0.999, RMSE < 0.01` | N=500、有界な線形写像（例: `x[t+1] = 0.9*x[t] + 0.05*sin(0.1*t)`）でアトラクタを均一に被覆 |
| `test_constant_target` | **(A)** | Y が定数 c の場合、予測値 = c | `Σ(w_i * c) / Σ(w_i) = c`（代数的恒等式、重みに依存しない） | `atol=1e-14, rtol=1e-14` | なし |

> **`test_linear_system_exact` からの変更理由**: Simplex projection は `exp(-d_i/d_min)` 重みの凸結合であり、線形関数の厳密な復元には逆距離重み（`w_1/w_2 = (x_2-q)/(q-x_1)`）が必要。指数重みは最近傍に偏るため、クエリ点の真の像ではなく最近傍の Y 値に引きずられる。有界な線形系で N=500 とすれば、ライブラリ間隔 δ ≈ 0.002 となり、バイアスは `~0.27 * |a| * δ` 程度に抑えられる。

#### 3.3.2 数学的性質テスト

| テスト名 | カテゴリ | 検証する性質 | 許容誤差 | 前提条件 |
|---|---|---|---|---|
| `test_prediction_in_target_range` | **(A)** | 予測値が E+1 近傍の Y 値の凸結合であるため、`min(Y) ≤ ŷ ≤ max(Y)` が成立 | exact（不等式チェック） | なし（指数重みは常に正） |
| `test_neighbor_count_equals_e_plus_1` | **(A)** | k = E + 1 近傍を使用（アルゴリズムの定義: `simplex_projection.py:118`） | exact | なし |
| `test_prediction_improves_with_optimal_e` | **(C)** | カオス系では最適な E が存在し、E=1 より高精度 | `corr(E_best) - corr(E=1) > 0.05` | Logistic map r=3.8, N≥500（E=8 まで探索可能なサンプル数）。過渡状態を除去（先頭50点を破棄） |
| `test_permutation_invariance_of_library` | **(A)** | ライブラリ点の順序入れ替えで結果不変 | `atol=1e-15` | テストデータに等距離点が存在しないこと（KDTree のタイブレーク回避） |

> **`test_prediction_improves_with_optimal_e` の設計**: Sugihara & May (1990) は E-vs-prediction skill 曲線に逆U字型のピークが存在することを示した。テストは「ピークの E が E=1 より有意に良い」ことを検証する。E の単調改善は保証されない。最小効果量 0.05 は Logistic map (dimension ≈ 1) で E=2 への改善が理論的に顕著であることに基づく。

#### 3.3.3 収束性・傾向テスト

| テスト名 | カテゴリ | 内容 | 判定基準 | 前提条件 |
|---|---|---|---|---|
| `test_noise_sensitivity` | **(C)** | 加法的ガウスノイズの増加に伴い予測精度が低下 | `corr(σ=0) - corr(σ=0.3) > 0.2`。σ ∈ {0, 0.05, 0.15, 0.3} の Spearman 順位相関 < 0 | Logistic map r=3.8 N=300、信号振幅 [0,1] に対する σ 比率 |
| `test_simplex_forecast_decay` | **(C)** | 予測ホライズン Tp の増加に伴い精度が低下 | `corr(Tp=1) > corr(Tp=5) + 0.1` | カオス系（Logistic map）。Tp はデータ構成で制御（edmkit は Tp を明示的に扱わない） |

> **`test_accuracy_improves_with_library_size` の削除理由**: Simplex projection 単体でのライブラリサイズに対する精度の単調改善は理論的に保証されていない。この性質は CCM の文脈（Sugihara et al. 2012）での収束性に由来する。CCM セクション（3.5）の `test_convergence_with_known_causality` に統合する。

#### 3.3.4 自己無撞着性テスト

| テスト名 | カテゴリ | 内容 | 許容誤差 |
|---|---|---|---|
| `test_batch_vs_individual` | **(A)** | 3Dバッチ入力と2D個別入力の結果一致 | `atol=1e-14`（同一 float64 パス、インデックスのみ異なる） |
| `test_numpy_vs_tensor` | **(A)** | `use_tensor=True/False` で同一結果 | `atol=1e-4`（float32 vs float64 の精度差。累積誤差は `~sqrt(k) × 1e-7`） |

#### 3.3.5 退化ケース

| テスト名 | 内容 |
|---|---|
| `test_single_query_point` | クエリ点1つ |
| `test_minimum_library_size` | ライブラリサイズ = E + 1（最小近傍数） |

---

### 3.4 `smap.py` — `smap()`

> S-Map のテストでは、正則化パラメータ `alpha` の影響を分離するため、各代数的テストに `alpha=0` 版（純粋な数学的性質のテスト）と `alpha=1e-10` 版（デフォルト設定での近似テスト）の2つを設ける。

#### 3.4.1 解析解テスト

| テスト名 | カテゴリ | 内容 | 根拠 | 許容誤差 |
|---|---|---|---|---|
| `test_theta_zero_equals_ols` (alpha=0) | **(A)** | `theta=0, alpha=0` で OLS と厳密一致 | θ=0 で W=I（Sugihara 1994 の定義）→ 正規方程式は OLS と同一 | `atol=1e-12, rtol=1e-12`（条件数 κ ≈ O(100) × eps_mach） |
| `test_theta_zero_approx_ols` (alpha=1e-10) | **(B)** | `theta=0, alpha=1e-10` で OLS に近似 | Tikhonov 正則化による係数バイアス ≈ `1e-10`（N=50, E=1 の場合: `alpha × tr(A) / a_11 × \|β\| ≈ 1.3e-10`） | `atol=1e-9`（10× safety margin） |
| `test_linear_system_recovery` (alpha=0) | **(A)** | `Y = a*X + b, theta=0, alpha=0` で完全復元 | OLS で真のパラメータを厳密に復元（線形回帰の基本性質） | `atol=1e-12, rtol=1e-12` |
| `test_linear_system_recovery` (alpha=1e-10) | **(B)** | 同上、`alpha=1e-10` で近似復元 | 傾き係数のバイアス ≈ `alpha × tr(A) / a_11 × \|a\| ≈ 8e-10`（N=50, X∈[0,1], a=2） | `atol=1e-8` |
| `test_constant_target` | **(A)** | Y が定数 c の場合、予測値 = c | 重み付き回帰で β_1_ols = 0 → Ridge 縮小 β_1_ridge = 0 も厳密。切片は正則化対象外（`eye[0,0]=0`） | `atol=1e-12, rtol=1e-12` |

> **`test_identity_prediction` (smap) の削除理由**: S-Map は全ライブラリ点を使った重み付き線形回帰であり、回帰超平面が特定のデータ点を通過する保証はない。Simplex projection と異なり、identity prediction は S-Map のアルゴリズムの性質ではない。θ が大きい場合に近似的に成立するが、条件が実装依存であり、安定したテストにならない。

#### 3.4.2 数学的性質テスト

| テスト名 | カテゴリ | 検証する性質 | 許容誤差 | 前提条件 |
|---|---|---|---|---|
| `test_theta_increases_locality` | **(A)** | θ 増加に伴い、遠方点の重み `exp(-θd/D_mean)` が減少 | exact（不等式チェック） | 距離 d > 0 の点について検証 |
| `test_regularization_effect` | **(A)** | alpha 増加に伴い非切片係数のノルムが縮小 | exact（不等式チェック） | リッジ回帰の凸最適化の標準的結果。EDM 固有の理論ではなく実装の正則化の性質 |
| `test_intercept_not_regularized` | **(A)** | 切片項は正則化されない（`smap.py:157` の `eye[0,0]=0` による） | `atol=1e-14`（well-conditioned data） | テストデータの条件数 κ(X^T X) < 1e8 |
| `test_permutation_invariance_of_library` | **(A)** | ライブラリ順序に依存しない（X^T W X は行順序不変） | `atol=1e-14` | なし（KDTree 不使用のためタイブレーク問題なし） |
| `test_smap_coefficients_state_dependent` | **(A)** | θ > 0 で回帰係数がクエリ点ごとに異なる | 異なるクエリ点間で係数の非一致を検証 | Sugihara (1994) の核心的主張。非線形系（Logistic map）を使用 |
| `test_smap_weight_formula` | **(A)** | 重み `exp(-θd/D_mean)` が実装と一致 | `atol=1e-15` | 原著論文の式と実装の直接照合 |

#### 3.4.3 theta に関する比較テスト

| テスト名 | カテゴリ | 内容 | 判定基準 | 前提条件 |
|---|---|---|---|---|
| `test_theta_zero_vs_nonzero_on_linear` | **(B)** | 線形系ではθ=0 と θ>0 の精度差が小さい | RMSE 比率 `RMSE(θ>0) / RMSE(θ=0)` ∈ [0.85, 1.15] | `alpha=0`, θ ∈ {0, 0.5, 1.0, 2.0}（θ≥4 は有限サンプルで有効標本サイズが過小になり不安定）。N≥100 の有界線形系。 |
| `test_theta_nonzero_better_on_nonlinear` | **(C)** | 非線形系では θ>0 が θ=0 より高精度 | `RMSE(θ=0) - RMSE(θ_best) > 0.1 × RMSE(θ=0)`（最低 10% の改善） | Logistic map r=3.8 N=300, E=2。θ ∈ {2, 4}（中程度の値。θ が大きすぎると過学習リスク）。 |

> **θ ≥ 4 を `test_theta_zero_vs_nonzero_on_linear` で除外する理由**: 高 θ では遠方点の重みが指数的に減衰し、有効標本サイズが N_eff ≈ N/5〜N/3 に縮小する。N=100, E=1 で θ=4 の場合、N_eff ≈ 20-33 となり、パラメータ数 E+1=2 に対して余裕が小さく、正則化バイアスと条件数の悪化が支配的になる。理論的にはθに依存しないはずの性質が、数値的に検証不能になる。

#### 3.4.4 自己無撞着性テスト

| テスト名 | カテゴリ | 内容 | 許容誤差 |
|---|---|---|---|
| `test_batch_vs_individual` | **(A)** | バッチ処理と個別処理の一致 | `atol=1e-12`（2D: `cdist` vs 3D: `pairwise_distance_np` で距離計算アルゴリズムが異なるため、simplex の `1e-14` より緩い） |

#### 3.4.5 数値安定性テスト

| テスト名 | 内容 | 判定基準 |
|---|---|---|
| `test_ill_conditioned_library` | ほぼ共線的なライブラリ点でも正則化により安定 | 予測値が有限（NaN/Inf なし）。alpha 増加で安定性が向上 |
| `test_large_theta_stability` | θ が大きくても NaN/Inf が発生しない | θ ∈ {10, 20, 50} で予測値が有限 |

> **`test_ill_conditioned_library` の注意**: α=1e-10 では条件数改善が限定的（κ_orig=1e12 → κ_reg ≈ 1e10）。テストは精度ではなく安定性（有限値の出力）を検証する。

---

### 3.5 `ccm.py`

#### 3.5.1 `pearson_correlation`

**解析解テスト**

| テスト名 | 内容 | 期待値 | 許容誤差 |
|---|---|---|---|
| `test_perfect_positive` | `Y = aX + b (a > 0)` | 1.0 | `atol=1e-14` |
| `test_perfect_negative` | `Y = -aX + b` | -1.0 | `atol=1e-14` |
| `test_uncorrelated` | 直交する正弦波 | ≈ 0 | `atol=0.01` |
| `test_self_correlation` | `X` と `X` | 1.0 | `atol=1e-14` |

**数学的性質テスト**

| テスト名 | 検証する性質 |
|---|---|
| `test_range` | -1 ≤ corr ≤ 1 |
| `test_symmetry` | `corr(X, Y) = corr(Y, X)` |
| `test_invariance_to_linear_transform` | `corr(aX+b, cY+d) = sign(ac) * corr(X, Y)` |
| `test_batch_consistency` | バッチ次元の各要素が個別計算と一致 |

> **実装上の注意**: `pearson_correlation`（`ccm.py:272`）は `cov / (std_X * std_Y)` を計算する際、std=0（定数入力）に対するガードがない。小さいライブラリサイズでのブートストラップで定数的な予測が生成された場合、NaN が発生する。テストでは最小ライブラリサイズを `5 × (E+1)` 以上に設定するか、アグリゲータに `np.nanmean` を使用する。

#### 3.5.2 `bootstrap` / `ccm`

**収束性テスト（CCM の本質的な検証）**

| テスト名 | カテゴリ | 内容 | 判定基準 | 前提条件 |
|---|---|---|---|---|
| `test_convergence_with_known_causality` | **(C)** | X→Y の因果がある系で、ライブラリサイズ増加に伴い cross-map 精度が収束 | `mean_corr(max_lib) - mean_corr(min_lib) > 0.1`。Spearman 順位相関（lib_sizes vs mean_correlations）> 0 | 結合 Logistic map（Bxy≥0.02, Byx=0）、N≥500、n_samples≥50、per-test seeded sampler |
| `test_no_convergence_without_causality` | **(C)** | 独立な系列では cross-map 精度が低く、収束しない | `mean_corr(max_lib) < 0.3` かつ `mean_corr(max_lib) - mean_corr(min_lib) < 0.1` | 独立な2つの Logistic map（r1=3.8, r2=3.7、異なる初期値）。共有外力なし |
| `test_ccm_asymmetry` | **(C)** | 一方向因果 X→Y で、Y xmap X の精度が X xmap Y より有意に高い | `corr(Y_xmap_X)[max_lib] - corr(X_xmap_Y)[max_lib] > 0.15` | Sugihara et al. (2012) Figure 1。注意: 「X causes Y」のとき「Y xmap X」が高精度（因果の逆方向に cross-map する） |
| `test_ccm_bidirectional` | **(C)** | 双方向因果の系で両方向の cross-map が収束 | 両方向で `mean_corr(max_lib) > mean_corr(min_lib) + 0.05` | 双方向結合 Logistic map（Bxy>0, Byx>0）。Sugihara et al. (2012) Figure 3B |

> **CCM の方向規約**: CCM は「効果変数のアトラクタに原因変数の情報が埋め込まれている」ことを検出する。したがって、X→Y の因果を検出するには、Y の埋め込みから X を cross-map する（Y xmap X）。テストではこの方向規約を明示的に検証する。

> **収束の定義**: CCM における「収束」はブートストラップサンプルの**期待値**に対する性質であり、個別サンプルでは非単調な振る舞いが生じうる。n_samples=20（edmkit デフォルト）では標準誤差が 0.05-0.10 と大きく、テストが不安定になる。n_samples≥50 で標準誤差を 0.03-0.06 に低減する。

**構成要素のテスト**

| テスト名 | 内容 |
|---|---|
| `test_custom_sampler` | カスタムサンプラーが正しく呼ばれる |
| `test_custom_aggregator` | カスタムアグリゲータ（例: 中央値）が適用される |
| `test_reproducibility_with_fixed_seed` | 同一シードで同一結果 |

**自己無撞着性テスト**

| テスト名 | カテゴリ | 内容 | 許容誤差 |
|---|---|---|---|
| `test_ccm_equals_aggregated_bootstrap` | **(A)** | `ccm()` の出力が `bootstrap()` + aggregator と一致 | exact（実装の定義から保証） |
| `test_with_simplex_convenience` | **(A)** | `ccm.with_simplex_projection()` が手動構築と同一結果 | exact |
| `test_with_smap_convenience` | **(A)** | `ccm.with_smap()` が手動構築と同一結果 | exact |

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
├── 有界線形系:     x[t+1] = a*x[t] + forcing (収縮しない)
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

1. **最小サイズの原則**: 単体テスト（代数的テスト）では N ≤ 50。統計的テストではテストの検出力に応じてサイズを決定。
2. **決定論的生成**: 乱数を使用する場合は固定シードを使い、再現性を保証する
3. **独立性**: 各テスト関数は他のテストに依存しない。特に CCM テストでは per-test の RNG を使用する
4. **conftest.py での共有**: 複数テストファイルで使うフィクスチャは `conftest.py` に集約する
5. **アトラクタ被覆**: カオス系フィクスチャでは過渡状態を除去（先頭50点を破棄）し、アトラクタ上の軌道のみを使用する

```python
# tests/conftest.py の設計

@pytest.fixture
def bounded_linear_series():
    """有界な線形系: x[t+1] = 0.9*x[t] + 0.05*sin(0.1*t), N=500
    アトラクタを均一に被覆するための非収縮線形写像。"""
    N = 500
    x = np.zeros(N)
    x[0] = 0.5
    for i in range(1, N):
        x[i] = 0.9 * x[i - 1] + 0.05 * np.sin(0.1 * i)
    return x

@pytest.fixture
def sine_wave():
    """x[t] = sin(2π * t / 20), N=100"""
    ...

@pytest.fixture
def logistic_map():
    """x[t+1] = 3.8 * x[t] * (1 - x[t]), x[0] = 0.4, N=500
    先頭50点を過渡状態として破棄し、N=500の定常軌道を返す。"""
    N_total = 550
    x = np.zeros(N_total)
    x[0] = 0.4
    for i in range(1, N_total):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
    return x[50:]  # 過渡除去

@pytest.fixture
def lorenz_series():
    """Lorenz system σ=10, ρ=28, β=8/3, dt=0.05, N=500
    total_time = 25.0 (≈ 23 Lyapunov times, 十分なアトラクタ被覆)
    先頭 20 time units を破棄。"""
    ...

@pytest.fixture
def causal_pair():
    """X→Y の一方向因果がある結合 Logistic map, N=1000
    rx=3.8, ry=3.5, Bxy=0.02, Byx=0
    先頭50点を過渡状態として破棄。"""
    ...

@pytest.fixture
def independent_pair():
    """独立な2つの Logistic map (r1=3.8, r2=3.7), N=1000
    異なる初期値、共有外力なし。"""
    ...

def make_seeded_sampler(seed: int):
    """テスト独立な RNG を持つサンプラーを生成。
    ccm.py のグローバル RNG (ccm.py:11) に依存しないことでテスト順序非依存を保証。"""
    rng = np.random.default_rng(seed)
    def sampler(pool, size):
        return rng.choice(pool, size=size, replace=True)
    return sampler
```

### 4.3 フィクスチャの理論的前提条件の検証

| フィクスチャ | 用途 | サイズ | 理論的前提条件 | 充足状況 |
|---|---|---|---|---|
| `bounded_linear_series` | simplex/smap 線形系テスト | N=500 | 線形写像、アトラクタの均一被覆 | ✓ 非収縮 + forcing で [0, 1] 近傍を均一に被覆 |
| `logistic_map` | simplex/smap カオス系テスト | N=500 | カオス (r=3.8 > 3.57)、定常軌道 | ✓ 過渡除去済、E≤8 の探索に十分 |
| `lorenz_series` | 多変量カオステスト | N=500, dt=0.05 | アトラクタ被覆 (≥ 10 Lyapunov times) | ✓ total=25.0 time units ≈ 23 Lyapunov times |
| `causal_pair` | CCM 収束テスト | N=1000 | 一方向因果、十分な結合強度 | ✓ Bxy=0.02, N=1000 で収束が検出可能 |
| `independent_pair` | CCM 非収束テスト | N=1000 | 真の独立性、共有外力なし | ✓ 異なる r 値、独立初期値 |

---

## 5. 許容誤差（Tolerance）の設計

### 5.1 基本方針

数値計算のテストでは、浮動小数点演算の性質を考慮して適切な許容誤差を設定する。許容誤差はテストカテゴリ（A/B/C）に応じて以下の原則で決定する。

| カテゴリ | 許容誤差の決定方法 | 根拠 |
|---|---|---|
| **(A) 代数的保証** | `atol = safety_margin × κ × eps_mach` | 誤差源は浮動小数点丸めのみ |
| **(B) 理論的近似** | `atol = safety_margin × (bias + κ × eps_mach)` | 既知のアルゴリズム的バイアスを加算 |
| **(C) 統計的期待** | 最小効果量の閾値 | 理論的に期待される効果の下限 |

### 5.2 許容誤差の一覧

#### 代数的保証テスト (A)

| テスト名 | モジュール | atol | rtol | 根拠 |
|---|---|---|---|---|
| `test_identity_embedding` 他 | embedding | exact (`array_equal`) | N/A | 純粋なインデックス操作。算術演算なし |
| `test_constant_target` | simplex | `1e-14` | `1e-14` | 重みの積和除算のみ: `~k × eps_mach ≈ 4 × 2.2e-16 ≈ 1e-15` |
| `test_identity_prediction` | simplex | `1e-15` | 0 | well-separated library (間隔>0.01) で `exp(-d/1e-6)` が 0 にアンダーフロー |
| `test_prediction_in_target_range` | simplex | 0 (不等式) | N/A | 正の重みの凸結合の代数的性質 |
| `test_theta_zero_equals_ols` (α=0) | smap | `1e-12` | `1e-12` | 条件数 κ ≈ O(100) × eps_mach |
| `test_linear_system_recovery` (α=0) | smap | `1e-12` | `1e-12` | OLS で真のパラメータを復元 |
| `test_constant_target` | smap | `1e-12` | `1e-12` | Ridge 縮小で β_1=0 は不動点。条件数改善の間接効果あり |
| `test_intercept_not_regularized` | smap | `1e-14` | `1e-14` | `eye[0,0]=0` による設計。条件数 κ < 1e8 の前提 |
| `test_batch_vs_individual` | simplex | `1e-14` | `1e-14` | 同一 float64 パス |
| `test_batch_vs_individual` | smap | `1e-12` | `1e-12` | `cdist` vs `pairwise_distance_np` のアルゴリズム差 |
| `test_numpy_vs_tensor` | simplex | `1e-4` | `1e-4` | float32 vs float64: `~sqrt(k) × 1e-7` の累積 |
| `test_numpy_tensor_agreement` | util | `1e-4` | `1e-4` | float32 二乗距離の桁落ち |
| `test_ccm_equals_aggregated_bootstrap` | ccm | exact | N/A | 実装の定義から保証 |

#### 理論的近似テスト (B)

| テスト名 | モジュール | atol / 判定基準 | 根拠 |
|---|---|---|---|
| `test_theta_zero_approx_ols` (α=1e-10) | smap | `atol=1e-9` | 係数バイアス ≈ 1e-10 + 10× safety margin |
| `test_linear_system_recovery` (α=1e-10) | smap | `atol=1e-8` | 傾きバイアス ≈ 8e-10 + safety margin |
| `test_linear_system_high_accuracy` | simplex | `rho > 0.999, RMSE < 0.01` | 有界線形系 N=500 でのバイアス ≈ 0.27\|a\|δ ≈ 0.0005 |
| `test_theta_zero_vs_nonzero_on_linear` | smap | RMSE 比率 ∈ [0.85, 1.15] | 有限標本 + 局所化で有効 N 減少 |

#### 統計的期待テスト (C) — 最小効果量

| テスト名 | モジュール | 最小効果量 | 根拠 |
|---|---|---|---|
| `test_prediction_improves_with_optimal_e` | simplex | `corr(E_best) - corr(E=1) > 0.05` | Logistic map dim≈1 で E=2 への改善は顕著 |
| `test_theta_nonzero_better_on_nonlinear` | smap | RMSE 改善 > 10% | Sugihara (1994): 非線形系で θ>0 の改善は 20-50% |
| `test_noise_sensitivity` | simplex | `corr(σ=0) - corr(σ=0.3) > 0.2` | ノイズ 30% で顕著な劣化が期待される |
| `test_simplex_forecast_decay` | simplex | `corr(Tp=1) - corr(Tp=5) > 0.1` | カオス系の予測ホライズン劣化 |
| `test_convergence_with_known_causality` | ccm | `Δcorr(max-min lib) > 0.1` | Bxy≥0.02 で十分な収束信号 |
| `test_no_convergence_without_causality` | ccm | `corr(max_lib) < 0.3` | 独立系列で高相関は生じない |
| `test_ccm_asymmetry` | ccm | `Δcorr(causal - noncausal) > 0.15` | ブートストラップ SE ≈ 0.07 の約 2σ |
| `test_ccm_bidirectional` | ccm | 両方向で `Δcorr > 0.05` | 双方向因果で両方が収束 |

### 5.3 比較関数の使い分け

```python
# 厳密一致（インデックス操作のみ）
np.testing.assert_array_equal(actual, expected)

# 代数的保証（丸め誤差のみ）
np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

# 理論的近似（既知のバイアスを含む）
np.testing.assert_allclose(actual, expected, atol=1e-9)

# 統計的期待（傾向検証）
assert mean_corr_max_lib - mean_corr_min_lib > 0.1  # 最小効果量
```

---

## 6. 実装固有の注意点

edmkit の実装には原著論文にはない以下の設計判断があり、テスト設計に影響する。

| # | 場所 | 処理 | テストへの影響 |
|---|---|---|---|
| 1 | `simplex_projection.py:126` | `d_min` を `1e-6` にクランプ | identity prediction で他の重みが 0 にアンダーフロー → `atol=1e-15` が可能。ただしこれは実装のアーティファクトであり、クランプ値が変更されればテストも変更が必要 |
| 2 | `smap.py:157-159` | 正則化項を `alpha × trace(X^T W X) × I'` でスケーリング | データ依存の正則化強度。Cenci et al. (2019) に近いが原著にはない |
| 3 | `smap.py:143` | `d_mean` を `1e-6` にクランプ | 全距離=0 の退化ケースで均一重み（θ=0 と等価）にフォールバック |
| 4 | `ccm.py:11` | グローバル RNG `np.random.default_rng(42)` | テスト実行順序で結果が変わる。テストでは per-test seeded sampler を使用 |
| 5 | `util.py:79,122` | `D.clamp(min_=0)` / `np.clip(D, 0, None)` | 二乗距離の桁落ちで負値が発生 → 0 にクリップ。float32 では `~1e-3` 以下の真の距離で精度低下 |
| 6 | `ccm.py:272` | `cov / (std_X * std_Y)` に std=0 ガードなし | 定数入力で NaN。最小 lib_size ≥ 5×(E+1) で回避 |
| 7 | `smap.py:162` | `np.linalg.solve` に条件数チェックなし | 高 θ + 小 α で ill-conditioned 化の可能性 |
| 8 | `simplex_projection.py:221` | Tensor パスの d_min は float32 の `.clip(min_=1e-6)` | float32 の距離クリップ（#5）と組み合わせで、クラスタ化されたデータで重み集中が強化される |

---

## 7. テスト実行戦略

### 7.1 pytest マーカー

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

### 7.2 実行コマンド

```bash
# 通常の開発サイクル（高速テストのみ）
uv run pytest tests/ -m "not slow and not gpu"

# 全テスト
uv run pytest tests/

# 特定モジュール
uv run pytest tests/test_simplex_projection.py -v
```

### 7.3 パラメタライズの活用

同一ロジックを複数データソースで検証する場合は `@pytest.mark.parametrize` を使う。

```python
@pytest.mark.parametrize("e", [1, 2, 3, 5])
@pytest.mark.parametrize("tau", [1, 2, 4])
def test_embedding_output_shape(sine_wave, e, tau):
    result = lagged_embed(sine_wave, tau=tau, e=e)
    assert result.shape == (len(sine_wave) - (e - 1) * tau, e)
```

---

## 8. テストファイル構成

移行後のテストファイル構成:

```
tests/
├── conftest.py                    # 共有フィクスチャ（make_seeded_sampler 含む）
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

## 9. 移行計画

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

## 10. pyEDM 比較テストの保持について

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

## 参考文献

1. Sugihara, G. & May, R.M. (1990). "Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series." *Nature*, 344, 734–741. — Simplex projection の原論文
2. Sugihara, G. (1994). "Nonlinear forecasting for the classification of natural time series." *Phil. Trans. R. Soc. Lond. A*, 348, 477–495. — S-Map の原論文
3. Sugihara, G., May, R., Ye, H., et al. (2012). "Detecting Causality in Complex Ecosystems." *Science*, 338, 496–500. — CCM の原論文
4. Takens, F. (1981). "Detecting strange attractors in turbulence." *Lecture Notes in Mathematics*, 898, 366–381. — 埋め込み定理
5. [EDM Algorithms in Depth - Sugihara Lab](https://sugiharalab.github.io/EDM_Documentation/algorithms_in_depth/) — pyEDM/rEDM 公式ドキュメント
6. [Explaining empirical dynamic modelling using verbal, graphical and mathematical approaches](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094587/) — EDM の解説論文 (2024)
7. Cenci, S., Sugihara, G., & Saavedra, S. (2019). "Regularized S-map for inference and forecasting with noisy ecological time series." *Methods in Ecology and Evolution*, 10, 650–660. — 正則化 S-Map
