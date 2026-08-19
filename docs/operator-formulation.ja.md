# EDM の作用素論的定式化

この文書は、simplex projection、S-map、Convergent Cross Mapping (CCM) が共通して何を計算しているのかを、力学系の定義から順に書き下したものである。

目的は二つある。
第一に、EDM 固有の語彙を、時系列統計、Koopman 解析、検索拡張予測と同じ土俵で議論できる形に翻訳する。
第二に、その翻訳が既存手法をどこまで説明し、どこから先が本文書独自の主張なのかを明示する。

出発点となる観察は、三つの分野が独立に同じ対象を指していることである。
統計は条件付き期待値 $\mathbb{E}[Y \mid X = q]$ の推定と呼び、作用素論は Koopman 作用素を作用させた観測量の条件付き期待値と呼び、検索拡張予測はデータストア上の条件付き集約と呼ぶ。
kernel analog forecasting については、測度保存かつエルゴード的な力学の下でこの条件付き期待値を一致推定することが示されている[^kaf]。

[^kaf]: Alexander & Giannakis, *Operator-theoretic framework for forecasting nonlinear time series with kernel analog techniques*, Physica D (2020), arXiv:1906.00464。アナログ予報そのものは Lorenz (1969) に遡る。

## 1. 力学系と観測量

**力学系**として、可逆な測度保存変換の四つ組 $(M, \mathcal{B}, \mu, \Phi)$ を固定する。
$M$ は状態空間、$\mathcal{B}$ はその可測集合の $\sigma$ 代数、$\Phi : M \to M$ は可測かつ可逆な写像、$\mu$ は $\Phi$ の不変確率測度である。
不変性は $\mu(\Phi^{-1}A) = \mu(A)$ を意味する。
さらに $\Phi$ は $\mu$ に関してエルゴード的とする。

観測量の空間を、二乗可積分な実数値関数のなす Hilbert 空間

$$
\mathcal{H} := L^2(M, \mathcal{B}, \mu; \mathbb{R}), \qquad
\langle f, g \rangle = \int_M f g \, d\mu, \qquad
\|f\| = \langle f, f \rangle^{1/2}
$$

とする。

**Koopman 作用素**を $U : \mathcal{H} \to \mathcal{H}$, $Uf = f \circ \Phi$ で定める。

> **命題 1.** $U$ は等長である。$\Phi$ が可逆なので $U$ はユニタリであり、$n \in \mathbb{Z}$ に対して $U^n f = f \circ \Phi^n$ が定義される。
>
> *証明.* $\|Uf\|^2 = \int (f \circ \Phi)^2 d\mu = \int f^2 \, d(\Phi_* \mu) = \int f^2 d\mu = \|f\|^2$。可逆性から $U^{-1} f = f \circ \Phi^{-1}$ が逆作用素を与える。$\square$

負の遅延と正の予測ホライズンを同じ記号で扱うために、以降 $U^n$ の指数は整数全体を動く。

実際に測定できる量は有限個の観測量 $g_1, \dots, g_d \in \mathcal{H}$ に限られる。
観測される時系列は、ある軌道 $z_t = \Phi^t z_0$ に沿った値 $g_i(z_t)$ の列である。
エルゴード性により時間平均が $\mu$ に関する空間平均へ収束するので、有限標本から推定される量は $\mu$ に関する量の推定になる。

## 2. 読み枠

**読み枠**を、Koopman 作用素の冪で生成される観測量の有限列として定める。

$$
\mathbf{u} = (u_1, \dots, u_E), \qquad u_j = U^{-\ell_j} g_{i_j} \in \mathcal{H},
\qquad (i_j, \ell_j) \in \{1, \dots, d\} \times \mathbb{Z}
$$

対応する評価写像を $u : M \to \mathbb{R}^E$, $u(z) = (u_1(z), \dots, u_E(z))$ と書く。

軌道上では $u_j(z_t) = g_{i_j}(z_{t - \ell_j})$ が成り立つ。
これが遅延座標である。
$d = 1$ かつ $\ell_j = (j-1)\tau$ とすれば古典的な遅延埋め込みになり、$i_j$ を動かせば多変量埋め込みになる。

この定義の要点は、遅延座標が特徴量の作り方ではなく、基底観測量が $U$ の下で生成する族であることを明示する点にある。
同じ族は Hankel-DMD や HAVOK の入力でもあり、Krylov 部分空間として扱われている[^hankel]。

[^hankel]: Arbabi & Mezić (2017)、Brunton et al. (2017)、Kamb et al. (2020)。遅延座標を並べた Hankel 行列が、観測量から生成される Krylov 部分空間の基底を与える。

読み枠が生成する $\sigma$ 代数と閉部分空間を

$$
\Sigma_{\mathbf{u}} := \sigma(u) \subseteq \mathcal{B}, \qquad
\mathcal{H}_{\mathbf{u}} := L^2(M, \Sigma_{\mathbf{u}}, \mu) \subseteq \mathcal{H}
$$

と書く。

二つの読み枠の**連結** $\mathbf{u} \vee \mathbf{v}$ を、列の連結として定める。
このとき $\Sigma_{\mathbf{u} \vee \mathbf{v}} = \sigma(\Sigma_{\mathbf{u}} \cup \Sigma_{\mathbf{v}})$ であり、$\mathcal{H}_{\mathbf{u}} + \mathcal{H}_{\mathbf{v}} \subseteq \mathcal{H}_{\mathbf{u} \vee \mathbf{v}}$ が成り立つ。
読み枠の集合はこの演算で束をなす。

## 3. 条件付き期待値作用素

中心となる対象は、$\Sigma_{\mathbf{u}}$ に関する条件付き期待値である。

$$
P_{\mathbf{u}} : \mathcal{H} \to \mathcal{H}_{\mathbf{u}}, \qquad
P_{\mathbf{u}} f = \mathbb{E}_\mu[\, f \mid \Sigma_{\mathbf{u}} \,]
$$

補作用素を $Q_{\mathbf{u}} := \mathrm{Id} - P_{\mathbf{u}}$ と書く。

> **命題 2.** $P_{\mathbf{u}}$ は $\mathcal{H}_{\mathbf{u}}$ への直交射影である。すなわち線形であり、$P_{\mathbf{u}}^2 = P_{\mathbf{u}}$、$P_{\mathbf{u}}^* = P_{\mathbf{u}}$、$\|P_{\mathbf{u}}\| \le 1$ を満たす。

これは条件付き期待値の $L^2$ 特徴づけそのものである。
Doob–Dynkin の補題により、$P_{\mathbf{u}} f$ は $\Sigma_{\mathbf{u}}$ 可測なので、可測関数 $F : \mathbb{R}^E \to \mathbb{R}$ が $\mu$ に関してほとんど一意に存在して $P_{\mathbf{u}} f = F \circ u$ と書ける。

EDM の全操作は、次の三つ組の選択に還元される。

$$
(\underbrace{\mathbf{u}}_{\text{どの部分空間か}},\;
 \underbrace{h \in \mathcal{H}}_{\text{何を予測するか}},\;
 \underbrace{n \in \mathbb{Z}}_{\text{いつの値か}})
\;\longmapsto\;
P_{\mathbf{u}} U^n h \;\in\; \mathcal{H}_{\mathbf{u}}
$$

## 4. 予測

> **命題 3（最良近似）.** $P_{\mathbf{u}} U^n h$ は $\mathcal{H}_{\mathbf{u}}$ の中で $U^n h$ への $L^2$ 最良近似である。
>
> $$P_{\mathbf{u}} U^n h = \operatorname*{arg\,min}_{f \in \mathcal{H}_{\mathbf{u}}} \|U^n h - f\|$$

simplex projection、S-map、GP-EDM、kernel analog forecasting は、いずれもこの $F$ の推定量である。
違いは推定の方法にあり、推定される対象は共通する。

> **命題 4（埋め込み定理の位置づけ）.** $u$ が $\mu$ に関してほとんど至るところ単射であり可測な逆を持つならば、$\Sigma_{\mathbf{u}} = \mathcal{B} \pmod \mu$ が成り立ち、したがって $P_{\mathbf{u}} = \mathrm{Id}$ となる。

Takens の定理は、$E$ が状態空間の次元の二倍を超え、かつ観測量と写像が生成的な条件を満たすとき、$u$ が埋め込みになることを主張する。
本文書の枠組みでは、その帰結が $P_{\mathbf{u}} = \mathrm{Id}$ という一つの等式になる。
埋め込み定理は射影が恒等作用素になる条件を与えている。

## 5. 予測スキルと角度

EDM は予測スキルを Pearson 相関 $\rho$ で測る慣習を持つ。
この慣習は射影の幾何から説明できる。

以下、$h$ は中心化されているとする。
測度保存性から $\int U^n h \, d\mu = \int h \, d\mu = 0$ であり、条件付き期待値の性質から $\int P_{\mathbf{u}} U^n h \, d\mu = 0$ も従うので、両者とも中心化されている。

> **命題 5（相関は角度の余弦である）.** $P_{\mathbf{u}} U^n h \neq 0$ のとき
>
> $$\mathrm{corr}\big(U^n h,\; P_{\mathbf{u}} U^n h\big) = \frac{\|P_{\mathbf{u}} U^n h\|}{\|U^n h\|} = \cos \angle\big(U^n h,\; \mathcal{H}_{\mathbf{u}}\big)$$
>
> *証明.* $P_{\mathbf{u}}$ の自己共役性と冪等性から $\langle U^n h, P_{\mathbf{u}} U^n h \rangle = \langle P_{\mathbf{u}} U^n h, P_{\mathbf{u}} U^n h \rangle = \|P_{\mathbf{u}} U^n h\|^2$。これを相関の定義に代入する。$\square$

直交射影に対する三平方の定理

$$
\|U^n h\|^2 = \|P_{\mathbf{u}} U^n h\|^2 + \|Q_{\mathbf{u}} U^n h\|^2
$$

を併せると、$\rho^2 = 1 - \|Q_{\mathbf{u}} U^n h\|^2 / \|U^n h\|^2$ が得られる。
$\rho^2$ は $U^n h$ のうち $\mathcal{H}_{\mathbf{u}}$ で説明される分散の割合である。

この等式は厳密な射影に対する理想化であり、有限標本の推定量に対しては成り立たない。
推定量は射影ではないので、予測値の尺度の誤り（較正誤差）は $\rho$ に現れない。
$\rho$ が固有スコアリング則でないという批判は、この差として正確に述べられる。

## 6. 交差写像

CCM は、原因側の観測量を結果側の読み枠から復元できるかを問う。

> **命題 6（交差写像は部分空間への所属を測る）.** $x \in \mathcal{H}$ を中心化された観測量、$\mathbf{u}_Y$ を別の変数から作った読み枠とする。交差写像スキルの理想極限は $\cos \angle(x, \mathcal{H}_{\mathbf{u}_Y})$ であり、
>
> $$\text{skill} = 1 \iff x \in \mathcal{H}_{\mathbf{u}_Y} \iff x \text{ は } \Sigma_{\mathbf{u}_Y} \text{ 可測}$$

CCM の非対称性は、部分空間の包含関係の非対称性そのものである。
$x \in \mathcal{H}_{\mathbf{u}_Y}$ は $y \in \mathcal{H}_{\mathbf{u}_X}$ を含意しない。
因果という語を使わずに、測定している量を述べきることができる。

この命題は本文書の定式化であり、既存文献に確立された翻訳ではない。
CCM と Koopman 解析を明示的に接続した先行研究は、調査した範囲では見当たらなかった。

Granger 型の問いは、同じ束の上の別の命題として書ける。

$$
\text{CCM:} \quad \|Q_{\mathbf{u}_Y} x\|
\qquad\qquad
\text{Granger 型:} \quad \|Q_{\mathbf{u}_Y} U^n y\| \;\text{と}\; \|Q_{\mathbf{u}_Y \vee \mathbf{u}_X} U^n y\| \;\text{の比較}
$$

CCM は $x$ が $\mathcal{H}_{\mathbf{u}_Y}$ に既に属するかを問い、Granger 型は読み枠を連結して部分空間を広げたときに残差が減るかを問う。
両者は問う位置が異なるだけで、どちらも読み枠が生成する部分空間の束の上の命題である。

この形にすると、派生手法も同じ枠に収まる。

- **UIC**：Granger 型と同型の入れ子比較であり、残差を $L^2$ ノルムではなく Kullback–Leibler 情報量で測る。
- **PCM**：第三の読み枠 $\mathbf{u}_Z$ に対して $Q_{\mathbf{u}_Z}$ を先に施してから角度を測る。偏相関が直交補空間への射影であることの直接の反映である。

## 7. 局所線形化

$P_{\mathbf{u}} U^n h = F \circ u$ の代表 $F : \mathbb{R}^E \to \mathbb{R}$ が点 $q$ で微分可能なとき、その微分 $DF(q)$ を考える。
S-map の係数はこの $DF$ の推定である。
局所定数の推定量では $DF$ の一致推定が得られないので、次数一以上の局所多項式が必要になる。

> **命題 7.** $DF$ は $U^n$ の微分ではなく $P_{\mathbf{u}} U^n$ の微分である。

$P_{\mathbf{u}} = \mathrm{Id}$ が成り立つとき、$F = (U^n h) \circ u^{-1}$ となり、$DF(q)$ は読み枠を座標に取った表示での $h \circ \Phi^n$ の微分になる。
真のヤコビアンとは座標変換 $Du$ を挟んで共役の関係にある。

$P_{\mathbf{u}} \neq \mathrm{Id}$ のときは、この関係すら成り立たない。
$DF$ は射影された関数の微分にとどまる。

生態学の文献で相互作用ヤコビアンと呼ばれている量を力学のヤコビアンと同一視するには、埋め込みの忠実性が要る。
この但し書きは枠組みから自動的に出る。

## 8. 収束

CCM はライブラリのサイズを変えたときのスキルの推移を証拠に使う。

有限標本から構成される推定量を $\hat{P}^{(N)}_{\mathbf{u}}$ と書くと、ライブラリサイズ曲線は $\hat{P}^{(N)}_{\mathbf{u}} \to P_{\mathbf{u}}$ という作用素推定の学習曲線である。

したがって収束は $\Sigma_{\mathbf{u}_Y}$ 可測性の証拠ではなく、推定量の一致性の現れである。
スキルの水準が上がる要因は可測性以外にも多くあるので、収束の観測だけでは命題 6 の右辺を結論できない。
CCM の既知の弱点が、この枠組みでは推定層と理論層の混同として位置づけられる。

## 9. 設計空間

推定の設計は二つの軸に分解される。

**軸 1：どの部分空間へ射影するか。**
$\sigma$ 代数が生成する $\mathcal{H}_{\mathbf{u}}$、有限辞書の張る $\mathrm{span}\{\psi_1, \dots, \psi_m\}$、Krylov 部分空間、再生核 Hilbert 空間などの選択肢がある。
埋め込み次元の選択、変数選択、multiview embedding、CCM の向きは、すべてこの軸に属する。

**軸 2：射影を有限標本からどう推定するか。**
局所定数（simplex projection、アナログ予報、$k$ 近傍回帰）、局所線形（S-map、LOESS）、カーネルリッジと Gauss 過程（kernel analog forecasting、GP-EDM）、大域最小二乗（DMD、EDMD）が並ぶ。

| 手法 | 軸 1（部分空間） | 軸 2（推定） |
| --- | --- | --- |
| simplex projection | $\mathcal{H}_{\mathbf{u}}$ | 局所定数 |
| S-map | $\mathcal{H}_{\mathbf{u}}$ | 局所線形 |
| GP-EDM | 再生核 Hilbert 空間 | カーネルリッジ |
| kernel analog forecasting | 再生核 Hilbert 空間 | カーネルリッジ |
| DMD, EDMD | 有限辞書の張る空間 | 大域最小二乗 |
| Hankel-DMD, HAVOK | Krylov 部分空間 | 大域最小二乗 |

二つの軸が直交していることが、この整理の実利である。
DMD と S-map の関係は、大域線形回帰と LOESS の関係にあたる。
同じ射影を異なる方法で推定しているにすぎない。

EDM の既存 API は両軸の語を混ぜている。
`simplex_projection(X, Y, Q, k=E+1)` において `E` は軸 1 の選択、`simplex` は軸 2 の選択であり、種類の異なる決定が同じ関数の引数に並んでいる。

## 10. 拡張の余地

### 閉包と記憶項

$P_{\mathbf{u}}$ と $U$ は交換しない。
$\mathrm{Id} = P_{\mathbf{u}} + Q_{\mathbf{u}}$ を繰り返し挿入すると、離散時間の Mori–Zwanzig 分解が得られる。

$$
U^n = \sum_{m=0}^{n-1} U^{\,n-1-m}\, P_{\mathbf{u}}\, U\, (Q_{\mathbf{u}} U)^m \;+\; (Q_{\mathbf{u}} U)^n
$$

$m = 0$ の項が Markov 項、$m \ge 1$ の項が記憶項、最後の項が直交項である。

遅延座標を増やす操作は、記憶項を Markov 項に吸収する操作にあたる。
埋め込み定理の条件が満たされるとき直交項が消え、遅延座標を用いた Mori–Zwanzig 発展が単一の Markov 項に帰着することは、既に報告されている[^mz]。
EDM における埋め込み次元の選択は閉包問題であり、記憶核と粗視化の理論がそのまま利用できる。

[^mz]: *Mori–Zwanzig mode decomposition: Comparison with time-delay embeddings*, arXiv:2311.09524。および Lin et al. による Mori–Zwanzig と Koopman 作用素の接続。

### 部分空間の束

写像 $\mathbf{u} \mapsto \mathcal{H}_{\mathbf{u}}$ は、読み枠の束から閉部分空間の束への写像である。
連結 $\vee$ が使えるので、因果ネットワークの推定を束の構造推定として定式化できる。
multiview embedding は複数の $P_{\mathbf{u}}$ の平均であり、条件付き独立性は束の中の直交関係である。

### スペクトル

$P_{\mathbf{u}} U P_{\mathbf{u}}$ の固有値は、圧縮された Koopman スペクトル、すなわち DMD 固有値である。
$DF$ の固有値はその局所版にあたる。
時変ヤコビアンの固有値を用いた動的安定性の解析は、DMD 固有値の局所版を見ていると述べられる。

### 確率的な系

$P_{\mathbf{u}}$ は条件付き期待値なので、決定論性を仮定していない。
観測ノイズと過程ノイズは $P_{\mathbf{u}} \neq \mathrm{Id}$ として最初から枠内にある。

## 11. 前提と適用範囲

**不変測度と定常性を仮定している。**
$L^2(\mu)$ の上で議論しているので、非定常な系はこの枠組みの外にある。

**理論層と推定層を分ける。**
$P_{\mathbf{u}}$ は $L^2(\mu)$ 上の射影であり、simplex projection と S-map は有限標本上のアルゴリズムである。
後者は前者の近似であって射影ではない。
elastic net で正則化した S-map や、ハイパーパラメータを周辺尤度で推定する Gauss 過程は、予測対象に依存するので観測量について線形ですらない。

**条件付き期待値は平均のみを扱う。**
予測分布や分位点を扱うには、$P_{\mathbf{u}}$ を Markov 核へ一般化する必要がある。
GP-EDM と UIC が実質的にそこにいる。
ただし核へ一般化すると $L^2$ の幾何（角度と三平方の定理）を失うので、二層に分けて保持するのが妥当だと考える。

## 12. 語彙の対応

| EDM | 時系列統計 | Koopman 解析 | 検索拡張予測 |
| --- | --- | --- | --- |
| 遅延埋め込み, SSR | ラグ行列, Hankel 行列 | Krylov 部分空間 | クエリエンコーダ, コンテキスト窓 |
| library | 学習集合 | データ行列 | データストア |
| prediction set | 検証集合 | | クエリ集合 |
| Theiler window | purging, embargo | | 自己リーク除去 |
| simplex projection | $k$ 近傍回帰, アナログ予報 | kernel analog forecasting | kNN-LM, kNN-MTS |
| 重み $\exp(-d/d_{(1)})$ | 適応バンド幅の指数カーネル | | 適応温度の softmax |
| S-map | LOESS, 局所線形回帰 | 局所線形近似 | 検索後の局所モデル当てはめ |
| $\theta$ | バンド幅の逆数 | | softmax 温度の逆数 |
| S-map 係数 | 局所回帰係数 | 有限時間ヤコビアン | 局所サロゲートの係数 |
| $T_p$ | 予測ホライズン, direct multi-step | $U^n$ の指数 | 予測長 |
| generateSteps | iterated multi-step | $U$ の反復適用 | 自己回帰デコード |
| $\rho$ | 相関スキル | 部分空間との角度の余弦 | |
| CCM | 関数従属性の検定 | 部分空間への所属 | クロスモーダル検索 |
| 収束 | 学習曲線 | 作用素推定の一致性 | データストアのスケーリング |
| surrogate | 制約付きリサンプリング | | 置換検定 |
| multiview embedding | 特徴部分集合のアンサンブル | 辞書の選択 | マルチインデックス融合 |

翻訳しきれず残るものは三つある。

1. **遅延座標を状態の忠実な複製とみなす主張。** 標準的な時系列の機械学習はラグ窓を特徴として使うだけで、それが状態と同型だとは主張しない。命題 4 の $P_{\mathbf{u}} = \mathrm{Id}$ がこの主張にあたる。
2. **交差写像の向き。** 結果側の再構成から原因側を復元するという向きは、決定論的な結合系に特有である。
3. **収束を証拠として使うこと。** 機械学習において学習曲線は診断の道具であって検定統計量ではない。

## 未決の設計判断

- 中心対象を $P_{\mathbf{u}}$ とするか $P_{\mathbf{u}} U^n$ とするか。$U$ は系から与えられ $P_{\mathbf{u}}$ は解析者が選ぶので、選択の所在が分離される前者を推す。
- 読み枠を観測量の有限族として定めるか、写像 $M \to \mathbb{R}^E$ として定めるか。連結 $\vee$ が自然に入り束構造が使える前者を推す。
- $L^2$ の層に留めるか、Markov 核の層まで上げるか。二層で保持する案を第 11 節に述べた。
