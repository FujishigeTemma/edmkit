# Testing TODO

`TESTING-CHECKLIST.md` を基準に `tests/` 全体を見直した、マクロ優先のレビュー結果。
次の改善フェーズでは、個別テストの足し引きより先に、下記の横断方針を揃えるのを優先する。

## Macro Findings

### 1. テストレイヤの役割分担が全体として曖昧

- 現状は `unit` / `property` / `trend` / `integration` / `smoke` の境界がファイルごとに揃っていない。
- `tests/test_e2e.py` と `tests/smoke_test.py` が配線確認と数値妥当性確認を同時に持っている。
- `tests/test_embed.py` も unit、integration、reference-implementation 比較が混在している。

影響:
- 失敗時に「実装バグ」なのか「閾値が brittle」なのか「配線崩れ」なのか切り分けにくい。
- 同じ性質を複数レイヤで重複検証しており、保守コストが増えている。

TODO:
- 各ファイルを次のどれかに寄せる: `unit`, `property`, `trend`, `integration`, `smoke`
- `smoke` は import と最小成功呼び出しだけに限定する
- `e2e` は配線確認に寄せ、数値の主張は module test 側へ戻す
- `test_embed.py` のような混在ファイルは責務ごとに分割または再命名する

### 2. 命名規則とファイル責務が一貫していない

- [`tests/test_embedding.py`](/Users/temma/Repositories/edmkit/tests/test_embedding.py) は `lagged_embed`、[`tests/test_embed.py`](/Users/temma/Repositories/edmkit/tests/test_embed.py) は `scan/select` を持つが、名前から責務の違いが分かりにくい。
- `TestAnalyticalSolutions`, `TestMathematicalProperties`, `TestConvergenceTrends` のようなクラス名は多くのファイルで使われている一方、何の API に対する分類なのかがファイル名頼みになっている。
- `mask` 系だけ別ファイルに出ている module と、同じファイルに混在している module があり、切り方の基準が揃っていない。

影響:
- 新規追加時の置き場判断が属人的になる。
- テスト一覧を見ても、どの API のどの責務がどこで保証されているか把握しづらい。

TODO:
- 命名規則を先に固定する
- 例: `test_embedding_lagged_embed.py`, `test_embedding_scan.py`, `test_simplex_projection_mask.py` のように API 単位へ寄せる
- クラス名は「観点」ではなく「対象 API + 観点」を優先する
- `mask`, `tensor`, `integration` のような横断テーマは分離基準を決めて全 module に統一する

### 3. テストデータ供給戦略が統一されていない

- [`tests/conftest.py`](/Users/temma/Repositories/edmkit/tests/conftest.py) に shared fixture がある一方で、`tests/test_embed.py` や [`tests/test_e2e.py`](/Users/temma/Repositories/edmkit/tests/test_e2e.py) は類似系列をローカル生成している。
- `sine_wave` と `bidirectional_pair` は未使用で、逆に複数箇所で使うべき系列が fixture 化されていない。
- 同じ logistic 系でも fixture とローカル helper で長さや transient の扱いが違う。

影響:
- あるテストだけ別系列を使っていても気づきにくい。
- 閾値調整がデータ差に引っ張られ、レビューしづらい。

TODO:
- 「shared fixture に置くもの」と「inline に留めるもの」の基準を明文化する
- 共有対象は `conftest.py` に寄せ、用途と想定観点を docstring に書く
- 未使用 fixture は削除するか、使う予定があるなら TODO に紐づける
- module ごとのローカル generator helper は、共有すべきなら fixture へ昇格する

### 4. assertion の強さに repo-wide の方針がない

- 一部ファイルは hand-computable な厳密テストを持つが、他では `not all NaN`, `isfinite`, `corr > 0` のような弱い assertion が多い。
- `scan`, `e2e`, `mask` 系に特に弱い smoke 的 assertion が残っている。
- 同じ「trend test」でも、順位相関まで見ているものと、単なる正値確認で済ませているものが混在する。

影響:
- どの程度の壊れ方なら落ちるのかがファイルごとに不揃い。
- 実装回帰を止めたいテストと、単に落ちないことを見るテストの区別が曖昧。

TODO:
- assertion policy を先に決める
- exact に見られるものは exact で見る
- 実装非依存な契約は property で見る
- 現象確認は trend に寄せ、`corr > 0` のような弱い閾値は原則避ける
- `shape/isfinite` 止まりのテストは integration/smoke に限定し、unit からは減らす

### 5. Hypothesis の使い方が file ごとにばらついている

- 良い property test と、実質 shape smoke に留まるものが混在している。
- `mask[:n_valid] = True` のような偏った生成や、`assume(...)` に頼る設計が見られる。
- strategy が module ごとにバラバラで、どのレベルまで shared にするかの方針もない。

影響:
- property-based test の価値が「本質契約の検証」ではなく「ランダム入力で落ちない確認」に寄りがち。
- 重い割に検出力の低いテストが混ざる。

TODO:
- Hypothesis を使う対象を絞る
- shape preservation、permutation invariance、masked subset equivalence、batch/unbatched consistency など、実装非依存な性質に集中させる
- 単なる shape 確認は example-based test に戻す
- shared にする strategy とローカルに置く strategy の基準を決める

### 6. trend test の位置づけが整理されていない

- `simplex`, `smap`, `ccm`, `generate` に trend test があるが、閾値根拠の書き方や壊れやすさの許容度が揃っていない。
- 方向比較が必要な CCM で、正方向のみ見て逆方向比較が不足している。
- generator テストでは、数値 solver 実装に引っ張られやすいチェックが混ざる。

影響:
- trend test が回帰保護になっているのか、単なる brittle なスナップショットになっているのか判別しづらい。

TODO:
- trend test は「方向」「順位」「単調傾向」の確認に限定する
- しきい値には短い理由コメントを付ける
- solver 実装差分に弱い exact trajectory 比較は減らす
- CCM は方向性比較を標準パターンにする

### 7. helper / reference implementation の扱いが整理されていない

- [`tests/test_embed.py`](/Users/temma/Repositories/edmkit/tests/test_embed.py) の `naive_scan` は reference のつもりだが、実装との独立性が弱い。
- 各 file にローカル helper が散らばっており、共有すべきか単発かの判断基準がない。

影響:
- 「参照実装との一致」が強い保証だと誤認されやすい。
- helper がテスト資産として再利用されず、似たコードが増える。

TODO:
- reference implementation を置く基準を明文化する
- 置くなら本体と異なる構造にする
- 独立性が弱い helper comparison は縮小し、外部契約ベースの検証へ寄せる

## Recommended Execution Order

1. テストレイヤと命名規則を先に決める
2. `conftest.py` と local helper の役割を整理する
3. assertion policy と Hypothesis/trend の使い分けを揃える
4. その方針に沿って各 module の個別 TODO を消化する

## Module-Level TODOs

### `embedding` / `scan` / `select`

- [`tests/test_embed.py`](/Users/temma/Repositories/edmkit/tests/test_embed.py) を `scan/select` 専用の名前に寄せる
- `naive_scan` 比較は縮小し、fold 数、NaN padding、選択順位などの外部契約ベースへ置き換える
- `scan` / `select` の invalid input と all-NaN 仕様を先に決める

### `simplex_projection`

- [`tests/test_simplex_projection.py`](/Users/temma/Repositories/edmkit/tests/test_simplex_projection.py) は input validation と shape contract を明示追加する
- [`tests/test_simplex_projection_mask.py`](/Users/temma/Repositories/edmkit/tests/test_simplex_projection_mask.py) は property を masked subset equivalence 中心に組み直す

### `smap`

- [`tests/test_smap.py`](/Users/temma/Repositories/edmkit/tests/test_smap.py) の `assume(...)` 依存を減らす
- [`tests/test_smap_mask.py`](/Users/temma/Repositories/edmkit/tests/test_smap_mask.py) は `mask=None` 後方互換、subset equivalence、異常系で責務を明確化する

### `ccm`

- [`tests/test_ccm.py`](/Users/temma/Repositories/edmkit/tests/test_ccm.py) は helper unit test、wrapper test、directional trend test を分ける
- 方向比較を CCM の標準 assertion パターンにする
- `bootstrap` / `ccm` の input validation を補う

### `metrics` / `splits` / `util`

- [`tests/test_util.py`](/Users/temma/Repositories/edmkit/tests/test_util.py) は unit test と property test の境界を整理し、異常系を厚くする
- [`tests/test_splits.py`](/Users/temma/Repositories/edmkit/tests/test_splits.py) は `stride` 境界と contiguous/coverage 契約を明文化する
- [`tests/test_metrics.py`](/Users/temma/Repositories/edmkit/tests/test_metrics.py) は public API の境界ケースを足す程度に留め、現状構成は大きく崩さない

### `generate`

- [`tests/test_generate.py`](/Users/temma/Repositories/edmkit/tests/test_generate.py) は common contract test と generator-specific trend test を明確に分ける
- exact trajectory 的な比較は減らし、shape / finite / boundedness / qualitative behavior へ寄せる

### `integration` / `smoke`

- [`tests/test_e2e.py`](/Users/temma/Repositories/edmkit/tests/test_e2e.py) は wiring test に寄せる
- [`tests/smoke_test.py`](/Users/temma/Repositories/edmkit/tests/smoke_test.py) は release 用の最小 API 呼び出しだけに絞る

## Open Questions

1. `tests/` の命名は module 単位に寄せるか、観点単位の大型ファイルを許容するか。
2. `mask` 系は全 module で別ファイルに揃えるか、実装本体に近い file へ統合するか。
3. `scan/select` の invalid input は明示的 `ValueError` に揃えるか、NumPy 例外に委ねるか。
4. generator テストは Euler 実装固定の回帰を守るのか、今後の solver 改善に追従したいのか。
5. smoke は配布確認のみに絞るか、最小 correctness も残すか。
