---
marp: true
theme: base3
_class: invert
paginate: true
header: 12/05(火)　データ同化B　中間報告　佐藤 匠
---

# **12月05日 中間報告**

### これまでの課題とこれからの実習課題

<br>

#### 0500-32-7354,　佐藤 匠

---

# 目次



- これまでの課題1~3について
  - [課題1] SIS法, SIR法の実装
  - [課題2] Lanczos 法による SV の計算
  - [課題3] LV, BV の計算

- これからの実習課題4~5について
  - [課題4] 実習課題の概要
  - [課題4] 機械学習におけるパラメータ推定
  - [課題4] 先行研究と計画

---

<!--_class: chap_head-->

# これまでの課題 1~3
## [課題1] SIS法, SIR法の実装

---

# [課題1] SIS法, SIR法の実装｜SIS, SIR のアイデア

<table>

<td>

<img src=../1010/SIR_concept.png height=400>

Arakida et al.(2016). 
doi: 10.5194/npg-2016-30.

</td>
<td>

${x^f}^{(k)} _{i+1} = \mathcal{M}({x^f}^{(k)} _{i})$　　　(時間発展)
↓　↓　↓
$\hat{w}^{(k)} _{i+1} = w^{(k)} _{i} \cdot l^{(k)}_i$　　　(重みの更新)

$$l^{(k)}_i = 
\frac{(\sqrt{2 \pi})^{-40}}{\sqrt{|R|}} \exp \left(\frac{R^{-1}}{-2}(y-H{x^f}^{(k)})^{2}\right).
$$
↓　↓　↓
$w^{(k)} _{i+1} = \hat{w}^{(k)} _{i} / \sum_j \hat{w}^{(j)} _{i}.$　　(正規化)
↓　↓　↓
Resampling　　　　(サイコロ $m$ 回)
重み $w_k = 1/m$
</td>
</table>

---

# [課題1] SIS法, SIR法の実装｜SIS法 結果(RMSE)

| $m=10^3$ | $m=10^5$ | |
|-------|--------|-|
| <img src=SIS_RMSE-1000.png width=450> | <img src=SIS_RMSE-100000.png width=450> | $\overline{{x^a_i}}$ <br> $= \sum_{k} w^{(k)}_i \cdot {x^f}^{(k)} _{i}.$ <br><br> とした。<br><br> 初期メンバーは$y^O(t=0)$ の周りに分散 $\sigma = 1$ の<br>正規分布でサンプリングした。|

---

# [課題1] SIS法, SIR法の実装｜SIS法 結果($N_\mathrm{eff}$)

| $m=10^3$ | $m=10^5$ | |
|-------|--------|-|
| <img src=SIS_N_eff-1000.png width=450> | <img src=SIS_N_eff-100000.png width=450> | $N_\mathrm{eff} (t_i)$ <br> $= \sum_k 1 / (w^{(k)}_i)^2$ <br><br> $w_i$ が一つだけ <br> 生き残る|

---

# [課題1] SIS法, SIR法の実装｜SIR法 結果($N_\mathrm{eff}$)

<center>

| $m=10^3$ | $m=10^5$ |
|-------|--------|
| <img src=../1017/SIR_N_eff-1000.png width=510> | <img src=../1017/SIR_N_eff-100000.png width=510> |

</center>

---

# [課題1] SIS法, SIR法の実装｜SIR法 結果(RMSE)

<center>

| $m=10^3$ | $m=10^5$ |
|-------|--------|
| <img src=../1017/SIR_RMSE-1000.png width=510> | <img src=../1017/SIR_RMSE-100000.png width=510> |

</center>

---

# [課題1] SIS法, SIR法の実装｜SIRアニメーション

<center>

| $m=10^3$ | $m=10^5$ |
|-------|--------|
| <img src=../1017/1000-0_50.gif width=510> | <img src=../1017/100000-0_50.gif width=510> |

</center>

---

# [課題1] SIS法, SIR法の実装｜閾値 $N_e$ と 擾乱 $|\eta|$

<center>
<table>
<td>
<img src=../1107/SIR_perturb/ave_SIR_perturb_vs_Ne.png height=500>
</td>
<td>

ランダム過程の大きさ: $|\eta|$ が小さいとき、
Resampling の頻度を増やす必要がある。

<br>

SIS $(N_e < 1)$ は動かない。

</td>
</table>
</center>

---

# [課題1] SIS法, SIR法の実装｜ランダム過程の大きさ

<center>

| 小さすぎ　$(\eta = 0.05)$ | 大きすぎ　$(\eta = 5.0)$ |
|-------|--------|
| <img src=../1017/too_small_perturb.gif width=510> | <img src=../1017/too_big_perturb.gif width=510> |

</center>

---

<!--_class: chap_head-->

# これまでの課題 1~3
## [課題2] Lanczos 法による SV の計算

---

<!--
# [課題2] SV の計算｜接線型コード

入力の微小な擾乱に対する出力の応答が線型であると近似する。

$$
\vec{y} + \delta{ \vec{y} } = \mathcal{ M } ( \vec{x} + \delta \vec{x} ), \qquad 
\delta \vec{y} = \mathcal{ M } ( \vec{x} + \delta \vec{x} ) - \vec{y}\approx \mathbf{L} \ \delta \vec{x}.
$$



$\mathbf{L}$ はモデル $\mathcal{M}$ を $\vec{x}$ の周りでテイラー展開した時の係数である。

$$
\mathbf{L} = \frac{ \partial \mathcal{M} }{ \partial \vec{x} } \in \mathbb{R}^{N \times J}, 
\qquad \mathrm{L}_{kl} = \frac{ \partial y_k }{ \partial x_l }.
$$

この行列 $\mathbf{L}$ を *接線型モデル (Tangent Linear model, TLM)* という。

```python
def TangentLinearCode(x0_1, x0_2, ..., x0_J, dx1, dx2, ..., dxJ): #入力 = 基本場 + 摂動
    # アルゴリズム
    Lx = (dy1, dy2, ..., dyN) #TLMと入力ベクトルxの積
    return Lx #出力 = 応答
```

---

# [課題2] SV の計算｜Adjoint コード

内積を $\langle \vec{a}, \vec{b} \rangle = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$ とする。接線型モデル $\mathbf{L}$ に対して

$$\langle \vec{y} , \mathbf{L} \vec{x} \rangle = \langle \mathbf{L}^\dagger \vec{y} , \vec{x} \rangle$$

を満たす行列 $\mathbf{L}^\dagger$ を *随伴モデル (Adjoint model, ADJ)* という。
$\mathbf{L}$ が実数値行列であるとき、$\mathbf{L}^\dagger = \mathbf{L}^T$ である。

<br>

```python
def AdjointCode(x0_1, x0_2, ..., x0_J, dy1, dy2, ..., dyN): #入力 = 基本場 + 応答
    # アルゴリズム
    LTy = (dx1, dx2, ..., dxJ) #ADJと入力ベクトルyの積
    return LTy #出力 = 摂動
```
--->

# [課題2] SV の計算｜Lanczos法

![bg]("white")
![bg](../1024/lorenz63_SV.gif)

TLM と ADJ に対して 
*leading Singular Vector*: $\vec{v}_1$ と $\sigma_1$ が存在して、

$(\mathbf{L}^T \mathbf{L}) \vec{v}_1 = \sigma_1 ^{~2} \vec{v}_1$

を満たす。$v_i, \sigma_i$ は SVD$(\mathbf{M}^T\mathbf{M})$ である。

| <img src=../1024/field0.png width=280> | → |<img src=../1024/field4.png width=290> |
|-|-|-|

---

# [課題2] SV の計算｜Lorenz-96 モデルの SV

<center>

| <img src=../1107/perturb/1_SingularVector.png height=250> | →→ | <img src=../1107/perturb/3_SingularVector.png height=250> | →→ | <img src=../1107/perturb/10_SingularVector.png height=250> |
|-|-|-|-|-|
| <img src=../1107/perturb_dX/1_SingularVector_dX.png height=250> | →→ | <img src=../1107/perturb_dX/3_SingularVector_dX.png height=250> | →→ | <img src=../1107/perturb_dX/10_SingularVector_dX.png height=250> |

</center>

---

<!--

# [課題2] SV の計算｜Lanczos法の収束

<table>

<td>
<img src=../1114/Lanczos_similarity.png height=550>
</td>

<td>

擾乱のベクトル $d\vec{X}_k$ をランダムな方向に 
$k \leq 100$ 個生成し、成長させた。

各メンバー同士の差の絶対値の平均。

倍精度 (Float 64) で計算しているので、
$10^{-16}$ 程度までの精度が出る。

</td>

</table>

---

# Lanczos 法｜各モードの成長率 $\sigma_i$ (特異値)

<table>

<td>
<img src=../1121/SingularValue_modes.png height=550>
</td>

<td>

mode:　$1 \leq i \leq 15$ までは成長モード
mode: $16 \leq i \leq 18$ までは伸縮しづらい
mode: $19  \leq i \leq 40$ までは減衰モード

</td>

</table>

--->

# Lanczos 法｜各モードの SV の成分 (特異ベクトル)

approx の $\Delta t = 0.05, \ \delta = 10^{-3}$ とすると、各成分の誤差は (10th~30thで) $10^{-4}$ 程度

<table>

<tr>
<th> approx </th>
<th> TLM </th>
<th> diff (approx - TLM) </th>
</tr>

<tr>
<td>
<img src=../1121/SingularVector_approx.png width=370>
</td>

<td>
<img src=../1121/SingularVector_TLM.png width=370>
</td>

<td>
<img src=../1121/SingularVector_diff.png width=370>
</td>
</tr>

</table>

---

<!--_class: chap_head-->

# これまでの課題 1~3
## [課題3] LV, BV の計算

---

# [課題3] LV, BV の計算｜Lyapnov Vector

<table>

<tr>
<th> α = 2.0 </th>
<th> α = 0.2 </th>
<th> 備考 </th>
</tr>

<tr>
<td> <img src=./LVs_heatmap_2.0.png width=400> </td>
<td> <img src=../1121/LVs_heatmap_0.2.png width=400> </td>
<td>

$\delta x_{i+1}^{~F} = \mathbf{M}_{(t_i, \ x_i^A)} \delta x_i^{~A}$

↓↓↓

$\delta x_{i+1}^{~A} = \alpha \frac{x_{i+1}^{~F}}{|x_{i+1}^{~F}|}$

($\alpha < |x_{i+1}^{~F}|$ のとき)


擾乱のメンバー数: 100
</td>
</tr>

</table>

---

# [課題3] LV, BV の計算｜LV の $|\delta x^k|$ の推移

<table>

<tr>
<th> α = 2.0 </th>
<th> α = 0.2 </th>
<th> 備考 </th>
</tr>

<tr>
<td> <img src=./LV_abs_x_2.0.png width=400> </td>
<td> <img src=../1121/LV_abs_x_0.2.png width=400> </td>
<td>

$|\delta x_i^k|, \ (k=1, 2, \cdots 100)$

100個のグラフの重描き

<br>

初期擾乱の大きさ

$|\delta x_0| = 1.0$

<br>

$\alpha$ 超えるとスケーリング
グラフの概形は相似

</td>
</tr>

</table>

---

# [課題3] LV, BV の計算｜Bred Vector

<table>

<tr>
<th> α = 2.0 </th>
<th> α = 0.2 </th>
<th> 備考 </th>
</tr>

<tr>
<td> <img src=./BVs_heatmap_2.0.png width=400> </td>
<td> <img src=../1121/BVs_heatmap_0.2.png width=400> </td>
<td>

$\delta x_{i+1}^{~F} = \mathcal{M}_{i}(x_i^A + \delta x_i^{A})$

↓↓↓

$\delta x_{i+1}^{~A} = \alpha \frac{x_{i+1}^{~F}}{|x_{i+1}^{~F}|}$

($\alpha < |x_{i+1}^{~F}|$ のとき)


擾乱のメンバー数: 100
</td>
</tr>

</table>

---


# # [課題3] LV, BV の計算｜BV の $|\delta x^k|$ の推移

<table>

<tr>
<th> α = 2.0 </th>
<th> α = 0.2 </th>
<th> 備考 </th>
</tr>

<tr>
<td> <img src=./BV_abs_x_2.0.png width=400> </td>
<td> <img src=../1121/BV_abs_x_0.2.png width=400> </td>
<td>

$|\delta x_i^k|, \ (k=1, 2, \cdots 100)$

100個のグラフの重描き

<br>

初期擾乱の大きさ

$|\delta x_0| = 1.0$

<br>

$\alpha$ 超えるとスケーリング
グラフの概形は相似

</td>
</tr>

</table>

---

<!-- _class: chap_head -->

# これからの実習課題4~5について

---

![bg]("white")
![bg]("white")
![bg 100%](https://wdc.kugi.kyoto-u.ac.jp/igrf/anime/16-20.gif)

# [課題4] 実習課題の概要

### テーマ: パラメータ推定

##### &emsp;時系列予測に用いられる機械学習モデル (AR, RNN) と <br> &emsp;データ同化の融合

<br>

#### 研究の背景

- 地球の磁場は **非線形** に変動している → 予測したい
- そのメカニズムはよくわかっていない → 予測しづらい
- 観測データはたくさんある → 機械学習が有効？

よくある「学習」＝ **RMSE** の逆伝搬
&emsp; → 『どの程度外れそうか』の予報 (UQ) が難しい → **尤度に基づいた学習の実装**


---

# [課題4] 機械学習におけるパラメータ推定と予測

例：**AR(4) モデル** による学習と予測
&emsp;&emsp;$Y_t = \varepsilon_t + c + \phi_1 \ Y_{t-1} + \phi_2 \ Y_{t-2} + \phi_3 \ Y_{t-3} + \phi_4 \ Y_{t-4}$ , （$c,\ \phi_i$ がパラメータ）



<table>
<tr>
<td> 
<img src=3a_train_test.png width=560 style="border:5px solid red;"> 

</td>
<td> <img src=ar_good.png width=560> </td>
</tr>
<tr>
<td> <img src=ar_bad_osc.png width=560> </td>
<td> <img src=ar_bad_exp.png width=560> </td>
</tr>
</table>

---

# [課題4] 先行研究と今後の計画

##### *AR(p) モデル* (Box and Jenkins(1970) など)：
確率過程をもとにしたモデル。歴史が古く、数学的によく調べられている。
状態空間表現が可能で、__KFを用いた応用例多数 (星谷、丸山 (1990)<sup>*1</sup>)__。

##### *Recurrent Neural Network* (Werbos(1990)<sup>*2</sup> など)：
再帰的な構造を取り入れることで時系列データに対応した深層学習モデル。
__RNN自体を 拡張KF とみなす学習法が研究されている (金城ら (1997)<sup>*3</sup>)__。

<br>

#### 今後の計画

- 🔥 先行研究・具体的な実装法のサーベイ (12/12 まで)
- 🔲 AR(p) の実装（12月中）
- 🔲 RNN の実装（1月中）

<!-- _footer: 1: <https://doi.org/10.2208/jscej.1990.416_349>, 2: <https://doi.org/10.1109/5.58337>, 3: <https://doi.org/10.5687/iscie.10.401> -->

---


<!---

  - → 最尤推定の手法は教科書にまとめられるほど。(沖本(2010) など)
    - 笛木ら (2021): <https://doi.org/10.14856/conf.32.0_529>
    - 京都大(2013): <http://www.kumst.kyoto-u.ac.jp/kougi/time_series/ex1218.html>

-->

---
![bg](bisque)

# 機械学習のデモ｜Recurrent Neural Network

<table>
<td>
<img src=./materials/mtg1108/CO2_again/9a_rnn.png height=510>
</td>

<td>
再帰的な構造で時系列データに対応したNNモデル<br>
　　　　　<img src=materials/sem1/rnn_comp.png width=350/> <br>
　　　　　　　　　↓ ↓ ↓ ↓<br>
　<img src=materials/sem1/rnn_tenkai.png width=600/>

</td>
</table>

<!--_footer: 画像: 農学情報科学 Biopapyrs <https://axa.biopapyrus.jp/deep-learning/rnn/>, CC BY 4.0-->