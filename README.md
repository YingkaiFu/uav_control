<div align="center">
  <img src="resources/luster-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">数据集</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
</div>


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmpretrain)](https://pypi.org/project/mmpretrain/)
[![PyPI](https://img.shields.io/pypi/v/mmpretrain)](https://pypi.org/project/mmpretrain)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmpretrain.readthedocs.io/zh_CN/1.x/)
[![badge](https://github.com/open-mmlab/mmpretrain/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpretrain/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpretrain/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpretrain)
[![license](https://img.shields.io/github/license/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/blob/1.x/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmpretrain.svg)](https://github.com/open-mmlab/mmpretrain/issues)

文档: <https://mmpretrain.readthedocs.io/zh_CN/latest>


## 简介
LusterPretrain 是一个基于 PyTorch 和 MMPretrain 的工业分类、预训练开源工具箱。它是 FBrain 项目的一部分。

当前分支代码目前支持 PyTorch 1.6 以上的版本。

<!-- ![示例图片](resources/seg_demo.gif) -->

### 主要特性

- **统一的基准平台**

  我们将各种各样的分类网络集成到了一个统一的工具箱，进行基准测试。


- **丰富的即插即用的算法和模型**

  LusterPretrain 支持了众多主流的和自研的骨干网络，例如 ResNet，MobileNet，FBNet 等.

- **速度快**

  训练速度比其他分类和预训练的代码库更快或者相当。

## 更新日志

最新版本 v1.0.0rc5 在 2023.02.01 发布。
如果想了解更多版本更新细节和历史信息，请阅读[更新日志](docs/en/notes/changelog.md)。

## 当前性能
**注意**：以下统计数据基于未进行目标过滤的原始预测结果得到

<table>
    <tr>
        <td>模型版本</td>
        <td>指标</td>
        <td><a href="#Graphite_raw">石墨</td>
        <td><a href="#Soldered_dot">焊点</a></td>
        <td><a href="#Photovoltaic_cells">光伏</td>
        <td><a href='#Li_battery_dry'>干料</a></td>
        <td><a href='#Li_battery_wed'>湿料</a></td>
    </tr>
    <tr>
        <td rowspan="4">MobileNetV2_Pretrain</td>
        <td>Top-1</td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
    </tr>
    <tr>
        <td>单轮时长</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(FP32)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(TR_fp16)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
       <tr>
        <td rowspan="4">MobileNetV2_Pretrain</td>
        <td>Top-1</td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
    </tr>
    <tr>
        <td>单轮时长</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(FP32)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(TR_fp16)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td rowspan="4">MobileNetV2_Pretrain</td>
        <td>Top-1</td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
    </tr>
    <tr>
        <td>单轮时长</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(FP32)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(TR_fp16)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td rowspan="4">MobileNetV2_Pretrain</td>
        <td>Top-1</td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
        <td>24.5% </td>
    </tr>
    <tr>
        <td>单轮时长</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(FP32)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
    <tr>
        <td>速度(TR_fp16)</td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
        <td> 36.9%  </td>
    </tr>
</table>

### 逐数据集性能指标
<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBNet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="9">ABB</td>
    <td>BarCode_ABB_Class1</td>
    <td>490</td>
    <td>473</td>
    <td>1377</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>BarCode_ABB_Class2</td>
    <td>3327</td>
    <td>1408</td>
    <td>4847</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Metal_ABB_Class1</td>
    <td>912</td>
    <td>789</td>
    <td>2664</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Metal_ABB_Class2</td>
    <td>2249</td>
    <td>1033</td>
    <td>3316</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Screw_ABB_Class1</td>
    <td>9917</td>
    <td>4329</td>
    <td>14004</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Screw_ABB_Class2</td>
    <td>7267</td>
    <td>3147</td>
    <td>10220</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Shim_ABB_Class1</td>
    <td>4292</td>
    <td>1836</td>
    <td>6093</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Shim_ABB_Class2</td>
    <td>5328</td>
    <td>2237</td>
    <td>7470</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>累计</td>
    <td>33782</td>
    <td>15252</td>
    <td>49991</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBNet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="17" id="Li_battery_dry">锂电-干料</td>
    <td>blankbrand</td>
    <td>651</td>
    <td>237</td>
    <td>608</td>
    <td></td>
    <td></td>
    <td>4</td>
    <td>1</td>
    <td>8</td>
    <td>1</td>
    <td>0</td>
    <td>2</td>
    <td>6</td>
    <td>0</td>
  </tr>
  <tr>
    <td>blankfold</td>
    <td>294</td>
    <td>133</td>
    <td>277</td>
    <td></td>
    <td></td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>blankother</td>
    <td>791</td>
    <td>302</td>
    <td>755</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>8</td>
    <td>2</td>
    <td>10</td>
    <td>5</td>
    <td>0</td>
    <td>0</td>
    <td>10</td>
  </tr>
  <tr>
    <td>brand</td>
    <td>586</td>
    <td>214</td>
    <td>508</td>
    <td></td>
    <td></td>
    <td>0</td>
    <td>11</td>
    <td>1</td>
    <td>12</td>
    <td>7</td>
    <td>0</td>
    <td>0</td>
    <td>10</td>
  </tr>
  <tr>
    <td>bubble</td>
    <td>308</td>
    <td>126</td>
    <td>256</td>
    <td></td>
    <td></td>
    <td>2</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>2</td>
    <td>2</td>
    <td>0</td>
  </tr>
  <tr>
    <td>crease</td>
    <td>822</td>
    <td>331</td>
    <td>827</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>0</td>
    <td>2</td>
    <td>1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>darkmark</td>
    <td>814</td>
    <td>304</td>
    <td>740</td>
    <td></td>
    <td></td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
  </tr>
  <tr>
    <td>darkspot</td>
    <td>1006</td>
    <td>393</td>
    <td>971</td>
    <td></td>
    <td></td>
    <td>20</td>
    <td>12</td>
    <td>15</td>
    <td>20</td>
    <td>10</td>
    <td>16</td>
    <td>20</td>
    <td>16</td>
  </tr>
  <tr>
    <td>drainfoil</td>
    <td>857</td>
    <td>342</td>
    <td>820</td>
    <td></td>
    <td></td>
    <td>15</td>
    <td>26</td>
    <td>26</td>
    <td>30</td>
    <td>25</td>
    <td>11</td>
    <td>30</td>
    <td>35</td>
  </tr>
  <tr>
    <td>drainfoil2</td>
    <td>939</td>
    <td>377</td>
    <td>864</td>
    <td></td>
    <td></td>
    <td>18</td>
    <td>6</td>
    <td>27</td>
    <td>17</td>
    <td>2</td>
    <td>17</td>
    <td>31</td>
    <td>22</td>
  </tr>
  <tr>
    <td>grid</td>
    <td>69</td>
    <td>17</td>
    <td>54</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>0</td>
    <td>3</td>
    <td>0</td>
    <td>0</td>
    <td>4</td>
    <td>4</td>
    <td>0</td>
  </tr>
  <tr>
    <td>other</td>
    <td>876</td>
    <td>347</td>
    <td>836</td>
    <td></td>
    <td></td>
    <td>3</td>
    <td>3</td>
    <td>1</td>
    <td>4</td>
    <td>3</td>
    <td>3</td>
    <td>4</td>
    <td>4</td>
  </tr>
  <tr>
    <td>pending</td>
    <td>131</td>
    <td>53</td>
    <td>124</td>
    <td></td>
    <td></td>
    <td>8</td>
    <td>4</td>
    <td>10</td>
    <td>9</td>
    <td>5</td>
    <td>8</td>
    <td>9</td>
    <td>8</td>
  </tr>
  <tr>
    <td>pit</td>
    <td>602</td>
    <td>243</td>
    <td>570</td>
    <td></td>
    <td></td>
    <td>9</td>
    <td>4</td>
    <td>12</td>
    <td>4</td>
    <td>3</td>
    <td>4</td>
    <td>10</td>
    <td>6</td>
  </tr>
  <tr>
    <td>powder</td>
    <td>1470</td>
    <td>567</td>
    <td>1288</td>
    <td></td>
    <td></td>
    <td>8</td>
    <td>19</td>
    <td>6</td>
    <td>13</td>
    <td>15</td>
    <td>4</td>
    <td>5</td>
    <td>19</td>
  </tr>
  <tr>
    <td>thindarkmark</td>
    <td>187</td>
    <td>73</td>
    <td>156</td>
    <td></td>
    <td></td>
    <td>4</td>
    <td>0</td>
    <td>9</td>
    <td>1</td>
    <td>2</td>
    <td>4</td>
    <td>7</td>
    <td>0</td>
  </tr>
  <tr>
    <td>累计</td>
    <td>10403</td>
    <td>4059</td>
    <td>9654</td>
    <td></td>
    <td></td>
    <td>94</td>
    <td>94</td>
    <td>121</td>
    <td>121</td>
    <td>77</td>
    <td>77</td>
    <td>130</td>
    <td>130</td>
  </tr>
</tbody>
</table>

<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBnet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="15" id="Li_battery_wed">锂电-湿料</td>
    <td>blackpoint</td>
    <td>1243</td>
    <td>501</td>
    <td>1144</td>
    <td></td>
    <td></td>
    <td>9</td>
    <td>17</td>
    <td>11</td>
    <td>19</td>
    <td>16</td>
    <td>8</td>
    <td>10</td>
    <td>17</td>
  </tr>
  <tr>
    <td>black</td>
    <td>1418</td>
    <td>567</td>
    <td>1312</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>2</td>
    <td>0</td>
    <td>4</td>
    <td>1</td>
    <td>0</td>
    <td>1</td>
    <td>2</td>
  </tr>
  <tr>
    <td>bubble</td>
    <td>1003</td>
    <td>395</td>
    <td>902</td>
    <td></td>
    <td></td>
    <td>13</td>
    <td>23</td>
    <td>10</td>
    <td>16</td>
    <td>16</td>
    <td>8</td>
    <td>15</td>
    <td>20</td>
  </tr>
  <tr>
    <td>darkmark</td>
    <td>960</td>
    <td>387</td>
    <td>849</td>
    <td></td>
    <td></td>
    <td>14</td>
    <td>9</td>
    <td>14</td>
    <td>10</td>
    <td>7</td>
    <td>12</td>
    <td>13</td>
    <td>9</td>
  </tr>
  <tr>
    <td>drainfoil</td>
    <td>739</td>
    <td>298</td>
    <td>665</td>
    <td></td>
    <td></td>
    <td>37</td>
    <td>54</td>
    <td>42</td>
    <td>41</td>
    <td>33</td>
    <td>44</td>
    <td>27</td>
    <td>68</td>
  </tr>
  <tr>
    <td>foilbump</td>
    <td>830</td>
    <td>272</td>
    <td>689</td>
    <td></td>
    <td></td>
    <td>3</td>
    <td>0</td>
    <td>4</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>2</td>
    <td>0</td>
  </tr>
  <tr>
    <td>foilother</td>
    <td>82</td>
    <td>39</td>
    <td>99</td>
    <td></td>
    <td></td>
    <td>0</td>
    <td>3</td>
    <td>1</td>
    <td>4</td>
    <td>0</td>
    <td>2</td>
    <td>1</td>
    <td>3</td>
  </tr>
  <tr>
    <td>normal</td>
    <td>595</td>
    <td>236</td>
    <td>533</td>
    <td></td>
    <td></td>
    <td>6</td>
    <td>5</td>
    <td>8</td>
    <td>8</td>
    <td>8</td>
    <td>6</td>
    <td>8</td>
    <td>9</td>
  </tr>
  <tr>
    <td>occlude</td>
    <td>366</td>
    <td>137</td>
    <td>322</td>
    <td></td>
    <td></td>
    <td>2</td>
    <td>2</td>
    <td>3</td>
    <td>2</td>
    <td>3</td>
    <td>1</td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>paintmark</td>
    <td>354</td>
    <td>139</td>
    <td>339</td>
    <td></td>
    <td></td>
    <td>48</td>
    <td>35</td>
    <td>36</td>
    <td>39</td>
    <td>39</td>
    <td>30</td>
    <td>64</td>
    <td>23</td>
  </tr>
  <tr>
    <td>pending</td>
    <td>423</td>
    <td>158</td>
    <td>387</td>
    <td></td>
    <td></td>
    <td>4</td>
    <td>3</td>
    <td>5</td>
    <td>4</td>
    <td>2</td>
    <td>7</td>
    <td>4</td>
    <td>4</td>
  </tr>
  <tr>
    <td>pit</td>
    <td>507</td>
    <td>194</td>
    <td>437</td>
    <td></td>
    <td></td>
    <td>0</td>
    <td>0</td>
    <td>2</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
  </tr>
  <tr>
    <td>verticalfold</td>
    <td>147</td>
    <td>58</td>
    <td>127</td>
    <td></td>
    <td></td>
    <td>2</td>
    <td>0</td>
    <td>1</td>
    <td>0</td>
    <td>2</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
  </tr>
  <tr>
    <td>whitepoint</td>
    <td>733</td>
    <td>292</td>
    <td>711</td>
    <td></td>
    <td></td>
    <td>20</td>
    <td>6</td>
    <td>15</td>
    <td>5</td>
    <td>6</td>
    <td>14</td>
    <td>20</td>
    <td>11</td>
  </tr>
  <tr>
    <td>累计</td>
    <td>9400</td>
    <td>3673</td>
    <td>8516</td>
    <td></td>
    <td></td>
    <td>159</td>
    <td>159</td>
    <td>152</td>
    <td>152</td>
    <td>133</td>
    <td>133</td>
    <td>169</td>
    <td>169</td>
  </tr>
</tbody>
</table>

### 石墨-原材

<table id="Graphite_raw">
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBNet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="8" id="Graphite_raw">石墨-原材</td>
    <td>DZ</td>
    <td>189</td>
    <td>71</td>
    <td>175</td>
    <td></td>
    <td></td>
    <td>10</td>
    <td>2</td>
    <td>7</td>
    <td>3</td>
    <td>2</td>
    <td>7</td>
    <td>9</td>
    <td>2</td>
  </tr>
  <tr>
    <td>HS</td>
    <td>97</td>
    <td>50</td>
    <td>92</td>
    <td></td>
    <td></td>
    <td>6</td>
    <td>5</td>
    <td>7</td>
    <td>3</td>
    <td>5</td>
    <td>7</td>
    <td>5</td>
    <td>6</td>
  </tr>
  <tr>
    <td>PD</td>
    <td>63</td>
    <td>44</td>
    <td>87</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>5</td>
    <td>2</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
  </tr>
  <tr>
    <td>PS</td>
    <td>68</td>
    <td>27</td>
    <td>77</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>3</td>
    <td>1</td>
    <td>2</td>
    <td>5</td>
    <td>1</td>
    <td>2</td>
    <td>2</td>
  </tr>
  <tr>
    <td>QB</td>
    <td>56</td>
    <td>27</td>
    <td>56</td>
    <td></td>
    <td></td>
    <td>1</td>
    <td>1</td>
    <td>1</td>
    <td>2</td>
    <td>1</td>
    <td>2</td>
    <td>0</td>
    <td>2</td>
  </tr>
  <tr>
    <td>TK</td>
    <td>75</td>
    <td>45</td>
    <td>94</td>
    <td></td>
    <td></td>
    <td>3</td>
    <td>3</td>
    <td>1</td>
    <td>4</td>
    <td>3</td>
    <td>1</td>
    <td>2</td>
    <td>4</td>
  </tr>
  <tr>
    <td>TW</td>
    <td>114</td>
    <td>53</td>
    <td>105</td>
    <td></td>
    <td></td>
    <td>3</td>
    <td>6</td>
    <td>3</td>
    <td>7</td>
    <td>5</td>
    <td>4</td>
    <td>4</td>
    <td>7</td>
  </tr>
  <tr>
    <td>累计</td>
    <td>662</td>
    <td>317</td>
    <td>686</td>
    <td></td>
    <td></td>
    <td>25</td>
    <td>25</td>
    <td>22</td>
    <td>22</td>
    <td>23</td>
    <td>23</td>
    <td>24</td>
    <td>24</td>
  </tr>
</tbody>
</table>

<table id="#gf">
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBNet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="15" id="Photovoltaic_cells">光伏-电池片</td>
    <td>0-YL</td>
    <td>309</td>
    <td>122</td>
    <td>344</td>
    <td>50</td>
    <td>37</td>
    <td>17</td>
    <td>13</td>
    <td>21</td>
    <td>7</td>
    <td>14</td>
    <td>12</td>
    <td>11</td>
    <td>16</td>
  </tr>
  <tr>
    <td>1-BB</td>
    <td>11</td>
    <td>3</td>
    <td>7</td>
    <td>4</td>
    <td>3</td>
    <td>3</td>
    <td>2</td>
    <td>0</td>
    <td>7</td>
    <td>0</td>
    <td>0</td>
    <td>0</td>
    <td>7</td>
  </tr>
  <tr>
    <td>2-BianBuYin</td>
    <td>638</td>
    <td>254</td>
    <td>664</td>
    <td>20</td>
    <td>40</td>
    <td>16</td>
    <td>24</td>
    <td>17</td>
    <td>19</td>
    <td>22</td>
    <td>20</td>
    <td>18</td>
    <td>23</td>
  </tr>
  <tr>
    <td>3-Yin</td>
    <td>812</td>
    <td>307</td>
    <td>841</td>
    <td>91</td>
    <td>101</td>
    <td>92</td>
    <td>53</td>
    <td>92</td>
    <td>49</td>
    <td>40</td>
    <td>84</td>
    <td>87</td>
    <td>46</td>
  </tr>
  <tr>
    <td>4-HeiBan</td>
    <td>729</td>
    <td>286</td>
    <td>814</td>
    <td>92</td>
    <td>118</td>
    <td>55</td>
    <td>82</td>
    <td>37</td>
    <td>93</td>
    <td>76</td>
    <td>47</td>
    <td>52</td>
    <td>71</td>
  </tr>
  <tr>
    <td>5-HeiXian</td>
    <td>86</td>
    <td>37</td>
    <td>98</td>
    <td>19</td>
    <td>41</td>
    <td>23</td>
    <td>11</td>
    <td>13</td>
    <td>19</td>
    <td>14</td>
    <td>9</td>
    <td>17</td>
    <td>23</td>
  </tr>
  <tr>
    <td>6-ZangWu</td>
    <td>175</td>
    <td>70</td>
    <td>192</td>
    <td>22</td>
    <td>68</td>
    <td>21</td>
    <td>44</td>
    <td>35</td>
    <td>32</td>
    <td>49</td>
    <td>17</td>
    <td>31</td>
    <td>43</td>
  </tr>
  <tr>
    <td>7-TongXinYuan</td>
    <td>188</td>
    <td>81</td>
    <td>197</td>
    <td>32</td>
    <td>19</td>
    <td>16</td>
    <td>15</td>
    <td>19</td>
    <td>8</td>
    <td>19</td>
    <td>13</td>
    <td>16</td>
    <td>11</td>
  </tr>
  <tr>
    <td>8-HuaShang</td>
    <td>232</td>
    <td>86</td>
    <td>205</td>
    <td>48</td>
    <td>82</td>
    <td>28</td>
    <td>48</td>
    <td>36</td>
    <td>43</td>
    <td>48</td>
    <td>28</td>
    <td>27</td>
    <td>47</td>
  </tr>
  <tr>
    <td>9-GuoKe</td>
    <td>38</td>
    <td>14</td>
    <td>40</td>
    <td>8</td>
    <td>26</td>
    <td>13</td>
    <td>7</td>
    <td>4</td>
    <td>11</td>
    <td>10</td>
    <td>5</td>
    <td>4</td>
    <td>14</td>
  </tr>
  <tr>
    <td>10-KaDianYin</td>
    <td>17</td>
    <td>6</td>
    <td>9</td>
    <td>18</td>
    <td>1</td>
    <td>4</td>
    <td>2</td>
    <td>3</td>
    <td>2</td>
    <td>2</td>
    <td>6</td>
    <td>5</td>
    <td>1</td>
  </tr>
  <tr>
    <td>11-SiJiaoFaHei</td>
    <td>52</td>
    <td>19</td>
    <td>51</td>
    <td>29</td>
    <td>3</td>
    <td>7</td>
    <td>9</td>
    <td>9</td>
    <td>1</td>
    <td>5</td>
    <td>9</td>
    <td>10</td>
    <td>3</td>
  </tr>
  <tr>
    <td>12-ZhuShanFaHei</td>
    <td>171</td>
    <td>69</td>
    <td>162</td>
    <td>22</td>
    <td>5</td>
    <td>8</td>
    <td>2</td>
    <td>6</td>
    <td>3</td>
    <td>2</td>
    <td>8</td>
    <td>6</td>
    <td>2</td>
  </tr>
  <tr>
    <td>13-Other</td>
    <td>723</td>
    <td>270</td>
    <td>719</td>
    <td>112</td>
    <td>23</td>
    <td>51</td>
    <td>42</td>
    <td>38</td>
    <td>36</td>
    <td>15</td>
    <td>58</td>
    <td>52</td>
    <td>29</td>
  </tr>
  <tr>
    <td>累计</td>
    <td>4181</td>
    <td>1624</td>
    <td>4343</td>
    <td>567</td>
    <td>567</td>
    <td>354</td>
    <td>354</td>
    <td>330</td>
    <td>330</td>
    <td>316</td>
    <td>316</td>
    <td>336</td>
    <td>336</td>
  </tr>
</tbody>
</table>

<table >
<thead>
  <tr>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th colspan="2">FBNet</th>
    <th colspan="2">ResNet50</th>
    <th colspan="2">ResNet18</th>
    <th colspan="2">M</th>
    <th colspan="2">MobileNetV2</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>数据集</td>
    <td>类别</td>
    <td>train</td>
    <td>val</td>
    <td>test</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
    <td>类别漏检</td>
    <td>类别过检</td>
  </tr>
  <tr>
    <td rowspan="3" id="Soldered_dot">焊点</td>
    <td>0-YL</td>
    <td>1589</td>
    <td>368</td>
    <td>1455</td>
    <td></td>
    <td></td>
    <td>53</td>
    <td>46</td>
    <td>68</td>
    <td>46</td>
    <td>52</td>
    <td>32</td>
    <td>58</td>
    <td>52</td>
  </tr>
  <tr>
    <td>1-BB</td>
    <td>930</td>
    <td>615</td>
    <td>904</td>
    <td></td>
    <td></td>
    <td>46</td>
    <td>53</td>
    <td>46</td>
    <td>68</td>
    <td>32</td>
    <td>52</td>
    <td>52</td>
    <td>58</td>
  </tr>
  <tr>
    <td>累计</td>
    <td>2519</td>
    <td>983</td>
    <td>2359</td>
    <td></td>
    <td></td>
    <td>99</td>
    <td>99</td>
    <td>114</td>
    <td>114</td>
    <td>84</td>
    <td>84</td>
    <td>110</td>
    <td>110</td>
  </tr>
</tbody>
</table>

## 安装

请参考[快速入门文档](docs/zh_cn/get_started.md#installation)进行安装，参考[数据集准备](docs/zh_cn/user_guides/2_dataset_prepare.md)处理数据。

## 快速入门

请参考[概述](docs/zh_cn/overview.md)对 MMSegmetation 进行初步了解

请参考[用户指南](https://mmsegmentation.readthedocs.io/zh_CN/1.x/user_guides/index.html)了解 mmseg 的基本使用，以及[进阶指南](https://mmsegmentation.readthedocs.io/zh_CN/1.x/advanced_guides/index.html)深入了解 mmseg 设计和代码实现。

同时，我们提供了 Colab 教程。你可以在[这里](demo/MMSegmentation_Tutorial.ipynb)浏览教程，或者直接在 Colab 上[运行](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/1.x/demo/MMSegmentation_Tutorial.ipynb)。

若需要将0.x版本的代码迁移至新版，请参考[迁移文档](docs/zh_cn/migration.md)。

<!-- ## 工业数据增强和模型库

已经支持的数据增强：

- [x] 缺陷贴图
- [x] ....


已经支持的训练模型：
- [x] 双通道对比输入
- [x] 多通道输入多通道输出 -->

## 基准测试和模型库

测试结果和模型可以在[模型库](docs/zh_cn/model_zoo.md)中找到。

已支持的骨干网络：

- [x] ResNet (CVPR'2016)
<!-- - [x] ResNeXt (CVPR'2017) -->
<!-- - [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest) -->
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)

<!-- - [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3) -->
<!-- - [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae) -->
<!-- - [x] [PoolFormer (CVPR'2022)](configs/poolformer) -->

已支持的算法：

- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)
- [x] [MaskFormer (NeurIPS'2021)](configs/maskformer)
- [x] [Mask2Former (CVPR'2022)](configs/mask2former)

已支持的数据集：

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/zh_cn/dataset_prepare.md#isaid)

如果遇到问题，请参考 [常见问题解答](docs/zh_cn/notes/faq.md)。

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMSegmentation 所作出的努力。请参考[贡献指南](.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMSegmentation 是一个由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型，从而不断为开源社区提供贡献。



## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。





