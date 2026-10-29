# 项目网站

"从视频重建可度量的物理场景"的项目站。静态页面，无构建步骤。

| 文件 | 内容 |
|---|---|
| `index.html` | 首页：项目定位、流水线概览、数据集、验证集结果表、场景浏览入口 |
| `pipeline.html` | 流程图（单条链）、平面视频的处理、步骤表、答案来源 |
| `videos.html` + `videos.js` | 逐视频浏览：原视频、3D 场景（还没有的显示"待 v1 重跑"）、逐题答案与来源 |
| `site.css` | 首页和流水线页共用的样式 |
| `data/` | 场景浏览页的数据（`videos.json`、缩略图、场景、叠加图），由 `tools/build_videos.py` 生成 |
| `media/val`、`media/test` | 指向原视频目录的软链（不入 git） |

## 运行

```bash
./serve.sh start    # 端口 9002（PORT=... 可改），支持 Range，视频可拖动
./serve.sh status
./serve.sh stop
```

## 更新场景数据

在仓库根目录：

```bash
python -B web/tools/build_videos.py                    # 验证集：读当前代码版本（pipeline/VERSION）的 harness 运行结果
python -B web/tools/build_videos.py --val-run <tag>    # 指定运行标签
python -B web/tools/build_videos.py --test-sub <csv> --test-fit <dir>   # 同时重建测试集标签页
```

不给 `--test-sub` 时保留 `data/videos.json` 里已有的测试集数据。脚本只读已有产物，不运行流水线。

验证集结果表在 `index.html` 里手工更新，数字取自 `docs/VERSIONS.md`。
