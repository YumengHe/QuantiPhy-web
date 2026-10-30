# 项目网站

"从视频重建可度量的物理场景"的项目站。静态页面，无构建步骤。

| 文件 | 内容 |
|---|---|
| `index.html` | 首页：项目定位、流水线概览、数据集、验证集结果表、场景浏览入口 |
| `pipeline.html` | 流程图（单条链）、平面视频的处理、步骤表、答案来源 |
| `videos.html` + `videos.js` | 逐题浏览验证集：原视频、3D 场景、叠加图、scene graph、答案与来源 |
| `site.css` | 首页和流水线页共用的样式 |
| `data/` | 场景浏览页的数据（`videos.json`、缩略图；`runtime/<版本>/` 是该版本渲染的叠加图和记录 `overlays.json`），由 `tools/build_videos.py` 生成 |
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
python -B web/tools/build_videos.py                    # 当前版本（pipeline/VERSION）的验证集运行
python -B web/tools/build_videos.py --val-run <tag>    # 指定版本
```

脚本只读该版本自己的产物，不运行流水线，也不退回到别的版本或别的字段：格式按 `VERSION_FORMAT`（每个版本的运动记录字段、处理单位）读，不符就报错。叠加图只有当前版本可以现场补渲染；构建旧版本时缺图报错。新版本先在 `VERSION_FORMAT` 里登记。

验证集结果表在 `index.html` 里手工更新，数字取自 `docs/VERSIONS.md`。
