# dataset_studio

`dataset_studio` is a lightweight companion app for merging and curating
recognition datasets without changing the existing `shelf_mini` workflow.

Current MVP:

- discover recognition-style datasets under `/workspace/vdata`
- preview merges from a run export into a working dataset
- explicitly remap unknown classes before merge
- copy merged files safely with merge history
- browse classes and inspect images by split
- rename classes
- move images across splits or classes
- trash images into `.dataset_studio/trash` instead of deleting them

Run locally:

```bash
python3 apps/web/main.py
```

Run in Docker:

```bash
docker compose up --build -d dataset_studio
```
