#!/usr/bin/env python3
"""Generate the Kaggle / Colab notebook templates from cell definitions.

Run: python scripts/make_notebooks.py
Output: notebooks/kaggle_lane.ipynb, notebooks/colab_lane.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def notebook(title: str, intro: str, cells: list) -> dict:
    return {
        "cells": [md(f"# {title}\n"), md(intro)] + cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


KAGGLE_INTRO = """\
Инструкция:
1. Включи Интернет (справа сверху: Settings -> Internet -> ON).
2. Загрузи этот блокнот в аккаунт, который будет сменой (kaggle-1 / kaggle-2).
3. В `Add-ons -> Secrets` добавь секрет `HF_TOKEN` (токен от Hugging Face
   c правами записи). Для каждой смены — свой токен/аккаунт.
4. Замени `YOUR_USERNAME` на свой GitHub-ник, а `YOUR_HF_USERNAME` — на свой HF-ник.
5. НЕ запускай две смены одновременно — они делят одну модель через lease.
6. Запусти все ячейки (Runtime -> Run all). Сессия сама сохранит и зальёт
   чекпоинт в конце бюджета, а при повторном запуске продолжит с того же места.
"""

COLAB_INTRO = """\
Инструкция:
1. Загрузи блокнот в Google Colab (Free).
2. Меню: панель слева -> ключик -> Add new secret -> `HF_TOKEN` (токен HF).
3. Замени `YOUR_USERNAME` и `YOUR_HF_USERNAME`.
4. Запусти все ячейки. То же самое: сессия сама всё сохранит.
"""

kaggle_cells = [
    code(
        "import os\n"
        "from kaggle_secrets import UserSecretsClient\n"
        "secret = UserSecretsClient()\n"
        'os.environ["HF_TOKEN"] = secret.get_secret("HF_TOKEN")\n'
    ),
    code(
        "# скачай код проекта (один раз на сессию)\n"
        "!git clone --depth 1 https://github.com/ExlentyEzz/mention.git /kaggle/working/mention\n"
        "!test -f /kaggle/working/mention/requirements-kaggle.txt && echo CLONE_OK || echo CLONE_FAILED\n"
        "%cd /kaggle/working/mention\n"
    ),
    code("!pip install -q -r requirements-kaggle.txt\n"),
    code(
        "!accelerate launch --num_processes=2 --mixed_precision=fp16 \\\n"
        "    scripts/run_lane.py \\\n"
        "    --config configs/pilot_40m.yaml \\\n"
        "    --lane kaggle-1 \\\n"
        "    --repo Melifyy/mention-ckpt\n"
    ),
]

colab_cells = [
    code(
        "import os\n"
        "from google.colab import userdata\n"
        'os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")\n'
    ),
    code(
        "!git clone --depth 1 https://github.com/ExlentyEzz/mention.git /content/mention\n"
        "!test -f /content/mention/requirements-kaggle.txt && echo CLONE_OK || echo CLONE_FAILED\n"
        "%cd /content/mention\n"
    ),
    code("!pip install -q -r requirements-kaggle.txt\n"),
    code(
        "!accelerate launch --num_processes=1 --mixed_precision=fp16 \\\n"
        "    scripts/run_lane.py \\\n"
        "    --config configs/pilot_40m.yaml \\\n"
        "    --lane colab \\\n"
        "    --repo Melifyy/mention-ckpt\n"
    ),
]


def _lane2_cells() -> list:
    return [
        code(
            "import os\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "secret = UserSecretsClient()\n"
            'os.environ["HF_TOKEN_KAGGLE2"] = secret.get_secret("HF_TOKEN_KAGGLE2")\n'
        ),
        code(
            "!rm -rf /kaggle/working/mention && git clone --depth 1 "
            "https://github.com/ExlentyEzz/mention.git /kaggle/working/mention\n"
            "!test -f /kaggle/working/mention/requirements-kaggle.txt && echo CLONE_OK "
            "|| echo CLONE_FAILED\n"
            "%cd /kaggle/working/mention\n"
        ),
        code("!pip install -q -r requirements-kaggle.txt\n"),
        code(
            "!accelerate launch --num_processes=2 --mixed_precision=fp16 \\\n"
            "    scripts/run_lane.py \\\n"
            "    --config configs/pilot_40m.yaml \\\n"
            "    --lane kaggle-2 \\\n"
            "    --repo Melifyy/mention-ckpt\n"
        ),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kaggle_lane.ipynb").write_text(
        json.dumps(notebook("Kaggle lane", KAGGLE_INTRO, kaggle_cells), indent=1),
        encoding="utf-8",
    )
    (OUT / "kaggle_lane2.ipynb").write_text(
        json.dumps(
            notebook("Kaggle lane 2", KAGGLE_INTRO, _lane2_cells()), indent=1
        ),
        encoding="utf-8",
    )
    (OUT / "colab_lane.ipynb").write_text(
        json.dumps(notebook("Colab lane", COLAB_INTRO, colab_cells), indent=1),
        encoding="utf-8",
    )
    print(
        "wrote",
        OUT / "kaggle_lane.ipynb",
        OUT / "kaggle_lane2.ipynb",
        OUT / "colab_lane.ipynb",
    )


if __name__ == "__main__":
    main()
