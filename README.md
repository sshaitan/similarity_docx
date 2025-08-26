# similarity_docx

Инструмент для поиска похожих параграфов в документе Microsoft Word (`.docx`).
По умолчанию используется модель [BGE‑M3](https://huggingface.co/BAAI/bge-m3),
но через флаги `--backend` и `--model` можно выбрать любую модель из
`FlagEmbedding` или `sentence-transformers` (например Jina v3, GTE-multilingual,
Qwen-Embedding).

## Установка

```bash
pip install -r requirements.txt
```

## Запуск из консоли

Простой пример запуска:

```bash
python main.py --input myfile.docx --html report.html
```

Альтернатива с моделью из `sentence-transformers`:

```bash
python main.py --input myfile.docx --backend st --model jinaai/jina-embeddings-v3 --html report_jina.html
```

Полезные параметры:

* `--backend` – бэкенд эмбеддингов (`bge` или `st`).
* `--model` – название модели.
* `--threshold` – минимальный порог схожести (по умолчанию `0.88`).
* `--topk` – сколько кандидатов сравнивать для каждого параграфа (`5`).
* `--dedupe` – режим удаления дубликатов перед эмбеддингом
  (`off`, `exact` или `normalized`).

Текстовый отчёт будет сохранён в `similar.txt`, HTML‑версия – в файле,
указанном через `--html`.

## Графический интерфейс

Для тех, кто предпочитает окна:

```bash
python __init__.py
```

Откроется небольшое окно, где можно выбрать DOCX файл, задать порог
схожести и запустить анализ. По завершении рядом с исходным документом
появятся файлы отчёта (`.txt` и `.html`).

## Лицензия

MIT

