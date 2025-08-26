"""Simple Tkinter GUI for the paragraph similarity tool.

The GUI is intentionally lightweight. It lets a user pick a DOCX file and
runs :mod:`main` in a background thread while streaming progress output into a
scrollable text widget.  Results are written next to the selected document: a
``*.txt`` file with a human readable list of similar paragraph pairs and, if
``--html`` is supported by :mod:`main`, an ``*.html`` report.

Run the interface with ``python __init__.py`` or ``python -m similarity_docx``
when this directory is on ``PYTHONPATH``.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


def _run_similarity(
    docx: Path, threshold: float, backend: str, model: str, log: tk.Text
) -> None:
    """Execute :mod:`main` for ``docx`` and append output to ``log``."""
    out_txt = docx.with_suffix(".txt")
    out_html = docx.with_suffix(".html")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("main.py")),
        "--input",
        str(docx),
        "--threshold",
        str(threshold),
        "--backend",
        backend,
        "--model",
        model,
        "--output",
        str(out_txt),
        "--html",
        str(out_html),
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log.insert(tk.END, line)
        log.see(tk.END)
    proc.wait()
    messagebox.showinfo(
        "Готово", f"Отчёты сохранены рядом с документом:\n{out_txt}\n{out_html}"
    )


def main() -> None:
    root = tk.Tk()
    root.title("Поиск похожих параграфов")

    file_var = tk.StringVar()
    thr_var = tk.DoubleVar(value=0.88)
    backend_var = tk.StringVar(value="bge")
    model_var = tk.StringVar(value="BAAI/bge-m3")

    def choose_file() -> None:
        fn = filedialog.askopenfilename(filetypes=[("Word", "*.docx")])
        if fn:
            file_var.set(fn)

    def start() -> None:
        path = Path(file_var.get())
        if not path.is_file():
            messagebox.showerror("Ошибка", "Выберите DOCX файл")
            return
        log.delete("1.0", tk.END)
        th = threading.Thread(
            target=_run_similarity,
            args=(
                path,
                float(thr_var.get()),
                backend_var.get(),
                model_var.get(),
                log,
            ),
            daemon=True,
        )
        th.start()

    frm = tk.Frame(root)
    frm.pack(padx=10, pady=10, fill=tk.X)

    tk.Label(frm, text="DOCX файл:").grid(row=0, column=0, sticky="w")
    tk.Entry(frm, textvariable=file_var, width=50).grid(
        row=0, column=1, padx=5, sticky="we"
    )
    tk.Button(frm, text="Выбрать", command=choose_file).grid(row=0, column=2)

    tk.Label(frm, text="Порог схожести:").grid(row=1, column=0, sticky="w")
    tk.Entry(frm, textvariable=thr_var).grid(row=1, column=1, padx=5, sticky="we")

    tk.Label(frm, text="Бэкенд:").grid(row=2, column=0, sticky="w")
    tk.OptionMenu(frm, backend_var, "bge", "st").grid(row=2, column=1, padx=5, sticky="we")

    tk.Label(frm, text="Модель:").grid(row=3, column=0, sticky="w")
    tk.Entry(frm, textvariable=model_var, width=50).grid(
        row=3, column=1, padx=5, sticky="we"
    )
    tk.Button(frm, text="Старт", command=start).grid(row=3, column=2)

    log = scrolledtext.ScrolledText(root, width=80, height=20)
    log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    root.mainloop()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    main()

