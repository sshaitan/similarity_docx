from tkinter import *
import logging
import tkinter.ttk as ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showwarning, showerror
import os
from old import get_paragraphs, get_max_paragraphs_len, calculate_str_len, get_model
import pickle
import random

logging.basicConfig(level=logging.DEBUG)

__author__ = "sshaitan"
__version__ = "0.1.0"
__license__ = "MIT"

MIN_PARAGRAPHS = 25
BATCH_SIZE = 12

PHRASES = ['попей чайку', 'сходи прогуляйся', 'похоже это на долго', 'а че там кстати с погодой',
           'ой, такие облака красивые', 'может кофейку?', 'оно ваще не повисло?', 'может забить',
           'посмотри инстаграмчик', 'даа, конечно супер быстро делается...',
           'ты кстати куришь? вот самое время...', 'почитай телегу',
           'блин, да че так медленно то...', 'а кнопки остановить все равно нету, хаха']


class DocxSimilarity(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.intro_text = None
        self.doc_file = None
        self.open_button = None
        self.filename = None
        self.index_file = None
        self.max_paragraph_len = None
        self.total_paragraphs = None
        self.paragraphs = []
        self.max_paragraph = 0
        self.model_length = 0
        self.model_length_label = None
        self.min_sim_label = None
        self.max_sim_label = None
        self.min_sim_entry = None
        self.min_sim_entry_value = DoubleVar()
        self.min_sim_entry_value.set(0.85)
        self.max_sim_entry = None
        self.max_sim_entry_value = DoubleVar()
        self.max_sim_entry_value.set(0.9999)
        self.index_button = None
        self.report_button = None
        self.progress_bar = None
        self.status_bar = None
        self.model_name = None
        self.clean_paragraphs = []
        self.short_report = None
        self.short_report_value = IntVar()
        self.short_report_value.set(1)
        self.is_stopped_indexing = False

        self.initialize()

    def open_file_dialog(self):
        self.total_paragraphs.config(text=f"Открываем документ...")
        self.max_paragraph_len.config(text=f"Открываем документ...")
        filename = fd.askopenfilename(filetypes=[("Документы Word 2010", ".docx")])
        if filename is not None and os.path.isfile(filename):
            self.model_name = os.path.splitext(filename)[0] + ".model"
            # проверим наличие файла-индекса
            if os.path.isfile(self.model_name):
                self.report_button.config(state=NORMAL)
                self.index_file.config(text=f'Файл индекса обнаружен: [{self.model_name}]')
                self.status_bar.config(text=f'Ура можно формировать отчет')
            else:
                self.status_bar.config(text=f'Готов индексировать')

            self.total_paragraphs.config(text=f"Анализ...")
            self.max_paragraph_len.config(text=f"Иди сдай анализы...")
            self.filename = filename
            self.doc_file.config(text=f"Выбран: [{filename}]")
            self.paragraphs = get_paragraphs(filename)
            self.total_paragraphs.config(text=f"Всего параграфов в документе: {len(self.paragraphs)}")
            self.max_paragraph = get_max_paragraphs_len(self.paragraphs)
            if self.max_paragraph > 8192:
                self.max_paragraph_len.config(text=f"Максимальная длинна параграфа: {self.max_paragraph}."
                                                   f" Слишком много текста в параграфе."
                                                   f" Могут возникнуть проблемы и снизится точность",
                                              fg="#FF0000")
                showwarning(title='Проблемы с документом обнаруживать!',
                            message='Добрый друг!\n'
                                    ' В твоем документе слишком длинные параграфы.\n'
                                    'Мы крайне рекомендуем сократить'
                                    ' текст в одном параграфе - разбить на два. \n'
                                    'Это повысить точность анализа '
                                    'и упростит чтение текста.')
            else:
                self.max_paragraph_len.config(text=f"Максимальная длинна параграфа: {self.max_paragraph}")
            self.model_length = calculate_str_len(self.max_paragraph)
            self.model_length_label.config(text=f"Выбранная размерность модели: {self.model_length}")
            self.index_button.config(state=NORMAL)

        return filename

    def block_ui(self):
        self.index_button.config(state=DISABLED)
        self.open_button.config(state=DISABLED)
        self.report_button.config(state=DISABLED)
        self.min_sim_entry.config(state=DISABLED)
        self.max_sim_entry.config(state=DISABLED)
        self.short_report.config(state=DISABLED)

    def unblock_ui(self):
        self.index_button.config(state=NORMAL)
        self.open_button.config(state=NORMAL)
        self.report_button.config(state=NORMAL)
        self.min_sim_entry.config(state=NORMAL)
        self.max_sim_entry.config(state=NORMAL)
        self.short_report.config(state=NORMAL)

    def make_index_stop(self):
        self.is_stopped_indexing = True
        self.block_ui()
        self.index_button.config(text="Индексировать", state=NORMAL, command=self.make_index)
        self.status_bar.config(text=f"Формирование индекса прервано. Готов.")
        self.unblock_ui()
        self.update_idletasks()
        self.update()

    def make_index(self):
        self.block_ui()
        self.status_bar.config(text="Загрузка модели машинного обучения...")
        model = get_model()
        self.update_idletasks()
        self.update()
        if os.path.exists(self.model_name):
            os.remove(self.model_name)
            self.index_file.config(text='Файл индекса не найден')
            self.update_idletasks()
            self.update()
        self.index_button.config(text="Остановка", state=NORMAL, command=self.make_index_stop)
        self.status_bar.config(text="Очистка слишком коротких параграфов...")
        paragraphs = [p for p in self.paragraphs if len(p) > MIN_PARAGRAPHS]
        self.clean_paragraphs = paragraphs
        self.status_bar.config(text=f"Всего будет обработано {len(paragraphs)} из {len(self.paragraphs)}")
        self.progress_bar.config(maximum=len(paragraphs))
        index = []
        adder = ""
        for num, paragraph in enumerate(paragraphs, start=1):
            if self.is_stopped_indexing:
                self.is_stopped_indexing = False
                return
            if num % 20 == 0:
                adder = random.choice(PHRASES)

            self.update_idletasks()
            self.update()
            index.append(model.encode(paragraph, batch_size=BATCH_SIZE, max_length=self.model_length)['dense_vecs'])
            self.progress_bar.config(value=num)
            self.status_bar.config(text=f"Всего будет обработано {len(paragraphs)} из {len(self.paragraphs)}, "
                                        f"делаю индекс: {num} {adder}")
        self.status_bar.config(text=f"Готово, сохраняю индекс {self.model_name}")
        with open(self.model_name, 'wb') as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.status_bar.config(text=f"Ура можно формировать отчет")
        self.unblock_ui()
        self.index_file.config(text=f'Файл индекса обнаружен: [{self.model_name}]')
        self.progress_bar.config(value=0)
        self.update_idletasks()
        self.update()

    def make_report(self):
        self.block_ui()
        is_filter_report = False if int(self.short_report_value.get()) == 1 else True
        if not os.path.isfile(self.model_name):
            showerror(title='Ошибка файл-индекса', message=f'Файл индекса поврежден или не найден.'
                                                           f' {self.model_name} Переиндексируйте.')
            self.report_button.config(state=DISABLED)
            return
        self.status_bar.config(text=f"Загрузка индекса для поиска")
        with open(self.model_name, 'rb') as handle:
            embeddings = pickle.load(handle)

        total = len(embeddings)
        self.progress_bar.config(maximum=total)
        result_file = os.path.splitext(self.model_name)[0] + '--result.txt'
        if os.path.exists(result_file):
            os.remove(result_file)

        m_min = float(str(self.min_sim_entry_value.get()).replace(',', '.'))
        m_max = float(str(self.max_sim_entry_value.get()).replace(',', '.'))

        paragraphs = [p for p in get_paragraphs(self.filename) if len(p) > MIN_PARAGRAPHS]
        self.update_idletasks()
        self.update()
        exists = []
        with open(result_file, 'w') as writer:
            for num, em in enumerate(embeddings):
                if num in exists and is_filter_report:
                    continue
                self.progress_bar.config(value=num)
                self.status_bar.config(text=f"проверяю параграф {num} из {total}")
                self.update_idletasks()
                self.update()
                for num_b, em_b in enumerate(embeddings):
                    similarity = em @ em_b.T
                    if m_min < similarity < m_max:
                        exists.append(num_b)
                        writer.write(f'Параграф: \n\n {paragraphs[num]} \n\n очень похож на параграф: \n\n '
                                     f'{paragraphs[num_b]} \n\n с точностью {similarity}'
                                     f' \n\n\n ============================== \n\n\n')
        self.progress_bar.config(value=0)
        self.status_bar.config(text=f"Файл отчета создан: {result_file}")
        self.unblock_ui()
        self.update_idletasks()
        self.update()
        os.system(result_file)

    def initialize(self):
        self.title(f""" Поиск одинаковых кусков текста в ворде (v{__version__}) """)
        self.intro_text = ttk.Label(text="""
        Данное приложение поможет Вам найти в документе Microsoft Word (DOCX)
        повторяющиеся по смыслу абзацы (параграфы) текста. 
        Щелкните "Открыть DOCX" и выполните проверку. После проверки вы сможете 
        просмотреть отчет. После индексации в папке с файлом будет создан файл индекса.
        Он используется только для формирования отчета. После получения отчета его можно удалить.
        Файл индекса будет создан с таким же именем и расширением .model
        НУЖЕН ДОСТУП В ИНТЕРНЕТ! Для обновления модели машинного обучения предоставьте доступ в Интернет.
        """)
        self.intro_text.grid(row=0, column=0, sticky=W, pady=10, padx=10)
        self.doc_file = ttk.Label(text="Файл не выбран.")
        self.doc_file.grid(row=1, column=0, sticky=W, pady=10, padx=10)
        self.open_button = ttk.Button(text="Открыть DOCX", command=self.open_file_dialog)
        self.open_button.grid(row=1, column=1, sticky=W, pady=10, padx=10)
        self.index_file = ttk.Label(text="Файл индекса не найден, требуется индексация")
        self.index_file.grid(row=2, column=0, sticky=W, pady=10, padx=10)
        self.max_paragraph_len = ttk.Label(text="Максимальная длинна параграфа: 0")
        self.max_paragraph_len.grid(row=3, column=0, sticky=W, pady=10, padx=10)
        self.total_paragraphs = ttk.Label(text="Всего параграфов в документе: 0")
        self.total_paragraphs.grid(row=4, column=0, sticky=W, pady=10, padx=10)
        self.model_length_label = ttk.Label(text="Выбранная размерность модели: недоступно")
        self.model_length_label.grid(row=5, column=0, sticky=W, pady=10, padx=10)
        self.short_report = ttk.Checkbutton(text="Отчет без фильтрации", variable=self.short_report_value)
        self.short_report.grid(row=5, column=1, sticky=W, pady=10, padx=10)
        self.min_sim_label = ttk.Label(text="Минимальный порог:")
        self.min_sim_label.grid(row=6, column=0, sticky=W, pady=10, padx=10)
        self.min_sim_entry = ttk.Entry(textvariable=self.min_sim_entry_value)
        self.min_sim_entry.grid(row=6, column=1, sticky=EW, pady=10, padx=10)
        self.max_sim_label = ttk.Label(text="Максимальный порог:")
        self.max_sim_label.grid(row=7, column=0, sticky=W, pady=10, padx=10)
        self.max_sim_entry = ttk.Entry(textvariable=self.max_sim_entry_value)
        self.max_sim_entry.grid(row=7, column=1, sticky=EW, pady=10, padx=10)
        self.index_button = ttk.Button(text="Индексировать", command=self.make_index)
        self.index_button.grid(row=8, column=0, sticky=W, pady=10, padx=10)
        self.index_button.config(state=DISABLED)
        self.report_button = ttk.Button(text="Отчет", command=self.make_report)
        self.report_button.grid(row=8, column=1, sticky=W, pady=10, padx=10)
        self.report_button.config(state=DISABLED)
        self.progress_bar = ttk.Progressbar(mode="determinate", orient=HORIZONTAL, length=800)
        self.progress_bar.grid(row=9, columnspan=2, sticky=W, pady=10, padx=10)
        self.status_bar = ttk.Label(text="Готов что-то делать, наверно надо открыть файл сперва...", relief=SUNKEN)
        self.status_bar.grid(row=10, columnspan=2, sticky=W, pady=10, padx=10)


app = DocxSimilarity()
app.mainloop()
