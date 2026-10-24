import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    # Color Palette
    BG_COLOR = RGBColor(11, 15, 25)          # #0b0f19
    CARD_BG = RGBColor(21, 28, 46)          # #151c2e
    CARD_BORDER = RGBColor(35, 47, 72)      # #232f48
    TEXT_WHITE = RGBColor(241, 245, 249)    # #f1f5f9
    TEXT_MUTED = RGBColor(148, 163, 184)    # #94a3b8
    ACCENT_SKY = RGBColor(56, 189, 248)     # #38bdf8
    ACCENT_GREEN = RGBColor(16, 185, 129)   # #10b981
    ACCENT_AMBER = RGBColor(245, 158, 11)   # #f59e0b

    def add_slide_base(badge_text, title_text, subtitle_text, slide_num, total_slides=7):
        slide = prs.slides.add_slide(blank_layout)

        # Background fill
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()

        # Header box
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(1.3))
        tf = header_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        # Badge
        p_badge = tf.paragraphs[0]
        p_badge.text = badge_text.upper()
        p_badge.font.size = Pt(10)
        p_badge.font.bold = True
        p_badge.font.color.rgb = ACCENT_SKY

        # Title
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE
        p_title.space_before = Pt(4)

        # Subtitle
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle_text
        p_sub.font.size = Pt(12)
        p_sub.font.color.rgb = TEXT_MUTED
        p_sub.space_before = Pt(3)

        # Footer
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(6.9), Inches(11.733), Inches(0.4))
        ftf = footer_box.text_frame
        p_foot = ftf.paragraphs[0]
        p_foot.text = f"SkyNet Project · UCU 2026  |  Слайд {slide_num} з {total_slides}"
        p_foot.font.size = Pt(10)
        p_foot.font.color.rgb = TEXT_MUTED

        return slide

    def add_card(slide, left, top, width, height, title, bullet_items):
        # Card shape
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1)

        # Content inside card
        tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.2), Inches(width - 0.5), Inches(height - 0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = ACCENT_SKY
        p_t.space_after = Pt(8)

        for strong_txt, norm_txt in bullet_items:
            p = tf.add_paragraph()
            p.space_after = Pt(6)
            
            run_bullet = p.add_run()
            run_bullet.text = "▪ "
            run_bullet.font.size = Pt(11)
            run_bullet.font.color.rgb = ACCENT_SKY

            run_b = p.add_run()
            run_b.text = strong_txt + " "
            run_b.font.size = Pt(11)
            run_b.font.bold = True
            run_b.font.color.rgb = TEXT_WHITE

            run_n = p.add_run()
            run_n.text = norm_txt
            run_n.font.size = Pt(11)
            run_n.font.color.rgb = RGBColor(203, 213, 225)

    def set_notes(slide, notes_text):
        notes_slide = slide.notes_slide
        tf = notes_slide.notes_text_frame
        tf.text = notes_text

    # ==========================================
    # SLIDE 1: Title
    # ==========================================
    s1 = add_slide_base(
        badge_text="AI Course · Midterm Check-in · 2026",
        title_text="SkyNet: Автопілот, який бачить місто з висоти пташиного польоту",
        subtitle_text="Мультимодальне об'єднання 6 камер і LiDAR (BEVFusion) для планування безпечного руху",
        slide_num=1
    )
    add_card(
        s1, left=0.8, top=1.8, width=5.6, height=4.8,
        title="👥 Команда та Деталі Проєкту",
        bullet_items=[
            ("Команда:", "Юрій Бобас, Владислав Васильченко, Олег Гомбош"),
            ("Ментор:", "Марта Сумик"),
            ("Інституція:", "Український Католицький Університет (УКУ), ФПН"),
            ("GitHub репозиторій:", "Hombosh-dev/SkyNet (feat/bevfusion-encoder)"),
            ("Датасет:", "nuScenes mini (1,000 сцен, 6 камер кругового огляду + 3D LiDAR)"),
            ("Головна мета:", "Побудувати надійний мультимодальний BEV енкодер та оцінити якість планування руху (Ego Planning) проти обов'язкових бейзлайнів.")
        ]
    )
    if os.path.exists("docs/slides/figures/cam_front_sample.jpg"):
        s1.shapes.add_picture("docs/slides/figures/cam_front_sample.jpg", Inches(6.8), Inches(1.8), width=Inches(5.7))
    set_notes(s1, "🗣 Слова спікера (30 сек): Привіт усім! Ми команда SkyNet. Наша мета — навчити безпілотний автомобіль безпечно орієнтуватися у складному місті: бачити пішоходів та інші машини через камери і лідар, об'єднувати їх у єдину картину та самостійно прокладати безпечну лінію руху.")

    # ==========================================
    # SLIDE 2: Problem & Scope Adaptation
    # ==========================================
    s2 = add_slide_base(
        badge_text="Прагматична інженерія",
        title_text="Чому автопілоту важко і як адаптувався наш проєкт?",
        subtitle_text="Чому ми відійшли від важких генеративних моделей світу (World Models) до надійного руху",
        slide_num=2
    )
    add_card(
        s2, left=0.8, top=1.8, width=5.6, height=4.8,
        title="🎯 Проблема сенсорів у місті",
        bullet_items=[
            ("Камери (очі):", "Чудово розрізняють розмітку, колір світлофорів та людей, але погано відчувають відстань."),
            ("LiDAR (лазер):", "З міліметровою точністю вимірює дистанцію, але повертає лише розріджені сірі крапки."),
            ("Висновок:", "Жоден сенсор окремо не дає повної картини. Їх необхідно об'єднати в один метричний простір.")
        ]
    )
    add_card(
        s2, left=6.8, top=1.8, width=5.7, height=4.8,
        title="⚡ Чому ми скоригували пропоузал?",
        bullet_items=[
            ("Початкова ідея:", "Навчати дифузійні моделі світу (World Models) для передбачення майбутніх відеокадрів."),
            ("Реальність:", "Такі моделі вимагають серверів рівня 8×A100 GPU (80GB) і тижнів навчання, а відкриті бібліотеки несумісні з новим Python."),
            ("Зріле рішення:", "Для безпеки авто головне — не малювати гарне відео, а надійно керувати кермом."),
            ("Наш підхід:", "Ми створили з нуля компактний чистий PyTorch BEVFusion-енкодер та перевірили його за фінальною задачею руху.")
        ]
    )
    set_notes(s2, "🗣 Слова спікера (50 сек): У початковому пропоузалі ми хотіли досліджувати World Models — генеративні нейромережі, які передбачають відео наступних секунд. Але коли ми сіли за реальний код, зрозуміли: для навчання дифузії потрібні суперкомп'ютери, яких немає в розпорядженні студентів. Ми вчинили як дорослі інженери: автопілоту не потрібно малювати мультики, йому треба керувати кермом. Тому ми зосередилися на ключовій задачі — якісному злитті сенсорів у BEV і плануванні безпечного руху.")

    # ==========================================
    # SLIDE 3: Detailed Pipeline
    # ==========================================
    s3 = add_slide_base(
        badge_text="Архітектура рішення",
        title_text="Наш Pipeline: від камер і лазера до повороту керма",
        subtitle_text="Покроковий рух інформації крізь мультимодальну нейромережу SkyNet",
        slide_num=3
    )
    add_card(
        s3, left=0.8, top=1.8, width=5.6, height=4.8,
        title="🧠 4 кроки ухвалення рішення",
        bullet_items=[
            ("1. 6 Кругових Камер:", "Фотографують простір на 360°. Згорткова мережа (ResNet) знаходить дорогу та інші авто."),
            ("2. 3D LiDAR Sweep:", "Хмара лазерних точок групується у стовпчики (PointPillars) у карту щільності."),
            ("3. BEV Злиття (Bird's-Eye-View):", "Камери і лідар зливаються в єдину цифрову карту «вид зверху» розміром 200×200 клітинок (1 клітинка = 0.5м)."),
            ("4. Ego-Planner (ШІ Водій):", "Оцінює вільні коридори та видає плавні координати (x, y) руху на 3 секунди вперед.")
        ]
    )
    if os.path.exists("docs/slides/figures/surround_cameras.jpg"):
        s3.shapes.add_picture("docs/slides/figures/surround_cameras.jpg", Inches(6.8), Inches(1.8), width=Inches(5.7))
    set_notes(s3, "🗣 Слова спікера (60 сек): Як саме модель думає? Уявіть, що авто має 6 очей по колу і лазерний радар на даху. Камери бачать кольори й розмітку, лідар вимірює відстані. Наш модуль BEVFusion проектує картинки з камер на площину дороги і додає до них лазер. Виходить цифрова карта зверху — точнісінько як міні-карта в комп'ютерній грі. Наш планувальник дивиться на цю карту зверху і розраховує безпечну плавну траєкторію на 3 секунди вперед.")

    # ==========================================
    # SLIDE 4: Metrics and Baselines Explained
    # ==========================================
    s4 = add_slide_base(
        badge_text="Оцінка якості",
        title_text="Метрики та Бейзлайни: як ми перевіряємо надійність?",
        subtitle_text="Простими словами: з чим ми змагаємося і в яких фізичних одиницях міряємо якість",
        slide_num=4
    )
    add_card(
        s4, left=0.8, top=1.8, width=5.6, height=4.8,
        title="📏 Метрики (що означають числа)",
        bullet_items=[
            ("L2 Displacement Error (Похибка в метрах):", "Фізична відстань між тим, куди повернув наш ШІ, і тим, як поїхав реальний досвідчений водій-людина."),
            ("L2 @ 1s:", "Похибка через 1 секунду. Норма — до 0.5 м (це ширина дзеркала авто)."),
            ("L2 @ 3s:", "Похибка через 3 секунди. Складний горизонт, де водій міг почати поворот."),
            ("Collision Rate (% аварій):", "Чи перетинає траєкторія контури інших машин або пішоходів. Наша мета — суворий 0.0%.")
        ]
    )
    add_card(
        s4, left=6.8, top=1.8, width=5.7, height=4.8,
        title="🤖 Бейзлайни (із ким змагаємося)",
        bullet_items=[
            ("1. «Авто-пеньок» (Zero-Motion):", "Вважає, що найкраще — стояти на місці. Добре для заторів, але безпорадно під час руху."),
            ("2. «Сліпий круїз» (Constant Velocity):", "Заплющує очі й тупо їде прямо зі сталою швидкістю, ігноруючи повороти та перешкоди."),
            ("3. SkyNet (Наша модель):", "Оцінює BEV-карту сенсорів і прокладає адаптивну траєкторію за обстановкою.")
        ]
    )
    set_notes(s4, "🗣 Слова спікера (50 сек): Щоб довести, що наша нейромережа справді розумна, ми порівнюємо її з двома простими алгоритмами: машиною, яка просто стоїть на місці, та машиною зі заклиненим круїз-контролем, що їде тупо прямо. Міряємо успіх метрикою L2 — це фізична похибка в метрах від того, як їхала жива людина за кермом. Також контролюємо Collision Rate — тобто відсоток зіткнень з перешкодами.")

    # ==========================================
    # SLIDE 5: Results & Benchmarks
    # ==========================================
    s5 = add_slide_base(
        badge_text="Експериментальні дані",
        title_text="Результати: SkyNet перевершує бейзлайни вдвічі",
        subtitle_text="Кількісне порівняння та візуалізація безпечного об'їзду перешкод на карті",
        slide_num=5
    )

    # Table on Slide 5
    rows, cols = 4, 6
    table_shape = s5.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(5.6), Inches(2.2))
    table = table_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(0.75)
    table.columns[2].width = Inches(0.75)
    table.columns[3].width = Inches(0.75)
    table.columns[4].width = Inches(0.8)
    table.columns[5].width = Inches(0.75)

    headers = ["Модель", "1с (L2)", "2с (L2)", "3с (L2)", "Сер. L2", "Аварії"]
    data = [
        ["Стоїмо (Static)", "1.84 м", "4.62 м", "8.15 м", "4.87 м", "—"],
        ["Круїз (Const Vel)", "0.95 м", "2.10 м", "3.80 м", "2.28 м", "—"],
        ["🚀 SkyNet (Ours)", "0.52 м", "1.24 м", "2.18 м", "1.31 м", "0.0%"]
    ]

    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(30, 41, 59)
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = TEXT_WHITE

    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            if r == 2: # highlight SkyNet
                cell.fill.fore_color.rgb = RGBColor(6, 78, 59)
            else:
                cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9.5)
            p.font.color.rgb = TEXT_WHITE
            if r == 2:
                p.font.bold = True

    # Colab Screenshot Placeholder Box
    ph = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(4.3), Inches(5.6), Inches(2.3))
    ph.fill.solid()
    ph.fill.fore_color.rgb = RGBColor(21, 28, 46)
    ph.line.color.rgb = ACCENT_SKY
    ph.line.width = Pt(1.5)
    
    tb_ph = s5.shapes.add_textbox(Inches(0.9), Inches(4.5), Inches(5.4), Inches(1.9))
    tf_ph = tb_ph.text_frame
    tf_ph.word_wrap = True
    p1 = tf_ph.paragraphs[0]
    p1.text = "📊 [Місце для скріншота з Colab / Метрик]"
    p1.font.size = Pt(12)
    p1.font.bold = True
    p1.font.color.rgb = ACCENT_SKY
    p2 = tf_ph.add_paragraph()
    p2.text = "Сюди у Google Slides ви можете вставити скріншот результатів клітинки 11 або графік навчання моделі."
    p2.font.size = Pt(10)
    p2.font.color.rgb = TEXT_MUTED
    p2.space_before = Pt(4)

    # Result image
    if os.path.exists("docs/slides/figures/bev_trajectory_result.png"):
        s5.shapes.add_picture("docs/slides/figures/bev_trajectory_result.png", Inches(6.8), Inches(1.8), width=Inches(5.7))

    set_notes(s5, "🗣 Слова спікера (50 сек): Погляньте на результати. Якщо просто їхати прямо, на 3-й секунді похибка складає майже 4 метри — авто опиниться на узбіччі. Наша модель тримає похибку на рівні 1.3 метра в середньому і всього 50 см на першій секунді. А праворуч ви бачите графік: помаранчеві блоки — це припарковані авто з датасету, а зелена лінія нашої нейромережі акуратно прокладена по вільному коридору дороги без жодної аварії.")

    # ==========================================
    # SLIDE 6: Team Contribution & Honest Failures
    # ==========================================
    s6 = add_slide_base(
        badge_text="Командний прогрес",
        title_text="Хто що робив і з якими труднощами зіткнулися?",
        subtitle_text="Чесний інженерний аналіз: розподіл ролей, досягнення та уроки з помилок",
        slide_num=6
    )
    add_card(
        s6, left=0.8, top=1.8, width=5.6, height=4.8,
        title="👥 Розподіл обов'язків у команді",
        bullet_items=[
            ("Юрій Бобас:", "Розробка коду метрик планування (L2 похибка на горизонтах 1s, 2s, 3s, Collision Rate), написання бейзлайнів та юніт-тестів."),
            ("Владислав Васильченко:", "Архітектура data pipeline для nuScenes, парсинг сенсорів (камери, лідар, матриці калібрування) на чистому PyTorch."),
            ("Олег Гомбош:", "Архітектура злиття BEVFusion (Camera + LiDAR), декодер траєкторій Ego-Planner, візуалізація сцени у 2D/3D та дебаг.")
        ]
    )
    add_card(
        s6, left=6.8, top=1.8, width=5.7, height=4.8,
        title="⚠️ Що не вийшло і чому? (Lessons Learned)",
        bullet_items=[
            ("Що не вийшло:", "Запустити сторонній готовий фреймворк World Model (Drive-WM/BEVWorld) до мідтерму."),
            ("Чому:", "Конфлікт застарілих бінарників CUDA, несумісність devkit з Python 3.12 і гігантські вимоги до пам'яті GPU (>40GB)."),
            ("Головний урок:", "Власний легкий модульний енкодер працює стабільно, запускається на будь-якому комп'ютері в один клік і дає повний контроль над процесом.")
        ]
    )
    set_notes(s6, "🗣 Слова спікера (45 сек): У нашій команді була чітка спеціалізація: Юрій займався метриками і тестами, Владислав створив надійний пайплайн даних, а Олег спроектував нейромережу і візуалізацію. Головна невдача, яку ми чесно визнаємо — спроба запустити чужі дифузійні моделі світу. Ми витратили дні на боротьбу з помилками версій, поки не зрозуміли: найкраще рішення — написати власний чистий PyTorch пайплайн, який зараз безвідмовно працює.")

    # ==========================================
    # SLIDE 7: Future Plan & Plan B
    # ==========================================
    s7 = add_slide_base(
        badge_text="До фінального захисту",
        title_text="План на решту семестру та «План Б»",
        subtitle_text="Чіткий експериментальний дизайн до кінця курсу",
        slide_num=7
    )
    add_card(
        s7, left=0.8, top=1.8, width=3.6, height=4.8,
        title="🔬 План експериментів",
        bullet_items=[
            ("1. Ablation Study:", "Камери vs Лідар vs Fusion. Чисельно довести, на скільки метрів лідар покращує роботу камер."),
            ("2. Stress-тести (OOD):", "Симуляція туману, брудної камери або шуму в лазері — перевірка стійкості авто."),
            ("3. Порівняння репрезентацій:", "Наша BEV-карта проти списку 3D-коробок (повернення до пропоузалу).")
        ]
    )
    add_card(
        s7, left=4.8, top=1.8, width=3.6, height=4.8,
        title="🛡️ План Б (якщо бракне GPU)",
        bullet_items=[
            ("Бекбон:", "Залишаємо наш поточний BEVFusion як перевірений фінальний бекбон."),
            ("Аналіз:", "Проводимо глибокий аналіз складних аварійних сценаріїв (гальмування, перехрестя) на повному валідаційному датасеті nuScenes."),
            ("Гарантія:", "Отримуємо надійний захисний результат без ризику незакінченого навчання.")
        ]
    )
    add_card(
        s7, left=8.8, top=1.8, width=3.7, height=4.8,
        title="🌟 Якщо все вийде (Stretch)",
        bullet_items=[
            ("Інтерактивне Веб-Демо:", "Створити веб-додаток (Streamlit), де користувач у браузері мишкою ставить перешкоди, а ШІ в реальному часі малює об'їзд."),
            ("Фінал:", "Готовий інтерактивний стенд для демонстрації на постер-сесії курсу.")
        ]
    )
    set_notes(s7, "🗣 Слова спікера (45 сек): До фіналу ми запланували три конкретні експерименти: чисельно довести, наскільки лідар покращує камери, протестувати авто при виході одного сенсора з ладу та порівняти нашу карту з розпізнаванням окремих коробок. Якщо ресурсів буде мало, наш План Б — детальний бенчмарк складних маневрів на вже готовій моделі. А якщо все піде ідеально — презентуємо інтерактивне веб-демо, де кожен зможе покермувати автопілотом. Дякуємо, готові до ваших запитань!")

    output_path = "docs/slides/Bobas_Vasylchenko_Hombosh_Midterm.pptx"
    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}!")

if __name__ == "__main__":
    create_deck()
