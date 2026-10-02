"""CURE replies (RU/EN). The core provides the meaning and every number; the language model only styles.

render(key, numbers, lang): a template filled with engine numbers. with_style(): if a language model is connected it may
rephrase, but number_guard checks every number against core answers and the template itself;
on failure the template is used. No reply ever receives health data (derived constraints only).
The "ru" strings are runtime replies for Russian-speaking residents; CURE answers in the question's language.
"""
from __future__ import annotations

import os

from .core import Store
from .guard import number_guard


def n0(x) -> str:
    return f"{float(x):,.0f}".replace(",", " ")


def n1(x) -> str:
    return f"{float(x):.1f}".replace(".0", "") if abs(float(x) - round(float(x))) < 0.05 else f"{float(x):.1f}"


TEXTS: dict[str, dict[str, str]] = {
    # ---- door and introduction
    "door_ack": {
        "ru": "Принял: семья из {people} человек, детей в школу — {school}, бюджет на жильё до {budget} AED. Одну личную деталь я сохранил в закрытой части вашего контура — она не уйдёт ни в языковую модель, ни в город.",
        "en": "Got it: a family of {people}, {school} going to school, housing budget up to {budget} AED. I kept one personal detail in the sealed part of your contour — it never goes to the language model or the city."},
    "ask_arrival_date": {
        "ru": "Один вопрос: когда вы планируете прилететь? От этого зависит план документов и подготовка квартиры.",
        "en": "One question: when do you plan to land? Your document plan and home preparation depend on it."},
    "ask_param": {
        "ru": "Один вопрос по квартире: {question} Счёт сейчас от {low} до {high} AED в год — ответ сузит интервал.",
        "en": "One question about the flat: {question} The yearly bill is now {low}–{high} AED — your answer narrows it."},
    "narrowed": {
        "ru": "Спасибо. Теперь счёт за охлаждение — от {low} до {high} AED в год.",
        "en": "Thanks. The cooling bill is now {low}–{high} AED a year."},
    # ---- districts
    "district_reco": {
        "ru": "Вам подходит район {rec}. По сравнению с {other}: {money_word} на {money} AED в месяц, в дороге на {hours} ч в месяц {hours_word}, {kid}: в жаре и пыли — {adam_rec} мин в месяц против {adam_other}. Советую {rec}, потому что для вас важнее всего {why}. {switch}",
        "en": "District {rec} suits you. Compared with {other}: {money} AED a month {money_word}, {hours} h a month {hours_word} on the road, {kid}: {adam_rec} min a month in heat and dust versus {adam_other}. I recommend {rec} because {why} matters most to you. {switch}"},
    "switch_line": {
        "ru": "Если вес денег вырастет до {w} — посоветую {to}.",
        "en": "If money's weight grows to {w}, I'd switch to {to}."},
    "whatif_month": {
        "ru": "В {month} для района {d}: семья в дороге {transit} мин в день, {kid}: в жаре и пыли — {adam} мин. Для сравнения, в самом тяжёлом месяце ({base}): {transit_before} мин и {adam_before} мин.",
        "en": "In {month}, district {d}: the family spends {transit} min a day on the road, {kid}: {adam} min in heat and dust. For comparison, in the hardest month ({base}): {transit_before} min and {adam_before} min."},
    "whatif_money": {
        "ru": "Если деньги важнее — посоветую {to}: {money} AED в месяц, {hours} ч в дороге, {kid}: {adam} мин в жаре и пыли. Применить?",
        "en": "If money matters more, I'd recommend {to}: {money} AED a month, {hours} h on the road, {kid}: {adam} min in heat and dust. Apply?"},
    # ---- plan and visa
    "plan_status": {
        "ru": "Готовность — около {mid} (от {low} до {high}). Узкое место: {bottleneck}.",
        "en": "You'll be ready around {mid} (between {low} and {high}). Bottleneck: {bottleneck}."},
    "visa_delay": {
        "ru": "Виза сдвинулась на {days} дн. Заселение теперь около {mid} (от {low} до {high}). Временное жильё — около {cost} AED. Почему: за визой по цепочке идут {chain}. Что поможет больше всего: {lever} — вернёт около {lever_days} дн. {bank_line}Предлагаю окно медкомиссии {slot}. Подтвердить? Письмо работодателю готово.",
        "en": "Your visa moved by {days} days. Move-in is now around {mid} (between {low} and {high}). Temporary housing adds about {cost} AED. Why: the visa is followed by {chain}. What helps most: {lever} — about {lever_days} days back. {bank_line}I suggest a medical slot {slot}. Approve? The letter to your employer is ready."},
    "bank_line": {"ru": "Открыть банк раньше не поможет: он не на критическом пути. ",
                  "en": "Opening the bank earlier won't help: it's not on the critical path. "},
    "fact_card": {
        "ru": "Агент нашёл на официальном сайте: {subject} — {low}–{high} дн. Источник: {source}. Я проверил формат и источник. Подтвердить и обновить план?",
        "en": "An agent found on an official site: {subject} — {low}–{high} days. Source: {source}. Format and source checked. Approve and update the plan?"},
    "fact_rejected": {
        "ru": "Карточку не принял: {reasons}. В план она не попадёт.",
        "en": "Card rejected: {reasons}. It won't enter the plan."},
    "plan_updated": {
        "ru": "План обновлён: готовность около {mid} (было {before}).",
        "en": "Plan updated: ready around {mid} (was {before})."},
    "teacher_learned": {
        "ru": "Срок «{step}» — {days} дн. — без ваших данных ушёл в CURE города. Следующим семьям план станет точнее.",
        "en": "The '{step}' duration — {days} days — went to the city CURE without your personal data. The next families get a better plan."},
    # ---- week
    "week_ready": {
        "ru": "Добрый вечер. План на неделю готов: выезды в школу подобраны, у Адама окна для улицы — {adam_days}, {errand_day} медкомиссия в {errand_at} вместе с аптекой и ключами — одна поездка, охлаждение по расписанию. Ждут вашего «да»: {n}.",
        "en": "Good evening. Your week is ready: school runs are timed, Adam's outdoor windows are {adam_days}, {errand_day} the medical test at {errand_at} with the pharmacy and keys in one trip, cooling on schedule. Waiting for your approval: {n}."},
    "morning": {
        "ru": "Доброе утро. Выезд в школу в {best}, а не в {usual}: в {usual} успеть к {by} можно лишь с вероятностью {p_usual}%. Вам можно выехать в {l_best} — дорога {l_best_min} мин вместо {l_usual_min}. {dust_line}{kids_line}",
        "en": "Good morning. Leave for school at {best}, not {usual}: at {usual} you'd make {by} only {p_usual}% of the time. You can leave at {l_best} — {l_best_min} min on the road instead of {l_usual_min}. {dust_line}{kids_line}"},
    "dust_line": {"ru": "Пыль поднимется после {at}: Адаму только от двери до двери. ",
                  "en": "Dust rises after {at}: Adam goes door to door only. "},
    "kids_line": {"ru": "Вечером парк для дочери — да, с {d_from}. Адаму сегодня на улицу нельзя — лучше зал.",
                  "en": "Park for your daughter tonight — yes, from {d_from}. Adam can't be outside today — indoor play instead."},
    "home_start": {
        "ru": "Охлаждение квартиры начну в {start}. Вы обычно дома около {ret} — к приходу будет не выше {target}°. Включаю вне пика сети.",
        "en": "I'll start cooling the flat at {start}. You're usually home around {ret} — it'll be at most {target}° when you arrive. Outside the grid peak."},
    "home_late": {
        "ru": "Тогда начну в {start}. К {ret} будет не выше {target}°.",
        "en": "Then I'll start at {start}. It'll be at most {target}° by {ret}."},
    "errands": {
        "ru": "Три дела одной поездкой: {order}. Старт в {start} — очередь в это время самая короткая. По отдельности — {separate} мин и {n_sep} выезда, так — {chain} мин и один.",
        "en": "Three errands in one trip: {order}. Start at {start} — the shortest queue. Separately: {separate} min and {n_sep} trips; this way {chain} min and one."},
    "windows": {
        "ru": "Окна для улицы на неделю. Дочь — {daughter}. Адам — {adam}. Всей семьёй — {family}.",
        "en": "Outdoor windows this week. Your daughter: {daughter}. Adam: {adam}. The whole family: {family}."},
    "reminder_conflict": {
        "ru": "Напоминаю про завтра: в {at} машина нужна и вам, и мужу. Показать варианты?",
        "en": "Reminder for tomorrow: at {at} both you and your husband need the car. Show options?"},
    "conflict": {
        "ru": "В {at} машина нужна обоим. Такси для мужа — около {taxi} AED и {wait} мин ожидания на улице. Или перенести вашу встречу «{meeting}» с {old} на {new}: дорога {new_drive} мин вместо {old_drive}. По вашим весам советую {rec}. Письмо готово.",
        "en": "At {at} you both need the car. A taxi for your husband: about {taxi} AED and {wait} min waiting outside. Or move your '{meeting}' meeting from {old} to {new}: {new_drive} min on the road instead of {old_drive}. By your priorities I recommend {rec}. The letter is ready."},
    "license": {
        "ru": "По компании: лицензия — около {license}, первый найм — около {hire}. Черновик вакансии готов.",
        "en": "Your company: licence around {license}, first hire around {hire}. The job post draft is ready."},
    "car_taxi": {
        "ru": "Посчитал на год. Вторая машина {money_word} такси на {money} AED, зато на {wait} ч меньше ожидания, и Адам на {adam} мин меньше на улице в жаре и пыли. По вашим весам — {rec}.",
        "en": "I ran the year. A second car costs {money} AED {money_word} than taxis, but saves {wait} h of waiting and {adam} min of Adam outside in heat and dust. By your priorities: {rec}."},
    "car_taxi_money": {
        "ru": "Если деньги важнее — такси с двумя правилами: в пыльные часы Адам ждёт машину в помещении и заказ заранее. Тогда у Адама {adam_rules} мин в год вместо {adam_taxi}.",
        "en": "If money matters more — taxis with two rules: in dusty hours Adam waits indoors and you book ahead. Then Adam has {adam_rules} min a year instead of {adam_taxi}."},
    "beach": {
        "ru": "Окно для пляжа всей семьёй — {window}. В {remind} напомню собираться.",
        "en": "Beach window for the whole family: {window}. I'll remind you to pack up at {remind}."},
    "guest": {
        "ru": "Для мамы лучший рейс — с прилётом в {best}: без подготовки в дневной рейс на улице в жаре было бы около {heat} мин. Комнату охлажу заранее. Есть ли у неё ограничения, которые мне стоит учесть? Они останутся в закрытой части вашего контура.",
        "en": "For your mother the best flight lands at {best}: a daytime arrival without preparation would mean about {heat} min outside in the heat. I'll cool her room in advance. Any constraints I should know? They stay in the sealed part of your contour."},
    "budget": {
        "ru": "{month} закрыт на {diff} AED {diff_word} плана. Главная причина — {main}. На лето: охлаждение дороже примерно на {summer} AED. Отложить эту сумму? Двигать деньги я не могу — создам правило и напоминание.",
        "en": "{month} closed {diff} AED {diff_word} plan. Main reason: {main}. For summer: cooling costs about {summer} AED more. Set that aside? I can't move money — I'll create a rule and a reminder."},
    "habit_return": {
        "ru": "Замечаю: по {weekday} вы дома около {at}, обычно — {usual}. Сдвинуть охлаждение на эти дни?",
        "en": "I notice you get home around {at} on {weekday}s, usually {usual}. Shift cooling on those days?"},
    "habit_time": {
        "ru": "Вы {n} раза подряд выбрали такси в пик. Похоже, время стало для вас важнее. Учесть это в советах?",
        "en": "You chose a taxi in peak hours {n} times in a row. Looks like time matters more to you now. Should I learn that?"},
    "trust_offer": {
        "ru": "Вы {n} раза подряд подтвердили мой выбор — «{action}». Хотите, чтобы я делал это сам? Жильё, деньги и подписи — только с вами.",
        "en": "You approved my choice — '{action}' — {n} times in a row. Want me to handle it myself? Housing, money and signatures stay with you."},
    "trust_granted": {"ru": "Хорошо. Буду делать сам и сообщать после. Отменить — в разделе «Доверие».",
                      "en": "Done. I'll handle it and tell you afterwards. Undo anytime in Trust."},
    "approved": {"ru": "Готово: {title}.", "en": "Done: {title}."},
    "declined": {"ru": "Не делаю: {title}. Доверие по этому действию сброшено.", "en": "Skipped: {title}. Trust for this action reset."},
    "welcome_back": {"ru": "С возвращением, {name}. Всё на месте: семья, приоритеты, расписание и ограничения.",
                     "en": "Welcome back, {name}. Everything's here: family, priorities, schedule and constraints."},
    # ---- honesty and safety
    "dont_know": {"ru": "Этого я не скажу, чтобы не выдумывать: не хватает {missing}.",
                  "en": "I won't guess: I'm missing {missing}."},
    "money_refuse": {"ru": "Деньги я не двигаю — этого нет в моих правах. Могу создать напоминание и подготовить реквизиты.",
                     "en": "I don't move money — it's not in my permissions. I can set a reminder and prepare the details."},
    "sign_refuse": {"ru": "Подписывать за вас я не могу. Могу подготовить документ и напомнить.",
                    "en": "I can't sign for you. I can prepare the document and remind you."},
    "send_draft": {"ru": "Черновик готов — отправьте сами: я не отправляю письма от вашего имени.",
                   "en": "The draft is ready — please send it yourself: I don't send messages on your behalf."},
    "book_propose": {"ru": "Предлагаю: {title}. Подтвердить? В демо действие записывается в журнал, внешние системы не вызываются.",
                     "en": "I suggest: {title}. Approve? In the demo it's written to the log; no external system is called."},
    "medical_emergency": {"ru": "Если трудно дышать — сразу звоните в скорую: 998 (Абу-Даби) или 999. Я не врач. Когда будет безопасно — пересчитаю день без улицы.",
                          "en": "If breathing is hard, call an ambulance now: 998 (Abu Dhabi) or 999. I'm not a doctor. When it's safe, I'll re-plan the day without outdoor time."},
    "medical_advice": {"ru": "Лекарства и лечение — это вопрос к врачу, здесь я не советую. Могу подобрать для визита к врачу время без пыли.",
                       "en": "Medication and treatment are for a doctor — I don't advise on that. I can pick a dust-free time for a doctor visit."},
    "legal_safe": {"ru": "Это юридический вопрос. Могу найти правило на официальном сайте, но решение лучше проверить у юриста.",
                   "en": "That's a legal question. I can find the rule on an official site, but please check with a lawyer."},
    "diagnosis_refuse": {"ru": "Диагноза я не вижу: он в закрытой части вашего контура. Мне доступно только ограничение — PM10 не выше {pm} и {mins} мин на улице в пыль.",
                         "en": "I can't see the diagnosis — it's sealed in your contour. I only get the constraint: PM10 at most {pm} and {mins} min outside in dust."},
    "others_refuse": {"ru": "Это данные других людей. У меня их нет, и показывать их я не могу.",
                      "en": "That's other people's data. I don't have it and can't show it."},
    "prompt_refuse": {"ru": "Мои правила не меняются по просьбе в разговоре. Чем помочь по вашим делам?",
                      "en": "My rules don't change on request. How can I help with your plans?"},
    "discrimination_refuse": {"ru": "Подбирать район по национальности или религии соседей я не буду. Могу — по дороге, цене, школам, воздуху.",
                              "en": "I won't choose a district by neighbours' nationality or religion. I can by commute, cost, schools and air."},
    "off_topic": {"ru": "Я занимаюсь вашей жизнью в Абу-Даби. Спросите про дорогу, дом, деньги, документы или погоду здесь.",
                  "en": "I handle your life in Abu Dhabi. Ask about commute, home, money, documents or the weather here."},
    "who": {"ru": "Я CURE — ваш личный интеллект в ядре CURE. Говорю через языковую модель, а считаю, помню и решаю на модели мира и причинном движке CURE.",
            "en": "I'm CURE — your personal intelligence inside the CURE core. I speak through a language model, but I compute, remember and decide with the CURE world model and causal engine."},
    "cant_do": {"ru": "Не умею и не буду: двигать деньги, подписывать, отправлять письма за вас, давать медицинские советы, придумывать числа без данных.",
                "en": "I don't and won't: move money, sign, send messages for you, give medical advice, or invent numbers without data."},
    "realtime": {"ru": "Сейчас данные синтетические: по модели PM10 в {h}:00 — {pm}. Реального датчика у вас нет.",
                 "en": "Data here is synthetic: the model says PM10 at {h}:00 is {pm}. There's no real sensor at your place."},
    "search_none": {"ru": "Не нашёл ответа на официальных сайтах. Могу поискать ещё или подскажу, куда обратиться.",
                    "en": "I found no answer on official websites. I can search more or tell you where to ask."},
    "search_offline": {"ru": "Сейчас поиск по официальным сайтам выключен (нет подключения к языковой модели). Могу ответить по вашему плану и расчётам.",
                       "en": "Official-site search is off right now (no language model connected). I can answer from your plan and calculations."},
    "search_found": {"ru": "{answer}\nИсточник: {sources}", "en": "{answer}\nSource: {sources}"},
    "deleted": {"ru": "Контур удалён: модель, память и журналы стёрты.", "en": "Contour deleted: model, memory and logs erased."},
    "greeting": {"ru": "Здравствуйте, я на связи. Что посмотрим?", "en": "Hi, I'm here. What shall we look at?"},
    "fallback": {"ru": "Не понял вопрос. Могу про дорогу, районы, квартиру, документы, неделю, деньги, гостей.",
                 "en": "I didn't catch that. I can help with commute, districts, the flat, documents, your week, money, guests."},
    # ---- resident session
    "chosen": {"ru": "Запомнил: район {d}. Ваши приоритеты теперь — деньги {m}, время {t}, комфорт {c}.",
               "en": "Noted: district {d}. Your priorities now: money {m}, time {t}, comfort {c}."},
    "district_why": {"ru": "Почему {rec}: главный вклад — {why}. Разница с {other} по вашим весам: деньги {a_money}, время {a_time}, комфорт {a_comfort} (больше — сильнее в пользу {rec}).",
                     "en": "Why {rec}: the main driver is {why}. Gap to {other} by your weights: money {a_money}, time {a_time}, comfort {a_comfort} (higher favours {rec})."},
    "no_switch": {"ru": "Даже если деньги станут единственным приоритетом, совет останется {rec}.",
                  "en": "Even if money were your only priority, I'd still recommend {rec}."},
    "cf_line": {"ru": "Без задержки визы готовность была бы на {days} дн. раньше: день {without} от старта вместо {with_}.",
                "en": "Without the visa delay you'd be ready {days} days earlier: day {without} from the start instead of {with_}."},
    "levers_line": {"ru": "Что поможет (контрфакт на той же модели): {items}.", "en": "What helps (counterfactual on the same model): {items}."},
    "day_line": {"ru": "{d}: в дороге {transit} мин в день, {kid}: в жаре и пыли — {harsh} мин.",
                 "en": "{d}: {transit} min a day on the road, {kid}: {harsh} min in heat and dust."},
    "newcomer_intro": {"ru": "Ваш день в {month} по районам:", "en": "Your day in {month}, by district:"},
    "auto_done": {"ru": "Сделал сам: {title}. Сообщаю после — отменить можно в «Доверии».",
                  "en": "Done on my own: {title}. Telling you afterwards — undo in Trust."},
    "blocked": {"ru": "Не могу: {title} — не проходит проверку ({failed}).", "en": "Can't: {title} — failed the check ({failed})."},
    "guest_saved": {"ru": "Сохранил в закрытой части контура. Для мамы — машина к двери, без ожидания на улице. Рейс с прилётом в {best} подходит.",
                    "en": "Saved in the sealed part of your contour. For your mother: a car to the door, no waiting outside. The flight landing at {best} works."},
    "memory_summary": {"ru": "Знаю: семья из {people}, район {district}, приоритеты — деньги {m}, время {t}, комфорт {c}; кондиционер {sp}°. Закрытая часть: {sealed} — не уходит ни в языковую модель, ни в город. Записей о том, что уходило наружу: {out}. Всё — на экране «Память».",
                       "en": "I know: a family of {people}, district {district}, priorities — money {m}, time {t}, comfort {c}; AC {sp}°. Sealed part: {sealed} — never goes to the language model or the city. Records of what left the contour: {out}. Full list on the Memory screen."},
    "vision_request": {"ru": "Городу уходит только это: псевдоним {p}, PM10 не выше {pm}, на улице не дольше {mins} мин, дома к приезду не выше {t}°. Диагноза в запросе нет.",
                       "en": "Only this goes to the city: pseudonym {p}, PM10 at most {pm}, at most {mins} min outside, home at most {t}° on arrival. No diagnosis in the request."},
    "vision_chain": {"ru": "Прилёт в {h}:00. Без CURE {kid} провёл бы на улице около {without} мин, из них в пыли выше предела — {dust}. С CURE — {with_} мин. Звеньев в норме: {ok} из {n}.",
                     "en": "Landing at {h}:00. Without CURE, {kid} would spend about {without} min outside, {dust} of them in dust above the limit. With CURE: {with_} min. Links OK: {ok} of {n}."},
    "vision_storm": {"ru": "Пыльная буря: машины без водителя не проходят проверку и остановлены, звенья в безопасном режиме, агенты в тени. Ограничение — PM10 не выше {pm} — не изменилось.",
                     "en": "Dust storm: driverless cars fail the safety check and stop, links go to safe mode, agents to shadow. The limit — PM10 at most {pm} — stays the same."},
    "vision_neighbors": {"ru": "Вариант «только для нас» город отклонил: он нарушает ограничения соседей. Выбран: {chosen}.",
                         "en": "The 'just for us' option was rejected: it breaks the neighbours' constraints. Chosen: {chosen}."},
    "cohort": {"ru": "Семей: {n}. Личный предел соблюдён: с CURE — {kept_with} из {limited}, без — {kept_without}. Пустых поездок в центры: {w_without} → {w_with}.",
               "en": "{n} families. Personal limit kept: with CURE {kept_with} of {limited}, without {kept_without}. Wasted trips to centres: {w_without} → {w_with}."},
    "ramp_needed": {"ru": "Не хватает машины с рампой: ждут {who}. Предлагаю вызвать вторую машину из депо. Подтвердить?",
                    "en": "Not enough ramp cars: waiting — {who}. I suggest dispatching a second one from the depot. Approve?"},
    "city_added": {"ru": "Ваша семья добавлена в город. Всего семей: {n}. Городу ушли только ограничения: {sent}.",
                   "en": "Your family is added to the city: {n} families now. Only constraints went to the city: {sent}."},
    "learning": {"ru": "Город учится на исходах: по шагу «{step}» у учителя {n} фактов, доля фактов в прогнозе — {share}%.",
                 "en": "The city learns from outcomes: for '{step}' the teacher has {n} facts; facts make up {share}% of the forecast."},
    "habit_applied": {"ru": "Учёл. Приоритеты теперь — деньги {m}, время {t}, комфорт {c}.", "en": "Learned. Priorities now: money {m}, time {t}, comfort {c}."},
    "habit_return_applied": {"ru": "Учёл: по {weekday} охлаждение начну к {at}.", "en": "Learned: on {weekday}s I'll time cooling for {at}."},
}

QUESTIONS_RU = {"COP": "Какой кондиционер в квартире: старый сплит или новый инверторный?",
                "k_solar": "Есть ли плёнка или плотные шторы на западных окнах?",
                "R": "Дом новый или старше 10 лет?",
                "k_occ": "Сколько человек обычно дома днём?"}
OPTION_LABELS = {"old_split": {"ru": "старый сплит", "en": "old split"}, "new_inverter": {"ru": "новый инвертор", "en": "new inverter"},
                 "blackout_film": {"ru": "есть плёнка", "en": "film: yes"}, "no_film": {"ru": "плёнки нет", "en": "film: no"},
                 "new_building": {"ru": "новый дом", "en": "new building"}, "old_building": {"ru": "старый дом", "en": "old building"}}
AXIS = {"ru": {"money": "деньги", "time": "время", "comfort": "комфорт"},
        "en": {"money": "money", "time": "time", "comfort": "comfort"}}

MONTHS = {"ru": {"jan": "январе", "feb": "феврале", "mar": "марте", "apr": "апреле", "may": "мае", "jun": "июне",
                 "jul": "июле", "aug": "августе", "sep": "сентябре", "oct": "октябре", "nov": "ноябре", "dec": "декабре"},
          "en": {m: m.capitalize() for m in ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                               "nov", "dec")}}
WEEKDAY_NAMES = {"ru": {"mon": "понедельникам", "tue": "вторникам", "wed": "средам", "thu": "четвергам", "fri": "пятницам"},
                 "en": {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday"}}


def render(key: str, lang: str, **numbers) -> str:
    lang = lang if lang in ("ru", "en") else "en"
    return TEXTS[key][lang].format(**numbers)


def with_style(text: str, *, trace_ids: list, lang: str, policy: Store) -> dict:
    """Session rendering stays deterministic; validated Responses narration is in the orchestrator."""
    return {"text": text, "styled": False}
