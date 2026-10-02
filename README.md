# CURE: backend + frontend

`backend/` — основной Python/FastAPI сервер. `front/` — интерфейс React/Electron.
Electron автоматически запускает сервер из соседней папки `backend`, ждёт успешные
`/health` и `/twin/state`, затем открывает интерфейс. Phone и Home hub используют
один сервер и общее состояние. При закрытии приложения его сервер останавливается.

## Запуск на Windows

В этой папке:

```powershell
.\CURE.cmd
```

Команда собирает актуальный интерфейс и запускает приложение.

Для установки на другом компьютере нужны Python 3.11 и Node.js 22.12+.
Распакуйте ZIP, запустите `INSTALL.cmd`, затем `CURE.cmd`. Установщик скачивает
зависимости, создаёт `backend/.venv` и собирает интерфейс; нужен интернет.
Альтернативная установка вручную:

```powershell
py -3.11 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
cd front
npm.cmd ci
cd ..
.\CURE.cmd
```

## Проверка интеграции

```powershell
cd front
npm.cmd run test:integration
```

Тест запускает реальные Electron и backend с временным хранилищем, проверяет
IPC, русский диалог, все сценарии недели, память, синхронизацию Home hub,
отрисовку и завершение процесса backend. Также проверяются пустое поле ввода,
создание профиля из пользовательского текста, дословное отображение ответов
и очистка устаревших данных при отключении сервера. Пользовательское хранилище
не используется. Успешный результат: строка `PASS` и код завершения 0.

## Как проверить backend вручную

В приложении нажмите `Backend: connected` в верхней панели. Раскроются реальные
JSON-ответы `/health` и `/twin/state` и время получения. `health.ready=true`
проверяет готовность конфигурации; интеграционный тест дополнительно проверяет
запросы, расчётные сценарии, состояние и его отображение.

При отдельном запуске через `backend/run.ps1`:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/twin/state | ConvertTo-Json -Depth 12
```

Интерактивные запросы доступны в http://127.0.0.1:8000/docs. При работе через
Electron порт выбирается автоматически; JSON можно посмотреть прямо в приложении.

Во frontend нет заранее заполненного семейного профиля, фиксированных дат недели
и статических ответов. При выборе Leila’s чат отправляет `rid: "leila"`, при выборе
New создаёт уникальный `guest-…`. Все сообщения, включая первое, отправляются
через `POST /twin/ask` с полями `rid`, `text`, `device`. Frontend показывает
серверные ответы. При потере связи отображается `Backend unavailable`,
а прежние данные очищаются. Демонстрационные YAML-источники backend сохранены:
ответ от работающего сервера не означает, что погода, тарифы и сценарии получены
из внешних источников.

## Отдельный сервер и настройки

```powershell
.\backend\run.ps1
```

API: http://127.0.0.1:8000/docs. Для подключения приложения к уже запущенному серверу:

```powershell
$env:CURE_BACKEND_URL = "http://127.0.0.1:8000"
.\CURE.cmd
```

Такой внешний сервер приложение не останавливает. `CURE_BACKEND_DIR` задаёт другой
каталог backend (относительно `front` или абсолютный путь). `CURE_PYTHON` задаёт
интерпретатор; автоматически проверяются виртуальные окружения backend, front и
корневой папки. `CURE_PORT` задаёт порт управляемого сервера; по умолчанию выбирается
свободный порт. Данные управляемого сервера находятся в `backend/core_store`;
`ARRIVAL_CORE_DIR` позволяет указать другое хранилище.

## Живые ответы нейросети

В `backend/.env` укажите `OPENAI_API_KEY` и перезапустите приложение.
Этот файл автоматически читается сервером; переменные окружения имеют приоритет.
Не отправляйте ключ в чат и не добавляйте его в Git.
`CURE_LANGUAGE_MODE=auto` включает OpenAI при наличии ключа; без ключа остаётся
локальный режим. `openai` требует ключ, `local` явно включает локальные шаблоны.
`OPENAI_MODEL` задаёт модель; по умолчанию используется уже настроенная `gpt-6.1-sol`.

В верхней панели `AI` означает настроенный режим OpenAI, `local` — локальные ответы.
Значок режима не проверяет доступность API или права на модель: это проверяется
при отправке сообщения. При ошибке API показывается ошибка; скрытой подмены
нейросетевого ответа шаблоном нет.

Модель объясняет подтверждённые результаты человеческим языком, уточняет профиль
и формулирует следующий шаг. Числа, даты, статус действий и ограничения проверяются
backend. В основном чате нет JSON и карточек `details`; технические данные остаются
в панели диагностики. При ожидании ответа показывается индикатор, повторная отправка
блокируется, а при ошибке введённый текст восстанавливается.

Поле сообщения растёт по высоте до 160 px, затем прокручивается внутри.
Enter отправляет, Shift+Enter добавляет новую строку. Пока CURE отвечает, можно
печатать следующий черновик; блокируется только повторная отправка. Обновления
истории сохраняют поле ввода и фокус. Отправленный текст виден сразу, затем
подтверждается серверной историей без дублирования.

Дополнительные настройки описаны в `backend/.env.example`.
Данные демонстрационные; внешние действия записываются в журнал.

## Mind и специалисты для новых задач

`Knows` теперь показывает Mind: текущую ситуацию, задачи, память с provenance,
inference и специалистов. В Console → Insights видны context/recall trace,
ссылки на использованные memories и фактическая история agent lifecycle.
Данные Лейлы из YAML сохраняют provenance `synthetic`, включая записанные там
реплики семьи. Новые пользовательские события получают `user`, результаты
модели — `inference`, research findings — `agent` и статус `pending`.

Mind находится в `Contour.extra.mind`, специалисты — в
`Contour.extra.dynamic_agents`, вместе с остальным зашифрованным Contour в
`backend/core_store` (`ARRIVAL_CORE_DIR`). Настоящий restart больше не вызывает
reset хранилища. Forget удаляет Mind вместе с resident. Прерванный при restart
агент становится `FAILED`, чтобы интерфейс не показывал исчезнувший worker
как работающий.

API: `GET /twin/mind`, `POST /twin/mind/recall`, `POST /twin/mind/infer`,
`POST /twin/agents/dynamic`, `GET /twin/agents/dynamic/{agent_id}`,
`POST /twin/agents/dynamic/{agent_id}/review`.
Для inference доступны `routine`, `availability`, `preference`, `current_situation`.
Recall комбинирует embeddings, связи с людьми/темами, importance, recency и
текущую тему. `OPENAI_EMBED_MODEL` по умолчанию `text-embedding-3-small`.
При ошибке embeddings recall явно возвращает локальный fallback и причину;
inference и specialist не подменяются локальным ответом.

В OpenAI режиме неизвестная задача в чате вызывает создание specialist spec
через Responses. Например: `I need to move my cat from Cairo to Abu Dhabi.
Figure out what I need.` Backend проверяет context, tools, sources и scope,
подписывает spec существующей HMAC схемой и выдаёт delegation contract.
Специалист выполняет hosted web search, затем отдельный Responses запрос
структурирует исследование. Подтверждённые runtime стадии видны в Phone/Console.
Timeout и writeback задаёт backend. Нет tools для отправки, оплаты, подачи,
подписи или бронирования. Findings не становятся подтверждёнными фактами или
численными параметрами расчётов. Ссылки проверяются по hostname и фактически
возвращённым search sources; содержание остаётся результатом агента для review.
Численные требования research пока не переносит: он направляет к официальной
странице. Это сохраняет границу существующего numerical gateway.

Review/writeback реализован на backend. Запрос принимает решения `approve`/`reject`
по отдельным `finding_id`. Перед записью CURE повторно проверяет подпись agent spec,
текущие grants/consent и привязку URL к фактически полученному official source.
Approved finding становится отдельной задачей в `Contour.extra.agent_tasks` с
provenance `agent`, отметкой `reviewed_by=user` и `affects_calculations=false`.
Он не изменяет hard constraints, профиль или численный relocation plan.
Frontend-кнопки для этого endpoint пока не подключены.

Все outbound contexts проходят существующий privacy boundary, расширенный
на вложенные поля и текст. Diagnosis/sealed/raw conversation/API keys не
передаются; физические constraints сохраняются. Responses использует
`store=false`, function loop остаётся в CURE. Локальный режим не вызывает API.

Проверки из корня проекта:

```powershell
backend/.venv/Scripts/python.exe -m pytest -q
cd front
npm.cmd run test:integration
cd ..
backend/.venv/Scripts/python.exe backend/tools/verify_live.py
```

Последняя команда выполняет реальные платные API requests с ключом из
`backend/.env`, использует изолированное временное resident storage и пишет
результат в `artifacts/live-verification.json`. Она проверяет answer/function,
embeddings, inference, unknown-task specialist, provider error и restart.
`OPENAI_REASONING_EFFORT=low` подходит для короткого demo; настройка остаётся
доступна через env. Исходный `MANIFEST.sha256.json` относится к архивному baseline,
а не к изменённым файлам.

Для specialist можно указать `OPENAI_MODEL_AGENT`; без него используется
`OPENAI_MODEL`. В текущем live-проходе основная модель — `gpt-6.1-sol`,
specialist — `gpt-5.4-mini`, reasoning — `low`. Обе модели проверены реальным
API. Выбор specialist модели сделан из-за нестабильных design timeouts на
основной модели; автоматической подмены при ошибках нет.

`cd front; npm.cmd run test:live` — отдельный opt-in тест с реальными API
requests из Phone: он проверяет lifecycle, результат и ссылки, Mind memory
и Console, сохраняет JSON и screenshot в `artifacts/`. Обычный
`test:integration` остаётся локальным и не расходует API.

## Передача проекта

ZIP содержит исходники backend и frontend, тесты, lock-файлы зависимостей,
шрифты с лицензией, `INSTALL.cmd` и `CURE.cmd`. Он не содержит `.env`, API-ключи,
пользовательское хранилище, виртуальные окружения, `node_modules`, кэши, логи
и промежуточные сборки. На новом компьютере используется чистое хранилище.
Для ответов нейросети получатель указывает собственный `OPENAI_API_KEY` в
`backend/.env` и `CURE_LANGUAGE_MODE=auto`, затем перезапускает приложение.
