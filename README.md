# red-hack-time-series-analyzer

Проект по созданию приложения для работы с времеными рядами и выявлению аномалий в них.

# Структура проекта

backend - бэкенд приложение
frontend - фронтенд приложение
notebooks - ноутбуки с обучением и оценкой разных моделей (CONV-AE, LSTM-AE, TranAD)

# Запуск проекта

## Production
Проект дотсупен в интернете по ссылке: https://germankoch.github.io/red-hack-time-series-analyzer/
Ссылка на эндпоинты бэкенд сервиса: https://time-series-outliers-detector-7d3cc62cf333.herokuapp.com/get-anomilies?time-series=RESPONSE&start=2024-04-15T23:32:00&end=2024-05-15T23:59:00


## Локально
Так же проект можно запустить локально. Необходимо запустить бэк и фронт.

Бэк:
```bash
pip install -r requirements-local.txt
python ./app.py
```

Фронт:
```bash
cd frontend
npm i
npm run dev
```


# Api
- Для получения точек аномалий для временного ряда между временными промежутками, необходимо выполнить GET запрос:
https://time-series-outliers-detector-7d3cc62cf333.herokuapp.com/get-anomilies?time-series=RESPONSE&start=2024-04-15T23:32:00&end=2024-05-15T23:59:00

Start-date и end-date могут быть переопределены. Параметр time-series определяет, какой временной ряд в расмотрении. Возможные значения параметра: RESPONSE, APDEX, THROUGHPUT, ERROR

- Для получения значений временного ряда, необходимо выполнить GET запрос:
https://time-series-outliers-detector-7d3cc62cf333.herokuapp.com/get-series?time-series=RESPONSE

Значения параметра time-series такие же как для предыдущего эндпоинта
