# Установка

1. Склонировать репозиторий:

```
git clone https://github.com/SanchoPanso/monitoring_system.git
cd monitoring_system
git submodule update --recursive --init
```

2. Создать и активировать виртуальное окружение

```
python3 -m venv venv
source venv/Scripts/activate
```

3. Установить torch:

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Это один из способов установить torch c поддержкой cuda 11.6 (которая должна быть уже установлена). Поддержка cuda нужна для работы с gpu. При возникновении проблем можно обратиться к сайту библиотеки https://pytorch.org/

4. Установить прочие зависимости
```
pip install -r requirements.txt
```

# Использование

Запустить трекинг с камеры 0 с записью результатов в папку output:
```
python run_traking.py
```
Для остановки нужно нажать `Esc`.
Прочие аргументы запуска лежат в файле `arg_parser.py`.