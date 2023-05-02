# Diffusion_project


Задачи данного проекты:

* Выбрать хотя бы два разных датасета 

    Было выбрано два датасета - MNIST, CIFAR10

* выбрать хотя бы две метрики качества 

    Было выбрано две метрики - bits per pixel, inception_score

* пообучать генераторы с разным количеством шагов (хотя бы 3-5 значений)

    Генераторы обучаются с шагами - 500, 1000, 2000

* сгенерировать примеры на тесте - и здесь так же интересно - что если мы остановим генерировать раньше , чем последний шаг по времени, который был использован для обучения (т.е. исследование early stopping on inference) 

    TO DO

* посмотреть на подтипы guidance (например, задавать класс объектов) с учетом сказанного выше + интересно, как влияют разные стратегии управления дисперсией при генерации (процедуры эволюции альфа)

    TO DO


## Структура репозитория 

* demo - ноутбуки с примерами как пользоваться имплементированными функциями
* duffusion_models - диффузионные модели(пока только ddpm)
* reverse_models - различные имплементации Unet
* metrics - метрики качества генерации
* utils - вспомогательные функции
* pictures - гифки диффузионного процесса
* research - исследования на разных датасетах