# Diffusion_project


Задачи данного проекты:

выбрать хотя бы два разных датасета (они могут быть и из компьютерного зрения, так и нет)
выбрать хотя бы две метрики качества (одну внутреннюю NLL, а другую внешнюю - например, качество распознавания by 3rd-party classifier)
пообучать генераторы с разным количеством шагов (хотя бы 3-5 значений)
сгенерировать примеры на тесте - и здесь так же интересно - что если мы остановим генерировать раньше , чем последний шаг по времени, который был использован для обучения (т.е. исследование early stopping on inference)
ну и посмотреть на подтипы guidance (например, задавать класс объектов) с учетом сказанного выше + интересно, как влияют разные стратегии управления дисперсией при генерации (процедуры эволюции альфа)
