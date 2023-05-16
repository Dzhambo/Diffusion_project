# Diffusion_project


## Краткое описание проекта:

* **Datasets - MNIST, CIFAR10**

* **Метрики качества - INCEPTION SCORE, FID** 

* **Генераторы обучаются с шагами - 500, 1000, 2000**

* **Построены графики метрик качества на каждом шаге генерации** 

* **Рассмотрены classifier guidance, classifier-free guidance** 

* **Рассмотрены разные стратегии управления дисперсией - linear, cosine, sigmoid** 


## Структура репозитория 

* **demo** - примеры использования имплементированных функций
* **duffusion_models** - имплементация диффузионных моделей DDPM, IDDPM, DDIM
* **reverse_models** - Unet
* **metrics** - метрики качества генерации
* **utils** - вспомогательные функции
* **pictures** - гифки диффузионного процесса
* **research** - исследования на датасетах CIFAR10, MNIST, FASHION MNIST
* **reports** - отчеты о проделанной работе