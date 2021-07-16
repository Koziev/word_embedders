# Автоэнкодерные модели для векторизации слов

## Простой автоэнкодер

Тренировка модели автоэнкодера реализована на tensorflow (Keras) в [wordchar_simple_autoencoder.py](./py/wordchar_simple_autoencoder.py).  
Список слов, на котором учится модель, лежит в файле [words.txt](./data/words.txt).  

Готовая модель, обученная на этом датасете, уже лежит в каталоге tmp. Она используется в [experiments_with_simple_autoencoder.py](./py/experiments_with_simple_autoencoder.py).
Результаты экспериментов:

### Интерполяция между двумя словами


Для двух заданных слов получаются эмбеддинги, затем делается 10 равных шагов от точки первого слова до второго. Получающиеся
точки декодируются обратно в символьное представление.

```
муха => моха => хохе => хоне => хлон => слон
работать => работатаь => рабофатаь => рабонатать => безоеьтаат => безнеьбиать => бездельиаать => бездельниааь => бездельничать
земной => земнойй => гемнойс => гомнойсй => гомнийси => комниески => космичекий => космически => космический
```

### Визуализация эмбеддингов с помощью t-SNE

Размер кружочков коррелирует с длиной слов. Взята 1000 случайных слов среди встречающихся в [sents.txt](./data/sents.txt)


![большая картинка](./tmp/simple_ae.tsne.png)


### Поиск ближайших слов

```
123 => 193(0.975), 293(0.975), 12°(0.959), 192(0.957), 152(0.955), 181(0.939), 146(0.938), 170(0.936), см.(0.936), 197(0.933)
муха => зуха(0.983), мухе(0.973), руха(0.970), гуса(0.967), мова(0.963), леха(0.960), гоша(0.960), хуза(0.958), эрла(0.954), фука(0.953)
я => 4(0.976), }(0.947), д(0.910), ма(0.897), бо(0.870), ны(0.862), кю(0.856), 2й(0.851), ээ(0.828), фз(0.826)
голограмма => голограммах(0.982), волнограмма(0.962), докудрамах(0.941), диплограммам(0.940), кинодрамах(0.938), голобрюхие(0.934), голубеграмма(0.934), колораткам(0.933), дискограммы(0.932), говнороками(0.931)
среднегодовой => среднекраевой(0.962), средневосточной(0.950), кроненпробкой(0.942), скандализуемой(0.940), средневолжский(0.940), стенографируемой(0.940), предстартовой(0.940), фри-джазовый(0.937), кислотно-розовой(0.935), красно-черный(0.934)
крошечная => шишечная(0.959), крошечных(0.956), слиточная(0.944), карличная(0.944), сгущенная(0.944), сманенная(0.940), келейная(0.938), клеверная(0.937), приточная(0.936), тройчатная(0.936)
стоять => спреть(0.947), струан(0.941), троясь(0.935), строям(0.935), снуясь(0.932), сплоят(0.929), слышь(0.928), мриять(0.927), слоить(0.926), стоите(0.926)
прокумекав => прошуршав(0.943), постоматов(0.940), протухнув(0.940), прокрякав(0.938), проселков(0.937), промоушнов(0.936), прокаркав(0.935), просодемику(0.933), перепугав(0.932), продешевив(0.932)
```




