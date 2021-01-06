# Windows 10 tensorflow ile nesne tanıma uygulaması tensorflow GPU ile kullanılması. 

## Kullanım Klavuzu
*Yayınlama: 22/10/2020 ve TensorFlow v1.15*

Windows 10, 8 veya 7'de TensorFlow'un birden çok nesne için yapay sinir ağları ile nesne tanıma ve sınıflandırıcı kullanımını göstermek üzere yazılmıştır. Tez kapsamında ve eğitim esnasında TensorFlow sürüm 1.5 kullanılarak yazılmıştır, ancak TensorFlow'un daha yeni sürümleri için de çalışması gerekmektedir.

Bu okuma dosyası, kendi nesne algılama sınıflandırıcınızı kullanmaya başlamak için gereken her adımı açıklar.: 
1. [Anaconda, CUDA ve cuDNN yüklenmesi](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#1-install-anaconda-cuda-and-cudnn)
2. [Nesne Algılama dizin yapısını ve Anaconda Sanal Ortamını kurma](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
3. [Resimleri toplama ve etiketleme](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#3-gather-and-label-pictures)
4. [Eğitim verilerinin oluşturulması](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#4-generate-training-data)
5. [Bir etiket haritası oluşturma ve eğitimi yapılandırma](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#5-create-label-map-and-configure-training)
6. [Eğitim](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#6-run-the-training)
7. [Çıkarım grafiğini(the inference graph) dışa aktarma](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#7-export-inference-graph)
8. [Yeni eğitilmiş nesne algılama sınıflandırıcınızı test etme ve kullanma](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier)

[Ek: Karşılaşılan Yaygın Hatalar](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors)

Bu github kod paketinde tehlikeli nesne tanıma başlıklı tez kapsamında tanınması amaçlanan tehlikeli nesneleri tanımak için kullanılmıştır. Eğitim, bunun dışında da istenen her şey için bir tanıma sınıflandırıcısı eğitmek için bu dosyaları kendi dosyalarınızla nasıl değiştireceğinizi açıklar. Yapay sinir ağları ile farklı mimarileri de kullanılabilmektedir. Ayrıca tanıma ve sınıflandırıcınızı bir görüntü, video veya web kamerası yayınında test etmek için Python komut dosyalarına sahiptir.

## Giriş

Bu öğreticinin amacı, birden çok nesne tanıma için kendi evrişimli sinir ağı (Konvolüsyonel Sinir ağı) nesne tanıma ve sınıflandırıcınızı sıfırdan başlayarak nasıl eğiteceğinizi açıklamaktır. Bu eğitimin sonunda, resimlerdeki, videolardaki veya bir web kamerası ile alınan görüntülerdeki belirli nesnelerin etrafına kutuları tanımlayıp çizebilen bir programınız olacak.

Tek bir nesne için bir sınıflandırıcı eğitmek için TensorFlow’un Nesne Algılama API'sinin nasıl kullanılacağına ilişkin birkaç iyi eğitici bulunmaktadır. Ancak bunlar genellikle bir Linux işletim sistemi kullandığınızı varsayar. Object Detection API, Linux tabanlı bir işletim sistemi üzerinde geliştirilmiş görünüyor. TensorFlow'u Windows'ta bir model eğitmek üzere ayarlamak için, Linux'ta düzgün çalışan komutların yerine kullanılması gereken birkaç geçici çözüm vardır. Ayrıca bu eğitim, yalnızca bir nesneyi değil birden çok nesneyi algılayabilen bir sınıflandırıcı eğitimi için talimatlar sağlar.

Bu öğretim kapsamındaki kodlar Windows 10 için yazılmıştır ve Windows 7 ve 8 için de çalışacaktır. Genel prosedür Linux işletim sistemleri için de kullanılabilir, ancak dosya yolları ve paket kurulum komutlarının buna göre değiştirilmesi gerekecektir. Bu öğreticinin ilk sürümünü yazarken TensorFlow-GPU v1.5'i kullanılmıştır, ancak muhtemelen TensorFlow'un gelecekteki sürümleri için çalışacaktır.

TensorFlow-GPU, bilgisayarınızın eğitim sırasında ekstra işlem gücü sağlamak için video kartını kullanmasına izin verir, bu nedenle bu eğitim için bilgisayarınızın video kartı kullanılacaktır. Edinilen tecrübelere göre, normal TensorFlow yerine TensorFlow-GPU kullanmak, eğitim süresini azaltır. TensorFlow'un yalnızca CPU sürümü de bu öğretici için kullanılabilir, ancak daha uzun sürecektir. Yalnızca CPU TensorFlow kullanıyorsanız, 1. Adımdaki CUDA ve cuDNN'yi yüklemenize gerek yoktur.
## İzlenecek Adımlar

### 1. Anaconda, CUDA ve cuDNN yüklenmesi

[Mark Jay tarafından yayınlanan videoyu takip ederek](https://www.youtube.com/watch?v=RplXYjxgZbw); Anaconda, CUDA ve cuDNN'yi kurma sürecini gösteren kısımları izleyin. Videoda gösterildiği gibi gerçekten TensorFlow'u kurmanıza gerek yoktur, çünkü bunu 2. Adımda yapacağız. Video TensorFlow-GPU v1.4 için hazırlanmıştır, bu nedenle en son TensorFlow sürümü için CUDA ve cuDNN sürümlerini indirin ve yükleyin videoda belirtildiği gibi CUDA v8.0 ve cuDNN v6.0 yerine uygun versiyonları indirip yükleyim. [TensorFlow web sitesi](https://www.tensorflow.org/install/gpu), TensorFlow'un en son sürümü için hangi CUDA ve cuDNN sürümlerinin gerekli olduğunu belirtir.

TensorFlow'un daha eski bir sürümünü kullanıyorsanız, kullandığınız TensorFlow sürümüyle uyumlu CUDA ve cuDNN sürümlerini kullandığınızdan emin olun. [Burada](https://www.tensorflow.org/install/source#tested_build_configurations), TensorFlow'un hangi sürümünün CUDA ve cuDNN'nin hangi sürümlerini gerektirdiğini gösteren bir tablodur.

Videoda belirtildiği gibi [Anaconda](https://www.anaconda.com/distribution/#download-section) programını kurduğunuzdan emin olun, çünkü Anaconda sanal ortamı bu eğitimin geri kalanı için kullanılacaktır. (Not: Anaconda'nın mevcut sürümü, TensorFlow tarafından resmi olarak desteklenmeyen Python 3.7'yi kullanıyor. Bununla birlikte, bu eğiticinin Adım 2.d'si sırasında bir Anaconda sanal ortamı oluştururken, ona Python 3.5'i kullanmasını söyleyeceğiz.)

Diğer işletim sistemlerine (Linux gibi) nasıl kurulacağı da dahil olmak üzere daha fazla kurulum ayrıntıları için [TensorFlow'un web sitesini](https://www.tensorflow.org/install) ziyaret edin. [Nesne algılama havuzu içerisindeki dosyaları](https://github.com/tensorflow/models/tree/master/research/object_detection) içerisinde var olan [yükleme talimatları](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) kontrol edilerek yüklenebilir.

### 2. TensorFlow Dizini ve Anaconda Sanal Ortamını kurun

TensorFlow Object Detection API, GitHub deposunda sağlanan belirli dizin yapısının kullanılmasını gerektirir. Ayrıca, birkaç ek Python paketi, PATH(sistem yolu) ve PYTHONPATH(python yolu) değişkenlerine özel yollar ve  eklemeler ile bir nesne tanıma modelini çalıştırmak veya eğitmek için her şeyi ayarlamak için birkaç ekstra kurulum komutu gerektirir.

Eğitimin bu bölümü, gereken tüm kurulumu tatbik ederek üzerinden gider. Oldukça önemlidir bu yüzden talimatları yakından takip edin, çünkü yanlış kurulum sonuçta gereksiz hatalara neden olabilir.

#### 2.a GitHub'dan TensorFlow Nesne Algılama API(uygulama) kısmını indirin

Doğrudan C:\ sürücüsü içinde bir klasör oluşturun ve "tensorflow_kod" olarak adlandırın. Bu çalışma dizini, tam TensorFlow nesne algılama çerçevesinin yanı sıra eğitim görüntüleriniz, eğitim verileriniz, eğitimli sınıflandırıcı, yapılandırma dosyaları ve nesne algılama sınıflandırıcısı için gereken diğer her şeyi içerecektir.

https://github.com/tensorflow/models adresinde bulunan tam TensorFlow nesne algılama kısmını  "Clone" veya "Download" düğmelerinden birini tıklayarak ve zip dosyasını indirerek indirin. İndirilen zip dosyasını açın ve "models-master" klasörünü doğrudan yeni oluşturduğunuz C:\tensorflow_kod dizinine çıkartın. "Models-master" dosyasını "modeller" olarak yeniden adlandırın.

**Note: TensorFlow modellerinin kısım kodu (nesne algılama API(uygulama)'sini içerir) geliştiriciler tarafından sürekli olarak güncellenir. Bazen, TensorFlow'un eski sürümleriyle işlevselliği bozan değişiklikler yaparlar. TensorFlow'un en son sürümünü kullanmak ve en son modeller deposunu indirmek her zaman en iyisidir. En son sürümü kullanmıyorsanız, kullandığınız sürüm için yürütmeyi aşağıdaki tabloda listelendiği gibi kopyalayın veya indirin.**

TensorFlow'un eski bir sürümünü kullanıyorsanız, kullanmanız gereken deponun hangi GitHub kaydını kullanmanız gerektiğini gösteren bir tablo aşağıda verilmiştir. Bunu, modeller deposu için sürüm dallarına giderek ve dalın son işleminden önce kesinleştirerek oluşturdum. (Research klasörünü resmi sürüm yayınını oluşturmadan önce son kayıt olarak kaldırırlar.)

| TensorFlow versiyonu | Kaydedilen Uygun GitHub Modelleri |
|--------------------|---------------------------------|
|TF v1.7             |https://github.com/tensorflow/models/tree/adfd5a3aca41638aa9fb297c5095f33d64446d8f |
|TF v1.8             |https://github.com/tensorflow/models/tree/abd504235f3c2eed891571d62f0a424e54a2dabc |
|TF v1.9             |https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b |
|TF v1.10            |https://github.com/tensorflow/models/tree/b07b494e3514553633b132178b4c448f994d59df |
|TF v1.11            |https://github.com/tensorflow/models/tree/23b5b4227dfa1b23d7c21f0dfaf0951b16671f43 |
|TF v1.12            |https://github.com/tensorflow/models/tree/r1.12.0 |
|TF v1.13            |https://github.com/tensorflow/models/tree/r1.13.0 |
|Güncel Versiyon     |https://github.com/tensorflow/models |

Bu öğretim serisinde başlangıçta TensorFlow v1.5 ve TensorFlow Nesne Algılama API'sinin bu [GitHub kaydı](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) kullanılarak yapılmıştır. Bu öğreticinin bazı bölümleri işe yaramazsa, TensorFlow v1.5'i yüklemek ve en güncel sürüm yerine tam olarak bu işlemi kullanmak gerekebilir.


#### 2.b TensorFlow'un model havuzundan COCO veri setiyle yetiştirilmiş Faster R-CNN Inception Versiyon 2 mimarisine sahip "Faster-RCNN-Inception-V2-COCO" modelini indirin.

TensorFlow, [model havuzundan](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) çeşitli nesne algılama modelleri (belirli sinir ağı mimarilerine sahip önceden eğitilmiş sınıflandırıcılar) sağlar . Bazı modeller (SSD-MobileNet modeli gibi) daha hızlı algılamaya izin veren, ancak daha az doğruluk sağlayan bir mimariye sahipken, bazı modeller (Faster-RCNN modeli gibi) daha yavaş algılama ancak daha fazla doğrulukla sağlar. Tez kapsamında iki mimari de kullanılmıştır.

Nesne tanıma sınıflandırıcınızı hangi model mimarileri üzerinde eğiteceğinizi seçebilirsiniz. Nesne algılayıcıyı düşük hesaplama gücüne sahip bir cihazda (akıllı telefon veya Raspberry Pi gibi) kullanmayı planlıyorsanız, SSD-MobileNet modelini kullanın. Dedektörünüzü yeterince güçlü bir dizüstü veya masaüstü bilgisayarda çalıştıracaksanız, Faster R CNN modellerinden birini kullanın.

Bu eğitimde, Faster R-CNN mimarilerinden Faster-RCNN-Inception-V2 modelini kullanımı anlatılacaktır. [Modeli buradan indirin.](Http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz) İndirilen fast_rcnn_inception_v2_coco_2018_01_28.tar.gz dosyasını WinZip gibi bir Zip arşiviyle açın ve çıkartın fast_rcnn_inception_v2_coco_2018_01_28 klasörünü C:\tensorflow_kod\models\research\object_detection klasörüne indirin. (Not: Model tarihi ve sürümü gelecekte büyük olasılıkla değişecektir, ancak yine de bu eğitimle çalışacaktır.)

#### 2.c Eğitimi verilen bu github kod deposunu indirin.!!

Bu sayfada bulunan tam depoyu indirin (en üste kaydırın ve orada bulunan "Clone" veya "Download" butonuna tıklayarak indirin.) ve tüm içeriği doğrudan C:\tensorflow_kod\models\research\object_detection dizinine çıkarın. Bu, öğreticinin geri kalanı için kullanılacak belirli bir dizin yapısı oluşturur.

Bu noktada, \object_detection klasörünüz şöyle görünmelidir:

Bu depo, bir "Tehlikeli nesne" algılayıcısını eğitmek için gereken görüntüleri(images), ek açıklama verilerini(annotation data), .csv dosyalarını ve TFRecords dosyalarını içerir. Bu görüntüleri ve verileri tehlikeli nesne tanıma dedektörünüzü yapma alıştırması yapmak için kullanabilirsiniz. Ayrıca eğitim verilerini oluşturmak için kullanılan Python komut dosyalarını da içerir. Nesne algılama sınıflandırıcısını görüntüler, videolar veya bir web kamerası yayınında test etmek için komut dosyalarına sahiptir. \Doc klasörünü ve dosyalarını yok sayabilirsiniz; bu benioku(readme) dosyası için kullanılan resimleri tutmak için oradalar.

Kendi "Tehlikeli" nesne dedektörünüzü eğitmek istiyorsanız, tüm dosyaları olduğu gibi bırakabilirsiniz. Her bir dosyanın nasıl oluşturulduğunu görmek için bu öğreticiyi takip edebilir ve ardından eğitimi çalıştırabilirsiniz. Yine de 4.Adımda açıklandığı gibi TFRecord dosyalarını (train.record ve test.record) oluşturmanız gerekecektir.


Kendi nesne algılayıcınızı eğitmek istiyorsanız, aşağıdaki dosyaları silin (klasörleri silmeyin):

- \object_detection\images\train ve \object_detection\images\test içindeki tüm dosyalar
- \object_detection\images “test_labels.csv” ve “train_labels.csv” dosyaları
- \object_detection\training içindeki tüm dosyalar
- \object_detection\inference_graph içindeki tüm dosyalar


Artık kendi nesne dedektörünüzü eğitmeye sıfırdan başlamaya hazırsınız. Bu eğitici, yukarıda listelenen tüm dosyaların silindiğini varsayacak ve kendi eğitim veri kümeniz için dosyaları nasıl oluşturacağınızı açıklayacaktır.

#### 2.d Yeni Anaconda sanal ortamını kurun

Daha sonra, tensorflow-gpu için Anaconda'da sanal bir ortam kurmaya çalışacağız. Windows'ta Başlat menüsünden Anaconda Komut İstemi yardımcı programını arayın, sağ tıklayın ve "Yönetici Olarak Çalıştır" seçeneğini tıklayın. Windows, bilgisayarınızda değişiklik yapmasına izin vermek isteyip istemediğinizi sorarsa, Evet'i tıklayın.

Açılan komut terminalinde, aşağıdaki komutu vererek "tensorflow_kod" adlı yeni bir sanal ortam oluşturun:
```
C:\> conda create -n tensorflow_kod pip python=3.5
```
Ardından, ortamı etkinleştirin ve aşağıdakileri yayınlayarak pip'i güncelleyin:
```
C:\> activate tensorflow_kod

(tensorflow_kod) C:\>python -m pip install --upgrade pip
```
Bu ortama tensorflow-gpu kurun:
```
(tensorflow_kod) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```

(Not: TensorFow'un yalnızca CPU sürümünü de kullanabilirsiniz, ancak çok daha yavaş çalışacaktır. Yalnızca CPU sürümünü kullanmak istiyorsanız, önceki komutta "tensorflow-gpu" yerine "tensorflow" kullanın. )

Aşağıdaki komutları izleyerek diğer gerekli paketleri kurun:
```
(tensorflow_kod) C:\> conda install -c anaconda protobuf
(tensorflow_kod) C:\> pip install pillow
(tensorflow_kod) C:\> pip install lxml
(tensorflow_kod) C:\> pip install Cython
(tensorflow_kod) C:\> pip install contextlib2
(tensorflow_kod) C:\> pip install jupyter
(tensorflow_kod) C:\> pip install matplotlib
(tensorflow_kod) C:\> pip install pandas
(tensorflow_kod) C:\> pip install opencv-python
```

(Not: "pandas" ve "opencv-python" kütüphaneleri TensorFlow kütüphanesinin kullanımı içim gerekli değildir, ancak Python komut dosyalarında TFRecords ( Tensorflow kayıtları) oluşturmak ve görüntüler, videolar ve web kamerası beslemeleriyle çalışmak için kullanılırlar.

#### 2.e PYTHONPATH kurulan yazılımlarla eklenerek ortam değişkenini yapılandırın

\models, \models\research ve \models\research\slim dizinlerine yerini gösteren PYTHONPATH konum noktaları oluşturulmalıdır. Bunu, aşağıdaki komutları (herhangi bir dizinden) vererek yapın:
```
(tensorflow_kod) C:\> set PYTHONPATH=C:\tensorflow_kod\models;C:\tensorflow_kod\models\research;C:\tensorflow_kod\models\research\slim
```
(Not: "tensorflow_kod" sanal ortamından her çıkıldığında, PYTHONPATH değişkeni sıfırlanır ve yeniden ayarlanması gerekir. Ayarlanıp ayarlanmadığını görmek için "echo% PYTHONPATH%” 'i kullanabilirsiniz.)

#### 2.f Protobuf'ları derleyin ve setup.py'yi çalıştırın

Ardından, model ve eğitim parametrelerini yapılandırmak için TensorFlow tarafından kullanılan Protobuf dosyalarını derleyin. Maalesef, TensorFlow’un Nesne Algılama API'sinde [yükleme sayfası](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) yayınlanan kısa protokol derleme komutu Windows'ta çalışmıyor. \Object_detection\protos dizinindeki her .proto dosyası komut tarafından ayrı ayrı çağrılmalıdır.

Anaconda Komut İstemi'nde, dizinleri \models\research dizinine değiştirin:

```
(tensorflow_kod) C:\> cd C:\tensorflow1\models\research
```


Ardından aşağıdaki komutu kopyalayıp komut satırına yapıştırın ve Enter tuşuna basın:

```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto
```

Bu, \object_detection\protos klasöründeki her .proto dosyasından bir .py dosyası oluşturur.

**(Not: TensorFlow ara sıra yeni .proto dosyalarını \ protos klasörüne ekler. ImportError: cannot import name 'something_something_pb2' diyen bir hata alırsanız, protoc komutunu yeni .proto dosyalarını dahil edecek şekilde güncellemeniz gerekebilir.)**

Son olarak, C:\tensorflow_kod\models\research dizininden aşağıdaki komutları çalıştırın:

```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

#### 2.g Çalıştığını doğrulamak için TensorFlow kurulumunu test edin

TensorFlow Nesne Algılama API'sinin tümü, nesne tanıma için önceden eğitilmiş modelleri kullanmak veya yeni nesne tanıma sistemi eğitmek için ayarlanmıştır. Jupyter ile object_detection_tutorial.ipynb komut dosyasını başlatarak bunu test edebilir ve kurulumunuzun çalıştığını doğrulayabilirsiniz. \object_detection dizininden şu komutu verin:

```
(tensorflow1) C:\tensorflow_kod\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```

Bu, komut dosyasını varsayılan web tarayıcınızda açar ve kodda her seferinde bir bölümden geçmenize olanak tanır. Üst araç çubuğunda bulunan "Çalıştır" düğmesine tıklayarak her bir bölümde ilerleyebilirsiniz. Bölümün yanındaki "[*]" metni bir sayı ile doldurulduğunda bölüm çalıştırılır (ör. "[1]" de).

(Not: betiğin bir bölümü GitHub'dan yaklaşık 74MB olan ssd_mobilenet_v1 modelini indirir. Bu, bölümü tamamlamanın biraz zaman alacağı anlamına gelir, bu yüzden sabırlı olun.)

Komut dosyası boyunca adım adım ilerledikten sonra, sayfanın alt bölümünde iki etiketli resim görmelisiniz. Bunu görürseniz, her şey düzgün çalışıyor demektir! Değilse, alt bölüm karşılaşılan tüm hataları bildirecektir. Bu sırada karşılaştığım hataların listesi için [Ek](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors) sayfasına bakın.

**Not: Tam Jupyter Not Defterini herhangi bir hata almadan çalıştırırsanız, ancak etiketli resimler hala görünmüyorsa, şunu deneyin: object_detection/utils/visualization_utils.py adresindeki kod dosyasını açın ve 29 ve 30. satırların etrafındaki matplotlib içeren yorum satırını yorum olmaktan çıkarın.**

### 3. Resimleri Toplayın ve Etiketleyin


Artık TensorFlow Nesne Algılama API'sinin tamamı kurulduğuna ve kullanıma hazır olduğuna göre, yeni bir algılama sınıflandırıcısını eğitmek için kullanacağı görüntüleri sağlamamız gerekiyor.

#### 3a. Resimleri Toplayın

TensorFlow, iyi bir algılama sınıflandırıcısı yetiştirmek için bir nesnenin yüzlerce görüntüsüne ihtiyaç duyar. Sağlam bir sınıflandırıcı yetiştirmek için, eğitim görüntülerinin görüntüde istenen nesnelerle birlikte rastgele nesnelere sahip olması ve çeşitli arka planlara ve aydınlatma koşullarına sahip olması gerekir. İstenilen nesnenin kısmen örtüldüğü, başka bir şeyle örtüştüğü veya resmin yalnızca yarısında olduğu bazı görüntüler olmalıdır.

Telefonunuzu, nesnelerin fotoğraflarını çekmek veya nesnelerin resimlerini Google Görsel Arama'dan indirmek için kullanabilirsiniz. Genel olarak en az 200 fotoğrafın olmasını tavsiye edilir.

Görüntülerin çok büyük olmadığından emin olun. Her biri 200 KB'tan küçük olmalı ve çözünürlükleri 720x1280'den fazla olmamalıdır. Görüntüler ne kadar büyük olursa, sınıflandırıcıyı eğitmek o kadar uzun sürer. Görüntülerin boyutunu küçültmek için bu depodaki resizer.py komut dosyasını kullanabilirsiniz.

İhtiyacınız olan tüm resimleri aldıktan sonra, bunların %20'sini  \object_detection\images\test dizinine ve % 80'ini \object_detection\images\train dizinine taşıyın. Hem \test hem de \train dizinlerinde çeşitli resimler olduğundan emin olun.

#### 3b. Etiket Resimleri

İşin fazla emek ve zaman sarfettiren kısmı geldi! Tüm resimler bir araya geldiğinde, her resimde istenen nesneleri etiketleme zamanı. LabelImg, görüntüleri etiketlemek için harika bir araçtır ve GitHub sayfasında, nasıl kurulacağı ve kullanılacağı konusunda çok net talimatlar vardır.

[LabelImg GitHub link](https://github.com/tzutalin/labelImg)

[LabelImg download link](https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1)

LabelImg'ı indirin ve kurun, \images\train dizininize işaret edin ve ardından her bir görüntüdeki her nesnenin etrafına bir kutu çizin. \images\test dizinindeki tüm görüntüler için işlemi tekrarlayın. Bu işlem biraz zaman alacaktır!

LabelImg, her görüntü için etiket verilerini içeren bir .xml dosyası kaydeder. Bu .xml dosyaları, TensorFlow eğiticisinin girdilerinden biri olan TFRecords oluşturmak için kullanılacaktır. Her bir görüntüyü etiketleyip kaydettiğinizde, \test ve \train dizinlerindeki her görüntü için bir .xml dosyası olacaktır.

### 4. Eğitim Verilerini Oluşturun

Görüntülerle etiketlendiğinde, TensorFlow eğitim modeline girdi verisi olarak hizmet eden TFRecords'u oluşturmanın zamanı geldi. Bu eğiticide, dizin yapımızla çalışmak için bazı küçük değişikliklerle birlikte [Dat Tran’s Raccoon Detector veri kümesinden](https://github.com/datitran/raccoon_dataset) xml_to_csv.py ve generate_tfrecord.py komut dosyalarını kullanır.

İlk olarak, görüntü .xml verileri, train ve test görüntüleri için tüm verileri içeren .csv dosyalarını oluşturmak için kullanılacaktır. \object_detection klasöründen Anaconda komut isteminde aşağıdaki komutu çalıştırın:

```
(tensorflow_kod) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
```

Bu, \object_detection\images klasöründe bir train_labels.csv ve test_labels.csv dosyası oluşturur.


Ardından, bir metin düzenleyicide generate_tfrecord.py dosyasını açın. 31. satırdan başlayan etiket haritasını, her nesneye bir kimlik numarasının atandığı kendi etiket haritanızla değiştirin. Adım 5b'de labelmap.pbtxt dosyası yapılandırılırken aynı numara ataması kullanılacaktır.


Örneğin, basketbol toplarını, gömlekleri ve ayakkabıları tespit etmek için bir sınıflandırıcı eğittiğinizi varsayalım. Generate_tfrecord.py içinde aşağıdaki kodu değiştireceksiniz:

```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'piyade_t':
        return 1
    elif row_label == 'pistol':
        return 2
    else:
        None
```
Değiştireceğiniz kod
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        None
```

Ardından, \object_detection klasöründen bu komutları yayınlayarak TFRecords dosyalarını oluşturun:

```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

Bunlar, \object_detection konumunda bir train.record ve bir test.record dosyası oluşturur. Bunlar, yeni nesne algılama sınıflandırıcısını eğitmek için kullanılacaktır.

### 5. Etiket Haritası Oluşturun ve Eğitimi Yapılandırın

Eğitimden önce yapılacak son şey, bir etiket haritası oluşturmak ve eğitim yapılandırma dosyasını düzenlemektir.

#### 5a. Etiket haritası

Etiket haritası, sınıf adlarının sınıf kimliği numaralarına eşlenmesini tanımlayarak eğiticiye her nesnenin ne olduğunu söyler. Yeni bir dosya oluşturmak ve bunu C:\tensorflow_kod\models\research\object_detection\training klasörüne labelmap.pbtxt olarak kaydetmek için bir metin düzenleyici kullanın. (Dosya türünün .txt değil .pbtxt olduğundan emin olun!) Metin düzenleyicide, etiket haritasını aşağıdaki biçimde kopyalayın veya yazın:

```
item {
  id: 1
  name: 'piyade_t'
}

item {
  id: 2
  name: 'pistol'
}

```

Etiket eşlemesi kimlik numaraları, generate_tfrecord.py dosyasında tanımlananlarla aynı olmalıdır. 4. Adımda bahsedilen basketbol, gömlek ve ayakkabı dedektörü örneği için, labelmap.pbtxt dosyası şöyle görünecektir:

```
item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}

item {
  id: 3
  name: 'shoe'
}
```

#### 5b. Eğitimi yapılandırın

Son olarak, nesne algılama eğitim konfigürasyon yapılandırılmalıdır. Eğitim için hangi modeli ve hangi parametrelerin kullanılacağını tanımlar. Bu, eğitimi yapmadan önceki son adımdır!

C:\tensorflow_kod\models\research\object_detection\samples\configs konumuna gidin ve faster_rcnn_inception_v2_pets.config dosyasını \object_detection\training dizinine kopyalayın. Ardından, dosyayı bir metin düzenleyiciyle açın. .config dosyasında, esas olarak sınıfların ve örneklerin sayısını değiştirerek ve dosya yollarını eğitim verilerine ekleyerek yapılacak birkaç değişiklik vardır.

faster_rcnn_inception_v2_pets.config dosyasında aşağıdaki değişiklikleri yapın. Not: Yollar tek eğik çizgilerle (ters eğik çizgiler DEĞİL) girilmelidir, aksi takdirde TensorFlow modeli eğitmeye çalışırken bir dosya yolu hatası verir! Ayrıca, yollar tek tırnak işareti (') değil, çift tırnak işareti (") içinde olmalıdır.

- Satır 9. num_classes, sınıflandırıcının algılamasını istediğiniz farklı nesnelerin sayısına değiştirin. Yukarıdaki basketbol, gömlek ve ayakkabı detektörü için num_classes: 3 olacaktır.

- Satır 106. fine_tune_checkpoint'i şu şekilde değiştirin:
    - fine_tune_checkpoint : "C:/tensorflow_kod/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

-  123 ve 125. satırlar. train_input_reader bölümünde, input_path ve label_map_path değerlerini şu şekilde değiştirin::
    - input_path : "C:/tensorflow_kod/models/research/object_detection/train.record"
    - label_map_path: "C:/tensorflow_kod/models/research/object_detection/training/labelmap.pbtxt"

-  Satır 130 daki  num_examples değişkenini \images\test dizinindeki görüntülerin sayısına değiştirin.

-  135 ve 137. satırlar. eval_input_reader bölümünde, input_path ve label_map_path değerlerini şu şekilde değiştirin:
    - input_path : "C:/tensorflow1/models/research/object_detection/test.record"
    - label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Değişiklikler yapıldıktan sonra dosyayı kaydedin. Bu kadar! Eğitim işi tamamen yapılandırıldı ve kullanıma hazır!

### 6. Eğitimi çalıştırma zamanı

*1.9 sürümünden itibaren, TensorFlow "train.py" dosyasını kullanımdan kaldırmış ve onu "model_main.py" dosyasıyla değiştirmiştir. Henüz model_main.py'nin düzgün çalışmasını sağlayamadım pycocotools kurulu olarak kullanılabilir. Bunun yerine train.py dosyası da /object_detection/legacy klasöründe hala mevcuttur. Train.py'yi /object_detection/legacy'den /object_detection klasörüne taşıyın ve ardından aşağıdaki adımları izlemeye devam edin.*


İşte başlıyoruz! \Object_detection dizininden, eğitime başlamak için aşağıdaki komutu verin:

```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```


Her şey doğru ayarlanmışsa ve düzgün çalışırsa, TensorFlow eğitimi başlatır. Gerçek eğitimin başlaması 30 saniye kadar sürebilir. Eğitim başladığında şöyle görünecek:

Eğitimin her adımı loss (kayıp) değerini rapor eder. Loss değeri eğitim ilerledikçe azalacaktır. Faster-RCNN-Inception-V2 modeliyle ilgili loss değeri eğitimin başında yaklaşık 3.0'da başladı ve hızla 0.8'in altına düştü. Modelinizin, loss (kayıp) değeri sürekli olarak 0,05'in altına düşene kadar eğitilmesine izin vermenizi öneririm, bu yaklaşık 40.000 adım veya yaklaşık 2 saat sürer (CPU'nuzun ve GPU'nuzun ne kadar güçlü olduğuna bağlı olarak). Not: Farklı bir model kullanılırsa kayıp numaraları farklı olacaktır. MobileNet-SSD yaklaşık 20 loss değeri ile başlar ve kayıp sürekli olarak 2'nin altına düşene kadar eğitilmelidir.


TensorBoard'u kullanarak eğitim işinin ilerlemesini görüntüleyebilirsiniz. Bunu yapmak için, Anaconda İsteminin yeni bir örneğini açın, tensorflow_kod sanal ortamını etkinleştirin, C:\tensorflow_kod\models\research\object_detection dizinine geçin ve aşağıdaki komutu verin:


```
(tensorflow_kod) C:\tensorflow_kod\models\research\object_detection>tensorboard --logdir=training
```

Bu, yerel makinenizde Kişisel BilgisayarAdı: 6006'da bir web sayfası oluşturacaktır ve bu bir web tarayıcısı aracılığıyla görüntülenebilir. TensorBoard sayfası, eğitimin nasıl ilerlediğini gösteren bilgiler ve grafikler sağlar. Önemli bir grafik, sınıflandırıcının zaman içindeki genel kaybını gösteren Kayıp grafiğidir.

Eğitim rutini, kontrol noktalarını her beş dakikada bir periyodik olarak kaydeder. Komut istemi penceresindeyken Ctrl + C tuşlarına basarak eğitimi sonlandırabilirsiniz. Eğitimi sonlandırmak için genellikle bir kontrol noktası kaydedildikten hemen sonrasını beklerim. Eğitimi sonlandırıp daha sonra başlatabilirsiniz; son kaydedilen kontrol noktasından başlayacaktır. En yüksek adım sayısındaki kontrol noktası, dondurulmuş çıkarım grafiğini oluşturmak için kullanılacaktır.


### 7.  Inference Graph dışarı aktarma

Eğitim tamamlandığına göre, son adım donmuş çıkarım grafiğini (.pb dosyası) oluşturmaktır. \ Object_detection klasöründen aşağıdaki komutu çalıştırın; burada “model.ckpt-XXXX” içindeki “XXXX” eğitim klasöründeki en yüksek numaralı .ckpt dosyasıyla değiştirilmelidir:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

Bu, \object_detection\inference_graph klasöründe bir frozen_inference_graph.pb dosyası oluşturur. (.Pb) dosyası, nesne algılama sınıflandırıcısını içerir.

### 8. Yeni Eğitilmiş Nesne Algılama Sınıflandırıcınızı Kullanın!

Nesne algılama sınıflandırıcısı kullanıma hazır! Bir görüntü, video veya web kamerası akışında test etmek için Python komut dosyaları bulunmaktadır.

Python komut dosyalarını çalıştırmadan önce, komut dosyasındaki NUM_CLASSES değişkenini, algılamak istediğiniz sınıfların sayısına eşit olacak şekilde değiştirmek gerekir.

Nesne algılayıcınızı test etmek için, nesnenin veya nesnelerin bir resmini \object_detection klasörüne taşıyın ve Object_detection_image.py'deki IMAGE_NAME değişkenini resmin dosya adıyla eşleşecek şekilde değiştirin. Alternatif olarak, (Object_detection_video.py kullanarak) nesnelerin içinde bulunduğu videoda nesne tanıması yapabilir  veya (Object_detection_webcam.py kullanarak) USB web kamerası takarak tanınması istenen nesneleri tanıyabilir hale getirilebilir.

Komut dosyalarından herhangi birini çalıştırmak için, Anaconda Komut İstemi'ne (“tensorflow_kod” veya kendi sanal ortamınızı etkinleştirilerek) “idle” yazın ve ENTER'a basın. Bu, IDLE (compiler- derleyici veya kod düzenleme asistanı)'yi açacaktır ve oradan, komut dosyalarından herhangi birini açıp çalıştırabilirsiniz.

Her şey düzgün çalışıyorsa, nesne dedektörü yaklaşık 10 saniye başlatılır ve ardından görüntüde algıladığı nesneleri gösteren bir pencere görüntüler!