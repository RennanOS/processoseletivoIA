Projeto 3 - Detecção de Máscaras Faciais

Relatório do Candidato

Nome Completo: Rennan Oliveira Santos


### 1️⃣ Resumo da Abordagem

Fine-tuning do YOLO11n para detecção de máscaras em 3 classes: `with_mask`, `without_mask` e `mask_weared_incorrect`.

Hiperparâmetros:<br> 
Épocas: 20<br> 
Tamanho da imagem: 400x267<br> 
Batch size: 8<br> 
Device: CPU<br> 
Workers: 4<br> 
Patience: 10 (early stopping)

Ajuste para desbalanceamento: Nenhum ajuste específico foi aplicado, seguindo a recomendação do projeto para observar o comportamento natural do modelo diante da classe minoritária.


### 2️⃣ Bibliotecas Utilizadas

Biblioteca	                   ||Finalidade<br> 
ultralytics 8.4	               ||Fine-tuning, exportação e inferência YOLO<br> 
Python 3.11	                     ||Linguagem base<br> 
opencv-python	4.5.0              ||Processamento de imagens<br> 
tensorflow-cpu 2.13.0	           ||Exportação TFLite<br> 
onnx / onnx2tf 1.16.0	           ||Conversão ONNX → TFLite



### 3️⃣ Técnica de Otimização do Modelo

O modelo foi exportado para TensorFlow Lite (formato para Edge AI) utilizando **Google Colab**, pois a exportação TFLite não é suportada nativamente no Windows pela Ultralytics.

Parâmetros de Exportação:
Formato: TFLite (float32)
Tamanho da imagem: 400x267
Quantização: Não aplicada


### 4️⃣ Resultados Obtidos

Métricas de Validação (mAP):

Classe	                 ||              mAP50

Geral	                                 0.750

with_mask     	                       0.965

without_mask	                         0.792

mask_weared_incorrect	                 0.494



Tamanho dos Arquivos:

Arquivo	        ||         Tamanho

model.pt	               5.30 MB

model.tflite	           10.1 MB


Analisando os resultados, “with_mask” apresentou um ótimo desempenho (0.965), “without_mask” apresentou um bom desempenho (0.792), já “mask_weared_incorrect”, como esperado devido ao desbalanceamento do dataset, apresentou um desempenho muito inferior (0.494).



### 5️⃣ Comentários Adicionais

A principal dificuldade encontrada durante o desenvolvimento do projeto foi a exportação do modelo para TensorFlow Lite em ambiente Windows. A biblioteca Ultralytics não suporta nativamente a exportação TFLite no Windows, exibindo a mensagem de erro "LiteRT export only supported on Linux x86 and macOS". Para contornar essa limitação, utilizou-se o Google Colab, que oferece ambiente Linux com suporte completo à exportação, permitindo a geração do arquivo model.tflite que foi então inserido na raiz do projeto.
O projeto proporcionou um aprendizado prático significativo sobre o fluxo completo de um projeto de Visão Computacional Embarcada. Primeiramente, houve um aprimoramento da experiência com o Google Colab como ferramenta de desenvolvimento, especialmente para contornar limitações de ambiente. Além disso, foi possível compreender na prática todo o pipeline: desde o fine-tuning de um modelo YOLO pré-treinado, passando pela validação e análise de métricas, até a exportação para um formato otimizado para dispositivos edge (TFLite) e a execução de inferência em imagens reais.
A principal limitação observada no modelo foi o desempenho inferior na classe mask_weared_incorrect, que obteve mAP50 de apenas 0.494, enquanto as demais classes apresentaram resultados significativamente superiores (0.965 para with_mask e 0.792 para without_mask). Esse comportamento era esperado e está diretamente relacionado ao desbalanceamento do dataset, que contém consideravelmente menos exemplos da classe minoritária. Em um cenário de produção, essa limitação poderia ser mitigada com a coleta de mais dados da classe, aplicação de técnicas de aumento de dados direcionadas ou uso de pesos para classes durante o treinamento.


### 6️⃣ Exemplo de Inferência

  Saída do Terminal

maksssksksss105.jpg            9  [9x with_mask]

maksssksksss107.jpg            1  [1x with_mask]

maksssksksss11.jpg             25  [2x mask_weared_incorrect, 23x with_mask]

maksssksksss113.jpg            5  [3x with_mask, 2x without_mask]

maksssksksss12.jpg             14  [12x with_mask, 2x without_mask]

TOTAL                          54



Comentários:

Abrindo as imagens anotadas na pasta (runs/detect/inferencia_exemplos/predicoes/), observou-se que as bounding boxes estão bem posicionadas sobre os rostos identificados, relativamente com precisão espacial. Não houve confusão aparente entre as classes, e o modelo classificou corretamente os rostos com máscara, sem máscara e com máscara incorreta. Quando a classe minoritária (mask_weared_incorrect) foi detectada, ela também foi corretamente identificada. Um ponto de atenção observado foi que as caixas de texto (labels) das detecções apresentaram-se em tamanho grande e, em alguns casos, sobrepondo partes das imagens, dificultando um pouco a identificação visual dos rostos. No entanto, trata-se de um comportamento padrão da visualização do Ultralytics e não compromete a qualidade da detecção, sendo apenas um aspecto estético da renderização. 


### Referências:

FIT - Flextronics Instituto de Tecnologia. Fundamentos de Inteligência Artificial para Sistemas Embarcados. PNAAT - Programa Nacional de Aprendizado Acelerado em Tecnologia. Disponível em: https://fit-tecnologia.org.br/pnaat/. Acesso em: 22 jul. 2026.

FIT - Flextronics Instituto de Tecnologia. Sistemas de Visão Computacional Embarcada para Automação e Controle de Qualidade. PNAAT - Programa Nacional de Aprendizado Acelerado em Tecnologia. Disponível em: https://fit-tecnologia.org.br/pnaat/. Acesso em: 22 jul. 2026.

FIT - Flextronics Instituto de Tecnologia. Otimização de Modelos em Sistemas Embarcados. PNAAT - Programa Nacional de Aprendizado Acelerado em Tecnologia. Disponível em: https://fit-tecnologia.org.br/pnaat/. Acesso em: 22 jul. 2026.
