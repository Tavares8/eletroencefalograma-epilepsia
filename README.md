# eletroencefalograma-epilepsia
Classificação de sinais neurais no auxílio ao diagnóstico médico para epilepsia.
Toy Example Epilepsia (EPL): Rede Neural Artificial (ANN) do tipo Perceptron Multicamadas (PMC), Deep Learning (DL), e Máquina de Vetor de Suporte (SVM), com treinamento supervisionado, para a classificação do eletroencefalograma para auxílio no diagnóstico de epilepsia.

Motivação
No estudo [2] é desenvolvida uma estrutura de extração de informações do EEG com sistemas DL e computação em nuvem para resolver o problema de análise de dados na epilepsia, este método apresenta 94% de acurácia na análise de uma grande quantidade de dados não supervisionados. Em [3] é considerada uma CNN para a geração automática de características a partir de dados de EEG's epilépticos intracranianos no domínio do tempo, com acurácia média de 87,51% em um grupo de 25 indivíduos.
No estudo de detecção de epilepsia os sinais de EEG analisados neste trabalho são descritos no trabalho [1], onde os autores fazem um estudo comparativo entre as propriedades dinâmicas da atividade elétrica cerebral de diferentes regiões de gravação e de diferentes estados fisiológicos e patológicos do cérebro. Este conjunto de dados está disponível publicamente pela Universidade de Bonn, da Alemanha.
Os resultados de [4] mostram que a detecção de epilepsia, para este conjunto de dados, pode ser realizada com uma taxa de precisão de até 99,6 %, com uma única característica de entrada. São extraídas características no domínio do tempo e da frequência, e classificadas usando um tipo de RNA conhecida como Elman Network. Outro exemplo de classificação do conjunto de dados de epilepsia da Universidade de Bonn, está em [5], o sistema é proposto tem três etapas: extração de características usando o método de Welch (FFT), redução de dimensionalidade usando PCA e sistema de reconhecimento imunológico artificial, obtém 100\% de acurácia de classificação.

Banco de dados
No início deste trabalho já se dispunha de um banco de dados da Universidade de Bonn [1].
O conjunto de dados completo consiste em cinco conjuntos (denotados A-E) contendo cada um 100 séries de sinais EEG com duração de 23,6 segundos e frequência amostral de 173,61 Hz. Estes segmentos foram selecionados e segmentados a partir de gravações de sinais EEG de multi-canais após inspeção visual de artefatos, devido à atividade muscular ou movimentos oculares. Os conjuntos A e B consistem em segmentos extraídos de gravações de sinais EEG de superfície que foram realizados em cinco voluntários saudáveis com olhos abertos e fechados, respectivamente, com 19 eletrodos, seguindo o esquema padronizado de colocação 10\20: FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz. 
Os conjuntos C, D e E originam-se de sinais EEG de gravações pré-cirúrgicas. Foram selecionados sinais EEG de cinco pacientes, todos com controle completo de convulsões após ressecção de uma das formações do hipocampo, o que foi corretamente diagnosticado como sendo a zona epileptogênica. Os segmentos no conjunto D são registrados dentro da zona epileptogênica e os do conjunto C da formação hipocampal do hemisfério oposto do cérebro, durante períodos livres de crise, o conjunto E é registrado durante uma crise epiléptica.
Todos os sinais EEG são gravados com o sistema de amplificador de 128 canais, usando uma referência comum. Os dados foram digitalizados a 173,61 amostras por segundo usando resolução de 12 bits. Filtro passa-banda é de 0,53-40 Hz (12 dB/oct). 
Neste estudo, são utilizadas duas classes; conjunto de dados B e E, correspondendo às classes saudável e doente, respectivamente. 
As amplitudes das gravações superficiais, do conjunto B, são tipicamente da ordem de mV. Para as gravações intracanianas, do conjunto E, as amplitudes podem exceder 1000 mV. 
São apresentadas às entradas dos classificadores 200 exemplos de EEG, em duas configurações:
1.	Sinal EEG bruto: com duração de 23,6 segundos, 200 séries temporais de EEG's de 4097 amostras. Sendo 4097 neurônios para a camada de entrada do classificador.

2.	Extração prévia de características: é feita a extração de um conjunto de 72 características relevantes, energia e valor RMS, de cada série EEG em janelas de 4 segundos, do tipo hamming. Há uma redução na dimensionalidade dos dados de entrada do classificador. O número de neurônios da camada de entrada do classificador é igual à 72.

Resultados
Cada configuração (sinal bruto e características extraídas) é classificada por mais de um classificador e como critério de comparação são observadas as métricas: tempo de treinamento, acurácia e precisão na validação.
São realizadas 1000 épocas de treinamento e k-fold = 10, são mostrados os resultados médios.

Os resultados da classificação do conjunto de dados de epilepsia mostram que, havendo a extração prévia de características, configuração 2, o algoritmo PMC alcança 98 % de acurácia e 100 % de precisão na validação em 1,49 segundos de treinamento. Todos os classificadores têm bons resultados com extração de características.
Com o intuito de avaliar os algoritmos sem a extração prévia de características, levantou-se os resultados com a configuração 1, sinal bruto, o algoritmo DL-CNN se destaca e alcança 87 % de acurácia e 97 % de precisão na validação.

Conclusões Preliminares
Os resultados preliminares mostram que, com diferentes arquiteturas, o classificador DL pode classificar os sinais EEG do conjunto de dados de epilepsia com 97% de precisão na validação, sem extração prévia de características. Levando-se em conta que não há uma preocupação quanto ao estudo de características dos sinais EEG, o método DL confirma a hipótese inicial de que, com dados brutos a capacidade de generalização é melhor que outros métodos clássicos. 
Os classificadores se mostraram capazes de detectar os exemplos de pacientes com epilepsia podendo, portanto serem aplicados no auxílio de diagnóstico médico. 

Referências
[1]	Andrzejak, R. et al., 2001. Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state. Physical Review E, 64(6), p.061907.

[2]	 {Hosseini2017, author    = {M. P. Hosseini and H. Soltanian{-}Zadeh and
	K. V. Elisevich and D. Pompili},
title     = {Cloud-based Deep Learning of Big {EEG} Data for Epileptic Seizure Prediction},
	
[3]	{Antoniades2016, 
	author={A. Antoniades and L. Spyrou and C. C. Took and S. Sanei}, 
	booktitle={2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP)}, 
	title={Deep learning for epileptic intracranial EEG data}.
[4]	{Srinivasan2005,
	author={v. Srinivasan and C. Eswaran and N. Sriraam},
	title={Artificial Neural Network Based Epileptic Detection Using Time-Domain and Frequency-Domain Features},
	journal={Journal of Medical Systems},
	

[5]	{Polat2008,
title = {Artificial immune recognition system with fuzzy resource allocation mechanism classifier, principal component analysis and \{FFT\} method based new hybrid automated identification system for classification of \{EEG\} signals},
	journal = {Expert Systems with Applications},


	

	

