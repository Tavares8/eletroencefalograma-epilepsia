import scipy.io as sio
import numpy as np
import pickle as pck
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from time import process_time

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#Input
sinal_ = sio.loadmat('sinal')
sinal_ = sinal_.get('sinal', 0)
#cria uma matriz
sinal = np.zeros(shape = (sinal_.shape[0], 1, sinal_.shape[1]))
#preenche a matriz com os valores desejados
for k in range(0, sinal_.shape[0]):
    sinal[k, 0] = sinal_[k]
quantidade_amostras = sinal.shape[2]
quantidade_exemplos = sinal.shape[0]

#Label
label_ = sio.loadmat('label')
label_ = label_.get('label', 0)
#cria uma matriz
label = np.zeros(shape = (label_.shape[0], ))
#preenche a matriz com os valores desejados
for k in range(0, label_.shape[0]):
    label[k] = label_[k, 0] - 1
label = label.astype(np.uint8)

print(sinal.shape)
print(label.shape)

Nclass = 2
epocas = 100
k = 10

# definir arquitetura da PMC - com uma camada intermediária
mlp = MLPClassifier(hidden_layer_sizes=(256,), activation='relu', max_iter=epocas, alpha=1e-4,
                     solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01)

# definir arquitetura da SVM
svm = svm.SVC(kernel='rbf',C=1, gamma='auto')

# definir arquitetura da adaboost
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 5), algorithm="SAMME", n_estimators = 100)

# definir arquitetura da bagging
bagging = BaggingClassifier()

# definir arquitetura da gradboost
gradboost = GradientBoostingClassifier()

#com o k folds definidos, separa dados de validação e treinamento - repetir k vezes (folds)

time_train_mlp =[]
acuracia_mlp = []
precisao_mlp = []

time_train_svm =[]
acuracia_svm = []
precisao_svm = []

time_train_adaboost =[]
acuracia_adaboost = []
precisao_adaboost = []

time_train_bagging = []
acuracia_bagging = []
precisao_bagging = []

time_train_gradboost = []
acuracia_gradboost = []
precisao_gradboost = []

# Validação cruzada com k folds
skf = StratifiedKFold(n_splits=k, random_state=None)

for train_index, test_index in skf.split(sinal,label): 
    #print("Train:", train_index, "Validation:", test_index) 
    
    sinais_treinamento, sinais_validacao = sinal[train_index], sinal[test_index] 
    labels_treinamento, labels_validacao = label[train_index], label[test_index]
    
    x_treinamento, y_treinamento = shuffle(sinais_treinamento, labels_treinamento, random_state = 42)
    x_validacao, y_validacao = shuffle(sinais_validacao, labels_validacao, random_state = 42)
    
# Faz x_treinamento e x_validação adequados para a entrada dos clf's   
    n_exemplos_treinamento = x_treinamento.shape[0]
    x_treinamento_vetor = x_treinamento.reshape((n_exemplos_treinamento, -1))
    
    n_exemplos_validacao = x_validacao.shape[0]
    x_validacao_vetor = x_validacao.reshape((n_exemplos_validacao, -1))

# Aqui fazer um treinamento de x épocas e uma validacao da MLP
    start = process_time()
    mlp.fit(x_treinamento_vetor, y_treinamento)
    end = process_time()
    time_mlp = end - start
    print('Métricas do treinamento da MLP')
    print('Tempo de treinamento_mlp com ' + str (epocas) + ' épocas: ' + str(time_mlp))      
    print("Erro no final do treinamento: %f" % mlp.loss_)
      
# Métricas da validacão mlp
    preds_val_mlp = mlp.predict(x_validacao_vetor)  
    print ('Métricas de uma validação MLP')
    print("Acertos do conjunto de validação: %f" % mlp.score(x_validacao_vetor, y_validacao))
    cm_val_mlp = confusion_matrix(y_validacao, preds_val_mlp)
    print('Matriz de Confusão')
    print(cm_val_mlp)
    TP = cm_val_mlp[0,0]
    FP = cm_val_mlp[0,1]
    FN = cm_val_mlp[1,0]
    TN = cm_val_mlp[1,1]

    acuracia_mlp_ = (TP+TN)*100/(len(y_validacao))
    precisao_mlp_ = TP*100/(TP+FP)
    print('acurácia_mlp_:  '+ str(acuracia_mlp_))
    print('precisao_mlp_:  '+ str(precisao_mlp_))
    print('###################################################################')
# Usar no calculo das médias da mlp
    time_train_mlp.append(time_mlp)
    acuracia_mlp.append(acuracia_mlp_)
    precisao_mlp.append(precisao_mlp_)
    
# salva a rede treinada    
#   pck.dump(mlp, open("trained_mlp_fold_number_" + str(n) + ".pickle", "wb")) 
##################################################################################
# Aqui fazer um treinamento e validacao da svm   
    start = process_time()
    svm.fit(x_treinamento_vetor, y_treinamento)
    end = process_time()
    time_svm = end - start
    print('Tempo de treinamento_svm com ' + str (epocas) + ' épocas: ' + str(time_svm))
# Métricas da validacão svm
    preds_val_svm= svm.predict(x_validacao_vetor)
    correct_outputs_val_svm = y_validacao
    n_acertos_val_svm = 0
    for u in range(0, len(correct_outputs_val_svm)):
        if preds_val_svm[u] == correct_outputs_val_svm[u]:
            n_acertos_val_svm += 1
    print ('Métricas de uma validação SVM') 
    print('Number of acertos_val_svm: ' + str(n_acertos_val_svm))  
    cm_val_svm = confusion_matrix(y_validacao, preds_val_svm)
    print('Matriz de Confusão')
    print(cm_val_svm)
    TP = cm_val_svm[0,0]
    FP = cm_val_svm[0,1]
    FN = cm_val_svm[1,0]
    TN = cm_val_svm[1,1]
    acuracia_svm_ = (TP+TN)*100/(len(y_validacao))
    precisao_svm_ = TP*100/(TP+FP)
    print('acurácia_svm_:  '+ str(acuracia_svm_))
    print('precisao_svm_:  '+ str(precisao_svm_)) 
    print('###################################################################')
# Usar no calculo das médias da svm  
    time_train_svm.append(time_svm)
    acuracia_svm.append(acuracia_svm_)
    precisao_svm.append(precisao_svm_)
##################################################################################
#  Aqui fazer um treinamento e validacao da AdaBoost
    start = process_time()
    adaboost.fit(x_treinamento_vetor, y_treinamento)
    end = process_time()
    time_adaboost = end - start
    print('Tempo de treinamento_adaboost: ' + str (epocas) + str(time_adaboost))
# Métricas da validação da Adaboost  
    preds_val_adaboost= adaboost.predict(x_validacao_vetor)
    correct_outputs_val_adaboost = y_validacao
    n_acertos_val_adaboost = 0
    for u in range(0, len(correct_outputs_val_adaboost)):
        if preds_val_adaboost[u] == correct_outputs_val_adaboost[u]:
            n_acertos_val_adaboost += 1
    print ('Métricas de uma validação adaboost') 
    print('Number of acertos_val_adaboost: ' + str(n_acertos_val_adaboost))  
    cm_val_adaboost = confusion_matrix(y_validacao, preds_val_adaboost)
    print('Matriz de Confusão')
    print(cm_val_adaboost)
    TP = cm_val_adaboost[0,0]
    FP = cm_val_adaboost[0,1]
    FN = cm_val_adaboost[1,0]
    TN = cm_val_adaboost[1,1]
    acuracia_adaboost_ = (TP+TN)*100/(len(y_validacao))
    precisao_adaboost_ = TP*100/(TP+FP)
    print('acurácia_adaboost_:  '+ str(acuracia_adaboost_))
    print('precisao_adaboos_:  '+ str(precisao_adaboost_)) 
    print('###################################################################')
# Usar no calculo das médias da adaboost  
    time_train_adaboost.append(time_adaboost)
    acuracia_adaboost.append(acuracia_adaboost_)
    precisao_adaboost.append(precisao_adaboost_)     
##################################################################################    
# Aqui fazer um treinamento e validacao da BaggingClassifier
    start = process_time()
    bagging.fit(x_treinamento_vetor, y_treinamento)
    end = process_time()
    time_bagging = end - start
    print('Tempo de treinamento_bagging: ' + str (epocas) + str(time_bagging))
    
# Métricas da validação da Bagging 
    preds_val_bagging = bagging.predict(x_validacao_vetor)
    correct_outputs_val_bagging = y_validacao
    n_acertos_val_bagging = 0
    for u in range(0, len(correct_outputs_val_bagging)):
        if preds_val_bagging[u] == correct_outputs_val_bagging[u]:
            n_acertos_val_bagging += 1
    print ('Métricas de uma validação bagging') 
    print('Number of acertos_val_bagging: ' + str(n_acertos_val_bagging))  
    cm_val_bagging = confusion_matrix(y_validacao, preds_val_bagging)
    print('Matriz de Confusão')
    print(cm_val_bagging)
    TP = cm_val_bagging[0,0]
    FP = cm_val_bagging[0,1]
    FN = cm_val_bagging[1,0]
    TN = cm_val_bagging[1,1]
    acuracia_bagging_ = (TP+TN)*100/(len(y_validacao))
    precisao_bagging_ = TP*100/(TP+FP)
    print('acurácia_bagging_:  '+ str(acuracia_bagging_))
    print('precisao_bagging_:  '+ str(precisao_bagging_)) 
    print('###################################################################')
# Usar no calculo das médias da bagging  
    time_train_bagging.append(time_bagging)
    acuracia_bagging.append(acuracia_bagging_)
    precisao_bagging.append(precisao_bagging_)     
##################################################################################  
# Aqui fazer um treinamento e validacao da GradientBoosting
    start = process_time()
    gradboost.fit(x_treinamento_vetor, y_treinamento)
    end = process_time()
    time_gradboost = end - start
    print('Tempo de treinamento_gradboost: ' + str (epocas) + str(time_gradboost))
     
# Métricas da validação da Gradboost
    preds_val_gradboost = gradboost.predict(x_validacao_vetor)
    correct_outputs_val_gradboost = y_validacao
    n_acertos_val_gradboost = 0
    for u in range(0, len(correct_outputs_val_gradboost)):
        if preds_val_gradboost[u] == correct_outputs_val_gradboost[u]:
            n_acertos_val_gradboost += 1
    print ('Métricas de uma validação gradboost') 
    print('Number of acertos_val_gradboost: ' + str(n_acertos_val_gradboost))  
    cm_val_gradboost = confusion_matrix(y_validacao, preds_val_gradboost)
    print('Matriz de Confusão')
    print(cm_val_gradboost)
    TP = cm_val_gradboost[0,0]
    FP = cm_val_gradboost[0,1]
    FN = cm_val_gradboost[1,0]
    TN = cm_val_gradboost[1,1]
    acuracia_gradboost_ = (TP+TN)*100/(len(y_validacao))
    precisao_gradboost_ = TP*100/(TP+FP)
    print('acurácia_gradboost_:  '+ str(acuracia_gradboost_))
    print('precisao_gradboost_:  '+ str(precisao_gradboost_)) 
    print('###################################################################')
# Usar no calculo das médias da Gradboost  
    time_train_gradboost.append(time_gradboost)
    acuracia_gradboost.append(acuracia_gradboost_)
    precisao_gradboost.append(precisao_gradboost_)  
   
media_time_train_mlp = sum(time_train_mlp) / float(len(time_train_mlp))
media_acuracia_mlp = sum(acuracia_mlp) / float(len(acuracia_mlp))
media_precisao_mlp = sum(precisao_mlp) / float(len(precisao_mlp))

media_time_train_svm = sum(time_train_svm) / float(len(time_train_svm))
media_acuracia_svm = sum(acuracia_svm) / float(len(acuracia_svm))
media_precisao_svm = sum(precisao_svm) / float(len(precisao_svm))

media_time_train_adaboost = sum(time_train_adaboost) / float(len(time_train_adaboost))
media_acuracia_adaboost = sum(acuracia_adaboost) / float(len(acuracia_adaboost))
media_precisao_adaboost = sum(precisao_adaboost) / float(len(precisao_adaboost))

media_time_train_bagging = sum(time_train_bagging) / float(len(time_train_bagging))
media_acuracia_bagging = sum(acuracia_bagging) / float(len(acuracia_bagging))
media_precisao_bagging = sum(precisao_bagging) / float(len(precisao_bagging))

media_time_train_gradboost = sum(time_train_gradboost) / float(len(time_train_gradboost))
media_acuracia_gradboost = sum(acuracia_gradboost) / float(len(acuracia_gradboost))
media_precisao_gradboost = sum(precisao_gradboost) / float(len(precisao_gradboost))

print('Tempo médio de treinamento MLP com ' + str(k) + ' kfold ' + str (media_time_train_mlp))
print('Tempo médio de treinamento SVM com ' + str(k) + ' kfold ' + str (media_time_train_svm))
print('Tempo médio de treinamento ADABOOST com ' + str(k) + ' kfold ' + str (media_time_train_adaboost))
print('Tempo médio de treinamento BAGGING com ' + str(k) + ' kfold ' + str (media_time_train_bagging))
print('Tempo médio de treinamento GRADBOOST com ' + str(k) + ' kfold ' + str (media_time_train_gradboost))

print('Médias das Validações com ' + str(k) + ' folds')
print('Acurácia_mlp: ' + str(media_acuracia_mlp))
print('Precisão_mlp: ' + str(media_precisao_mlp))

print('Acurácia_svm: ' + str(media_acuracia_svm))
print('Precisão_svm: ' + str(media_precisao_svm))

print('Acurácia_adaboost: ' + str(media_acuracia_adaboost))
print('Precisão_adaboost: ' + str(media_precisao_adaboost))

print('Acurácia_bagging: ' + str(media_acuracia_bagging))
print('Precisão_bagging: ' + str(media_precisao_bagging))

print('Acurácia_gradboost: ' + str(media_acuracia_gradboost))
print('Precisão_gradboost: ' + str(media_precisao_gradboost))

