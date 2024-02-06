from flask import Flask, request, abort, jsonify
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging
import sys
from loguru import logger
import json
#import torch


logging.basicConfig(level=logging.INFO)
logger.add(sys.stderr)
logger.add("logs/floor_logs.log", rotation="10 MB")

def normalization(x):
    return (x - -100)/(0 - -100)
"""
class Encoder(torch.nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim, window):
        super().__init__()

        self.encoder_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, kernel_size=(3, 2), stride=(1, 2), padding=(1, 0)),
            torch.nn.ReLU(True)
        )

        self.flatten = torch.nn.Flatten(start_dim=1)

        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(3 * window * 32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class MultiOutputRegressionHead(torch.nn.Module):
    def __init__(self, encoded_space_dim=9):
        super(MultiOutputRegressionHead, self).__init__()
        self.hid1 = torch.nn.Linear(encoded_space_dim, 32) # nb in, nb out ?
        self.hid2 = torch.nn.Linear(32, 32)
        self.outp = torch.nn.Linear(32, 2)

        torch.nn.init.xavier_uniform_(self.hid1.weight)
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight)
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.outp.weight)
        torch.nn.init.zeros_(self.outp.bias)

    def forward(self, x, nb_layers=1):
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = self.outp(z)

        return z
"""
# Modèles fondation

model_elevator_fonda_v_oct_3eme_porte_ouverte_fond = joblib.load('model/fonda/ascenseur_3eme_porte_ouverte/knn_model_elevator_detection_au_fond.joblib')
model_elevator_fonda_v_oct_all_floors_fond = joblib.load('model/fonda/ascenseur_tous_etages/knn_model_elevator_detection_au_fond.joblib')
model_elevator_fonda_v_oct_all_floors_fond_agreg = joblib.load('model/fonda/ascenseur_tous_etages/knn_model_elevator_detection_au_fond_agreg.joblib')
model_elevator_fonda_v_oct_3eme_porte_ouverte_agreg = joblib.load('model/fonda/ascenseur_3eme_porte_ouverte/knn_model_elevator_detection_pas_que_au_fond_3eme_agreg.joblib')
model_elevator_fonda_v_oct_all_floors_pas_que_fond_agreg = joblib.load('model/fonda/ascenseur_tous_etages/knn_model_elevator_detection_pas_que_au_fond_agreg.joblib')
model_elevator_fonda_v_oct_all_floors_all_cluster_data_agreg = joblib.load('model/fonda/ascenseur_tous_etages/knn_model_elevator_detection_all_cluster_data.joblib')

model_floor_fonda_v_mars = joblib.load('model/fonda/new_models/knn_model_floor.joblib')
model_elevator_fonda_v_mars = joblib.load('model/fonda/new_models/knn_model_elevator_detection.joblib')
#model_elevator_fonda_v_mars = joblib.load('model/fonda/new_models/knn_model_elevator_detection_rp_in_elevator.joblib')

model_cluster_RDC_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor0_cluster_model.joblib')
model_cluster_Etage1_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor1_cluster_model.joblib')
model_cluster_Etage2_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor2_cluster_model.joblib')
model_cluster_Etage3_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor3_cluster_model.joblib')
model_cluster_Etage4_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor4_cluster_model.joblib')
model_cluster_Etage5_fonda_v_mars = joblib.load('model/fonda/new_models/knn_floor5_cluster_model.joblib')

knn_c00_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor00_pos_X_model.joblib")
knn_c00_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor00_pos_Y_model.joblib")
knn_c01_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor01_pos_X_model.joblib")
knn_c01_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor01_pos_Y_model.joblib")
knn_c02_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor02_pos_X_model.joblib")
knn_c02_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor02_pos_Y_model.joblib")
knn_c03_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor03_pos_X_model.joblib")
knn_c03_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor03_pos_Y_model.joblib")

knn_c10_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor10_pos_X_model.joblib")
knn_c10_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor10_pos_Y_model.joblib")
knn_c11_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor11_pos_X_model.joblib")
knn_c11_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor11_pos_Y_model.joblib")
knn_c12_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor12_pos_X_model.joblib")
knn_c12_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor12_pos_Y_model.joblib")

knn_c20_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor20_pos_X_model.joblib")
knn_c20_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor20_pos_Y_model.joblib")
knn_c21_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor21_pos_X_model.joblib")
knn_c21_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor21_pos_Y_model.joblib")
knn_c22_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor22_pos_X_model.joblib")
knn_c22_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor22_pos_Y_model.joblib")

knn_c30_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor3_pos_X_model.joblib")
knn_c30_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor3_pos_Y_model.joblib")

knn_c30_x_model_fonda_v_mars = joblib.load("model/fonda/08_2022_models/knn_floor3_pos_X_model.joblib")
knn_c30_y_model_fonda_v_mars = joblib.load("model/fonda/08_2022_models/knn_floor3_pos_Y_model.joblib")

knn_c40_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor40_pos_X_model.joblib")
knn_c40_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor40_pos_Y_model.joblib")
knn_c41_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor41_pos_X_model.joblib")
knn_c41_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor41_pos_Y_model.joblib")

knn_c50_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor50_pos_X_model.joblib")
knn_c50_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor50_pos_Y_model.joblib")

knn_c50_x_model_fonda_v_mars = joblib.load("model/fonda/08_2022_models/knn_floor5_pos_X_model.joblib")
knn_c50_y_model_fonda_v_mars = joblib.load("model/fonda/08_2022_models/knn_floor5_pos_Y_model.joblib")

knn_c51_x_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor51_pos_X_model.joblib")
knn_c51_y_model_fonda_v_mars = joblib.load("model/fonda/new_models/knn_floor51_pos_Y_model.joblib")

model_floor_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_model_floor.joblib')
model_elevator_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_model_elevator_detection.joblib')


model_cluster_RDC_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor0_cluster_model.joblib')
model_cluster_Etage1_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor1_cluster_model.joblib')
model_cluster_Etage2_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor2_cluster_model.joblib')
model_cluster_Etage3_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor3_cluster_model.joblib')
model_cluster_Etage4_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor4_cluster_model.joblib')
model_cluster_Etage5_fonda_v_sept = joblib.load('model/fonda/sept_2022_v0_models/knn_floor5_cluster_model.joblib')
knn_c00_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor00_pos_X_model.joblib")
knn_c00_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor00_pos_Y_model.joblib")
knn_c01_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor01_pos_X_model.joblib")
knn_c01_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor01_pos_Y_model.joblib")
knn_c02_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor02_pos_X_model.joblib")
knn_c02_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor02_pos_Y_model.joblib")
knn_c03_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor03_pos_X_model.joblib")
knn_c03_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor03_pos_Y_model.joblib")
knn_c10_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor10_pos_X_model.joblib")
knn_c10_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor10_pos_Y_model.joblib")
knn_c11_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor11_pos_X_model.joblib")
knn_c11_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor11_pos_Y_model.joblib")
knn_c12_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor12_pos_X_model.joblib")
knn_c12_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor12_pos_Y_model.joblib")
knn_c20_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor20_pos_X_model.joblib")
knn_c20_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor20_pos_Y_model.joblib")
knn_c21_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor21_pos_X_model.joblib")
knn_c21_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor21_pos_Y_model.joblib")
knn_c22_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor22_pos_X_model.joblib")
knn_c22_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor22_pos_Y_model.joblib")

knn_c30_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor30_pos_X_model.joblib")
knn_c30_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor30_pos_Y_model.joblib")

knn_c31_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor31_pos_X_model.joblib")
knn_c31_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor31_pos_Y_model.joblib")

knn_c40_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor40_pos_X_model.joblib")
knn_c40_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor40_pos_Y_model.joblib")
knn_c41_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor41_pos_X_model.joblib")
knn_c41_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor41_pos_Y_model.joblib")
knn_c50_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor50_pos_X_model.joblib")
knn_c50_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor50_pos_Y_model.joblib")

knn_c51_x_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor51_pos_X_model.joblib")
knn_c51_y_model_fonda_v_sept = joblib.load("model/fonda/sept_2022_v0_models/knn_floor51_pos_Y_model.joblib")


# Modeles reseaux de neurones et tests agregation sur plusieurs secondes

encoder_weights_fonda = "model/fonda/ameliorations_reseaux_neurones_et_agregation/auto_enc/encoder.pt"
ff_encoder_head_weights_fonda = "model/fonda/ameliorations_reseaux_neurones_et_agregation/auto_enc/ae_head_regressor.pt"

knn_x_y_3sec_avg_regressor_fonda = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/KNN_agregated_regression/knn_x_y_3sec_avg_regressor.joblib")
knn_floor_3sec_avg_classifier_fonda = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/KNN_agregated_regression/knn_floor_3sec_avg_classifier.joblib")

model_floor_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_model_floor.joblib')
#model_elevator_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_model_elevator_detection.joblib')


model_cluster_RDC_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor0_cluster_model.joblib')
model_cluster_Etage1_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor1_cluster_model.joblib')
model_cluster_Etage2_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor2_cluster_model.joblib')
model_cluster_Etage3_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor3_cluster_model.joblib')
model_cluster_Etage4_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor4_cluster_model.joblib')
model_cluster_Etage5_fonda_v_avg = joblib.load('model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor5_cluster_model.joblib')
knn_c00_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor00_pos_X_model.joblib")
knn_c00_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor00_pos_Y_model.joblib")
knn_c01_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor01_pos_X_model.joblib")
knn_c01_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor01_pos_Y_model.joblib")
knn_c02_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor02_pos_X_model.joblib")
knn_c02_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor02_pos_Y_model.joblib")
knn_c03_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor03_pos_X_model.joblib")
knn_c03_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor03_pos_Y_model.joblib")
knn_c10_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor10_pos_X_model.joblib")
knn_c10_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor10_pos_Y_model.joblib")
knn_c11_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor11_pos_X_model.joblib")
knn_c11_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor11_pos_Y_model.joblib")
knn_c12_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor12_pos_X_model.joblib")
knn_c12_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor12_pos_Y_model.joblib")
knn_c20_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor20_pos_X_model.joblib")
knn_c20_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor20_pos_Y_model.joblib")
knn_c21_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor21_pos_X_model.joblib")
knn_c21_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor21_pos_Y_model.joblib")
knn_c22_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor22_pos_X_model.joblib")
knn_c22_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor22_pos_Y_model.joblib")

knn_c30_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor30_pos_X_model.joblib")
knn_c30_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor30_pos_Y_model.joblib")

knn_c31_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor31_pos_X_model.joblib")
knn_c31_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor31_pos_Y_model.joblib")

knn_c40_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor40_pos_X_model.joblib")
knn_c40_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor40_pos_Y_model.joblib")
knn_c41_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor41_pos_X_model.joblib")
knn_c41_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor41_pos_Y_model.joblib")
knn_c50_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor50_pos_X_model.joblib")
knn_c50_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor50_pos_Y_model.joblib")

knn_c51_x_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor51_pos_X_model.joblib")
knn_c51_y_model_fonda_v_avg = joblib.load("model/fonda/ameliorations_reseaux_neurones_et_agregation/agregated_train_base/knn_floor51_pos_Y_model.joblib")


# modeles tests streetgo
model_floor_fonda_streetgo = joblib.load('model/fonda/streetgo/knn_model_floor_3.joblib')
model_elevator_fonda_streetgo = joblib.load('model/fonda/streetgo/knn_model_elevator.joblib')


# Modèles Pépinière

model_cluster_pep = joblib.load('model/pep/model_cluster_pep2.joblib')
model_X_pep = joblib.load('model/pep/model_X_pep3.joblib')
model_Y_pep = joblib.load('model/pep/model_Y_pep3.joblib')

# Modèles Maison Victor Hugo

knn_floor_model_vic = joblib.load("model/victor_hugo/knn_floor_model_only_cluster_30_03.joblib")
knn_floor1_cluster_model_vic = joblib.load("model/victor_hugo/knn_floor1_cluster_model.joblib")
knn_floor2_cluster_model_vic = joblib.load("model/victor_hugo/knn_floor2_cluster_model.joblib")

knn_c20_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC20_position_model.joblib")
knn_c21_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC21_position_model.joblib")
knn_c22_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC22_position_model.joblib")
knn_c23_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC23_position_model.joblib")
knn_c24_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC24_position_model.joblib")
knn_c25_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC25_position_model.joblib")
knn_c26_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterC26_position_model.joblib")
knn_e12e_position_model_vic = joblib.load("model/victor_hugo/knn_floor2_clusterE12E_position_model.joblib")

# Modeles Metro Bellecour

knn_floor_model = joblib.load("model/bellecour/knn_floor_model.joblib")
knn_floor2_cluster_model = joblib.load("model/bellecour/knn_floor2_cluster_model.joblib")
knn_floor3_cluster_model = joblib.load("model/bellecour/knn_floor3_cluster_model.joblib")

knn_c20_position_model = joblib.load("model/bellecour/knn_floor2_clusterC20_position_model.joblib")
knn_c22_position_model = joblib.load("model/bellecour/knn_floor2_clusterC22_position_model.joblib")
knn_c23_position_model = joblib.load("model/bellecour/knn_floor2_clusterC23_position_model.joblib")
knn_c2b_position_model = joblib.load("model/bellecour/knn_floor2_clusterC2B_position_model.joblib")
knn_c2c_position_model = joblib.load("model/bellecour/knn_floor2_clusterC2C_position_model.joblib")
knn_c2d_position_model = joblib.load("model/bellecour/knn_floor2_clusterC2D_position_model.joblib")
knn_c2e_position_model = joblib.load("model/bellecour/knn_floor2_clusterC2E_position_model.joblib")
knn_c2f_position_model = joblib.load("model/bellecour/knn_floor2_clusterC2F_position_model.joblib")
knn_f2_ce1_position_model = joblib.load("model/bellecour/knn_floor2_clusterCE1_position_model.joblib")
knn_f2_ce2_position_model = joblib.load("model/bellecour/knn_floor2_clusterCE2_position_model.joblib")
knn_f2_ce4_position_model = joblib.load("model/bellecour/knn_floor2_clusterCE4_position_model.joblib")
knn_f2_ce5_position_model = joblib.load("model/bellecour/knn_floor2_clusterCE5_position_model.joblib")
knn_f2_cs2_position_model = joblib.load("model/bellecour/knn_floor2_clusterCS2_position_model.joblib")
knn_f2_cs3_position_model = joblib.load("model/bellecour/knn_floor2_clusterCS3_position_model.joblib")
knn_f2_cs4_position_model = joblib.load("model/bellecour/knn_floor2_clusterCS4_position_model.joblib")
knn_f2_cs6_position_model = joblib.load("model/bellecour/knn_floor2_clusterCS6_position_model.joblib")

knn_c30_position_model = joblib.load("model/bellecour/knn_floor3_clusterC30_position_model.joblib")
knn_c31_position_model = joblib.load("model/bellecour/knn_floor3_clusterC31_position_model.joblib")
knn_c32_position_model = joblib.load("model/bellecour/knn_floor3_clusterC32_position_model.joblib")
knn_c34_position_model = joblib.load("model/bellecour/knn_floor3_clusterC34_position_model.joblib")
knn_c35_position_model = joblib.load("model/bellecour/knn_floor3_clusterC35_position_model.joblib")
knn_f3_ce1_position_model = joblib.load("model/bellecour/knn_floor3_clusterCE1_position_model.joblib")
knn_f3_ce2_position_model = joblib.load("model/bellecour/knn_floor3_clusterCE2_position_model.joblib")
knn_f3_ce4_position_model = joblib.load("model/bellecour/knn_floor3_clusterCE4_position_model.joblib")
knn_f3_ce5_position_model = joblib.load("model/bellecour/knn_floor3_clusterCE5_position_model.joblib")
knn_f3_cs2_position_model = joblib.load("model/bellecour/knn_floor3_clusterCS2_position_model.joblib")
knn_f3_cs3_position_model = joblib.load("model/bellecour/knn_floor3_clusterCS3_position_model.joblib")
knn_f3_cs4_position_model = joblib.load("model/bellecour/knn_floor3_clusterCS4_position_model.joblib")
knn_f3_cs6_position_model = joblib.load("model/bellecour/knn_floor3_clusterCS6_position_model.joblib")

# modeles gare de Lyon

knn_floor_model_gdl = joblib.load("model/gare_de_lyon/knn_floor_model.joblib")
knn_cluster_model_gdl = joblib.load("model/gare_de_lyon/knn_floor1_cluster_model.joblib")

knn_c10_position_model_gdl = joblib.load("model/gare_de_lyon/knn_floor1_clusterC10_position_model.joblib")
knn_c11_position_model_gdl = joblib.load("model/gare_de_lyon/knn_floor1_clusterC11_position_model.joblib")
knn_c12_position_model_gdl = joblib.load("model/gare_de_lyon/knn_floor1_clusterC12_position_model.joblib")

# Modèles outils

model_state = joblib.load('model/outils/model_state.joblib')
model_stair = joblib.load('model/outils/model_stair_.joblib')

# Modèles state fonda
model_state_fonda = joblib.load('model/state/model_state_fonda.joblib')

app = Flask(__name__)


@app.route('/')
def index():
    return jsonify({'version': '1.1'})


@app.route('/fondation_streetgo', methods=['POST'])
def fondation_streetgo():
    content = request.json

    if content is None:
        logger.debug(f"ERROR : Missing data, received:\n{json.dumps(content)}", level="ERROR")
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        logger.debug(f"ERROR : Missing tab data, received:\n{json.dumps(content)}", level="ERROR")
        return jsonify({'error': 'missing tab data'})

    tab = content["tab"]
    if len(tab) != 25:
        logger.debug(f"ERROR : Not enough data. 25 values needed, got {len(tab)}}", level="ERROR")
        return jsonify({'error': '25 values needed'})

    tab_100 = np.array(tab)
    tab_90 = np.array(tab)

    tab_100[tab_100 == 0] = -100
    tab_90[tab_90 == -100] = -90

    #in_elevator = model_elevator_fonda_v_mars.predict(tab_90.reshape(1, 25))
    used_beacons_idx = [ 3, 5, 6, 7, 23, 24 ]
    in_elevator = model_elevator_fonda_streetgo.predict(tab_100[used_beacons_idx].reshape(1, -1))

    used_beacons_idx = [0, 3, 4, 5, 6, 9, 13, 23, 24]
    floor = model_floor_fonda_streetgo.predict(tab_100[used_beacons_idx].reshape(1, -1)) #model floor with elevator beacon
    
    response = {
        "floor": int(floor),
        "cluster": int(0),
        "pos_x": int(0),
        "pos_y": int(0),
        "elevator": False
    }
    
    logger.debug(f"Success : detected {floor} using {tab_100}", level="SUCCESS")
    logger.debug(f"Success : Complete response : {json.dumps(response)}", level="SUCCESS")
                 
    return response

@app.route('/fondation', methods=['POST'])
def fondation():
    content = request.json

    if content is None:
        logger.debug(f"ERROR : Missing data, received:\n{json.dumps(content)}", level="ERROR")
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        logger.debug(f"ERROR : Missing tab data, received:\n{json.dumps(content)}", level="ERROR")
        return jsonify({'error': 'missing tab data'})

    tab = content["tab"]
    if len(tab) != 25:
        logger.debug(f"ERROR : Not enough data. 25 values needed, got {len(tab)}}", level="ERROR")
        return jsonify({'error': '25 values needed'})

    tab_100 = np.array([tab])
    tab_90 = np.array([tab])

    tab_100[tab_100 == 0] = -100
    tab_90[tab_90 == -100] = -90

    #in_elevator = model_elevator_fonda_v_mars.predict(tab_90.reshape(1, 25))
    in_elevator = model_elevator_fonda_v_oct_all_floors_all_cluster_data_agreg.predict(tab_100.reshape(1, 25))
    if in_elevator:
        close_elevator = model_elevator_fonda_v_sept.predict(tab_100.reshape(1, 25))
        if close_elevator:
            return {
                "floor": 100,
                "cluster": 100,
                "pos_x": -70,
                "pos_y": 40,
                "elevator": True
            }
        else:
            models_dict = {
                (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
                (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
                (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
                (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
                (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
                (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
                (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
                (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
                (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
                (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
                (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
                (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
                (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
                (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
                (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
                (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
            }

            floor = model_floor_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1, 24)) #model floor with elevator beacon

            if floor == 0 :
                cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 1:
                cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 2:
                cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 3:
                cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 4:
                cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 5:
                cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]

            pos_X = models_dict[(int(floor), int(cluster))][0].predict(np.delete(tab_100, 7).reshape(1,24))[0]
            pos_Y = models_dict[(int(floor), int(cluster))][1].predict(np.delete(tab_100, 7).reshape(1,24))[0]

            return {
                "floor": int(floor),
                "cluster": int(cluster),
                "pos_x": int(pos_X),
                "pos_y": int(pos_Y),
                "elevator": True
            }
    else:
        models_dict = {
            (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
            (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
            (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
            (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
            (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
            (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
            (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
            (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
            (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
            (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
            (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
            (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
            (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
            (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
            (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
            (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
        }

        floor = model_floor_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1, 24)) #model floor with elevator beacon

        if floor == 0 :
            cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 1:
            cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 2:
            cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 3:
            cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 4:
            cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 5:
            cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]

        pos_X = models_dict[(int(floor), int(cluster))][0].predict(np.delete(tab_100, 7).reshape(1,24))[0]
        pos_Y = models_dict[(int(floor), int(cluster))][1].predict(np.delete(tab_100, 7).reshape(1,24))[0]

        return {
            "floor": int(floor),
            "cluster": int(cluster),
            "pos_x": int(pos_X),
            "pos_y": int(pos_Y),
            "elevator": False
        }

@app.route('/fondation_v0_avg', methods=['POST'])
def fondation_v0_avg():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = np.array(content["tab"])

    if tab.shape != (3, 25):
        return jsonify({'error': '3 rows of 25 values. Given shape : {}'.format(tab.shape)})

    tab_100 = tab
    tab_100[tab_100 == 0] = -100

    tab_100 = np.mean(tab_100, axis=0)

    #in_elevator = model_elevator_fonda_v_mars.predict(tab_90.reshape(1, 25))
    in_elevator = model_elevator_fonda_v_oct_all_floors_all_cluster_data_agreg.predict(tab_100.reshape(1, 25))
    if in_elevator:
        close_elevator = model_elevator_fonda_v_sept.predict(tab_100.reshape(1, 25))
        if close_elevator:
            return {
                "floor": 100,
                "cluster": 100,
                "pos_x": -70,
                "pos_y": 40,
                "elevator": True
            }
        else:
            models_dict = {
                (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
                (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
                (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
                (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
                (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
                (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
                (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
                (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
                (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
                (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
                (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
                (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
                (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
                (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
                (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
                (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
            }

            floor = model_floor_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1, 24)) #model floor with elevator beacon

            if floor == 0 :
                cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 1:
                cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 2:
                cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 3:
                cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 4:
                cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
            elif floor == 5:
                cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]

            pos_X = models_dict[(int(floor), int(cluster))][0].predict(np.delete(tab_100, 7).reshape(1,24))[0]
            pos_Y = models_dict[(int(floor), int(cluster))][1].predict(np.delete(tab_100, 7).reshape(1,24))[0]

            return {
                "floor": int(floor),
                "cluster": int(cluster),
                "pos_x": int(pos_X),
                "pos_y": int(pos_Y),
                "elevator": True
            }
    else:
        models_dict = {
            (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
            (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
            (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
            (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
            (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
            (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
            (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
            (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
            (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
            (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
            (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
            (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
            (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
            (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
            (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
            (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
        }

        floor = model_floor_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1, 24)) #model floor with elevator beacon

        if floor == 0 :
            cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 1:
            cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 2:
            cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 3:
            cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 4:
            cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]
        elif floor == 5:
            cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100, 7).reshape(1,24))[0]

        print(floor[0], cluster)

        pos_X = models_dict[(int(floor[0]), int(cluster))][0].predict(np.delete(tab_100, 7).reshape(1,24))[0]
        pos_Y = models_dict[(int(floor[0]), int(cluster))][1].predict(np.delete(tab_100, 7).reshape(1,24))[0]

        return {
            "floor": int(floor),
            "cluster": int(cluster),
            "pos_x": int(pos_X),
            "pos_y": int(pos_Y),
            "elevator": False
        }

@app.route('/fondation_knn_regression', methods=['POST'])
def fondation_knn_regression():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = np.array(content["tab"])

    if tab.shape != (3, 25):
        return jsonify({'error': '3 rows of 25 values. Given shape : {}'.format(tab.shape)})

    tab_100 = tab
    tab_100[tab_100 == 0] = -100

    #in_elevator = model_elevator_fonda_v_mars.predict(tab_90.reshape(1, 25))
    in_elevator = model_elevator_fonda_v_oct_all_floors_all_cluster_data_agreg.predict(tab_100[-1].reshape(1, 25))
    if in_elevator:
        close_elevator = model_elevator_fonda_v_sept.predict(tab_100[-1].reshape(1, 25))
        if close_elevator:
            return {
                "floor": 100,
                "cluster": 100,
                "pos_x": -70,
                "pos_y": 40,
                "elevator": True
            }
        else:

            knn_x_y_3sec_avg_regressor_fonda


            floor = model_floor_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1, 24)) #model floor with elevator beacon

            if floor == 0 :
                cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 1:
                cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 2:
                cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 3:
                cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 4:
                cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 5:
                cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]

            avg_tab = np.mean(tab_100, axis=0)
            pred_pos = knn_x_y_3sec_avg_regressor_fonda.predict(np.delete(avg_tab, 7).reshape(1, 24))[0]

            pos_X = pred_pos[0]
            pos_Y = pred_pos[1]


            return {
                "floor": int(floor),
                "cluster": int(cluster),
                "pos_x": int(pos_X),
                "pos_y": int(pos_Y),
                "elevator": True
            }
    else:
        models_dict = {
            (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
            (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
            (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
            (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
            (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
            (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
            (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
            (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
            (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
            (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
            (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
            (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
            (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
            (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
            (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
            (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
        }

        floor = model_floor_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1, 24)) #model floor with elevator beacon

        if floor == 0 :
            cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 1:
            cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 2:
            cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 3:
            cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 4:
            cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 5:
            cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]

        avg_tab = np.mean(tab_100, axis=0)
        pred_pos = knn_x_y_3sec_avg_regressor_fonda.predict(np.delete(avg_tab, 7).reshape(1, 24))[0]

        pos_X = pred_pos[0]
        pos_Y = pred_pos[1]

        return {
            "floor": int(floor),
            "cluster": int(cluster),
            "pos_x": int(pos_X),
            "pos_y": int(pos_Y),
            "elevator": False
        }



"""@app.route('/fondation_ae', methods=['POST'])
def fondation_ae():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = np.array(content["tab"])

    if tab.shape != (3, 25):
        return jsonify({'error': '3 rows of 25 values. Given shape : {}'.format(tab.shape)})

    tab_100 = tab
    tab_100[tab_100 == 0] = -100

    enc_model = Encoder(encoded_space_dim=12, fc2_input_dim=128, window=3)
    enc_model.load_state_dict(torch.load(encoder_weights_fonda))
    enc_model.eval()

    ff_nn_head_model = MultiOutputRegressionHead(encoded_space_dim=12)
    ff_nn_head_model.load_state_dict(torch.load(ff_encoder_head_weights_fonda))
    ff_nn_head_model.eval()

    #in_elevator = model_elevator_fonda_v_mars.predict(tab_90.reshape(1, 25))
    in_elevator = model_elevator_fonda_v_oct_all_floors_all_cluster_data_agreg.predict(tab_100[-1].reshape(1, 25))
    if in_elevator:
        close_elevator = model_elevator_fonda_v_sept.predict(tab_100[-1].reshape(1, 25))
        if close_elevator:
            return {
                "floor": 100,
                "cluster": 100,
                "pos_x": -70,
                "pos_y": 40,
                "elevator": True
            }
        else:
            models_dict = {
                (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
                (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
                (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
                (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
                (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
                (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
                (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
                (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
                (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
                (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
                (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
                (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
                (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
                (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
                (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
                (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
            }

            floor = model_floor_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1, 24)) #model floor with elevator beacon

            if floor == 0 :
                cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 1:
                cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 2:
                cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 3:
                cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 4:
                cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
            elif floor == 5:
                cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]

            norm_tab = normalization(tab_100)
            norm_tab = np.delete(norm_tab, 7, 1) # remove elevator beacon column
            normalized_input_tensor = torch.tensor(norm_tab, dtype=torch.float32)


            pos_pred = ff_nn_head_model(enc_model(
                            torch.unsqueeze(torch.unsqueeze(normalized_input_tensor, dim=0), dim=0)
                                )
                        ).detach().numpy()[0]

            pos_X = pos_pred[0]
            pos_Y = pos_pred[1]

            return {
                "floor": int(floor),
                "cluster": int(cluster),
                "pos_x": int(pos_X),
                "pos_y": int(pos_Y),
                "elevator": True
            }
    else:
        models_dict = {
            (0, 0) : [ knn_c00_x_model_fonda_v_sept, knn_c00_y_model_fonda_v_sept ],
            (0, 1) : [ knn_c01_x_model_fonda_v_sept, knn_c01_y_model_fonda_v_sept ],
            (0, 2) : [ knn_c02_x_model_fonda_v_sept, knn_c02_y_model_fonda_v_sept ],
            (0, 3) : [ knn_c03_x_model_fonda_v_sept, knn_c03_y_model_fonda_v_sept ],
            (1, 0) : [ knn_c10_x_model_fonda_v_sept, knn_c10_y_model_fonda_v_sept ],
            (1, 1) : [ knn_c11_x_model_fonda_v_sept, knn_c11_y_model_fonda_v_sept ],
            (1, 2) : [ knn_c12_x_model_fonda_v_sept, knn_c12_y_model_fonda_v_sept ],
            (2, 0) : [ knn_c20_x_model_fonda_v_sept, knn_c20_y_model_fonda_v_sept ],
            (2, 1) : [ knn_c21_x_model_fonda_v_sept, knn_c21_y_model_fonda_v_sept ],
            (2, 2) : [ knn_c22_x_model_fonda_v_sept, knn_c22_y_model_fonda_v_sept ],
            (3, 0) : [ knn_c30_x_model_fonda_v_sept, knn_c30_y_model_fonda_v_sept ],
            (3, 1) : [ knn_c31_x_model_fonda_v_sept, knn_c31_y_model_fonda_v_sept ],
            (4, 0) : [ knn_c40_x_model_fonda_v_sept, knn_c40_y_model_fonda_v_sept ],
            (4, 1) : [ knn_c41_x_model_fonda_v_sept, knn_c41_y_model_fonda_v_sept ],
            (5, 0) : [ knn_c50_x_model_fonda_v_sept, knn_c50_y_model_fonda_v_sept ],
            (5, 1) : [ knn_c51_x_model_fonda_v_sept, knn_c51_y_model_fonda_v_sept ],
        }

        floor = model_floor_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1, 24)) #model floor with elevator beacon

        if floor == 0 :
            cluster = model_cluster_RDC_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 1:
            cluster = model_cluster_Etage1_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 2:
            cluster = model_cluster_Etage2_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 3:
            cluster = model_cluster_Etage3_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 4:
            cluster = model_cluster_Etage4_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]
        elif floor == 5:
            cluster = model_cluster_Etage5_fonda_v_sept.predict(np.delete(tab_100[-1], 7).reshape(1,24))[0]


        norm_tab = normalization(tab_100)
        norm_tab = np.delete(norm_tab, 7, 1) # remove elevator beacon column
        normalized_input_tensor = torch.tensor(norm_tab, dtype=torch.float32)


        pos_pred = ff_nn_head_model(enc_model(
                        torch.unsqueeze(torch.unsqueeze(normalized_input_tensor, dim=0), dim=0)
                            )
                    ).detach().numpy()[0]

        pos_X = pos_pred[0]
        pos_Y = pos_pred[1]

        return {
            "floor": int(floor),
            "cluster": int(cluster),
            "pos_x": int(pos_X),
            "pos_y": int(pos_Y),
            "elevator": False
        }"""

@app.route('/victor_hugo', methods=['POST'])
def victor_hugo():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = content["tab"]
    if len(tab) != 56:
        return jsonify({'error': '56 values needed'})

    tab_ = np.array([tab])
    tab_[tab_ == 0] = -900

    position_data = [[ "F2P0_", 1120, 1412 ],
                    [ "F2P1_", 790, 1580 ],
                    [ "F2P2_", 790, 1780 ],
                    [ "F2P3_", 1120, 1580 ],
                    [ "F2P4_", 1120, 1780 ],
                    [ "F2P5_", 790, 1950 ],
                    [ "F2P6_", 790, 2200 ],
                    [ "F2P8_", 1030, 1950 ],
                    [ "F2P9_", 1030, 2200 ],
                    [ "F2P10", 790, 2420 ],
                    [ "F2P11", 1090, 2420 ],
                    [ "F2P12", 790, 2715 ],
                    [ "F2P13", 790, 2970 ],
                    [ "F2P14", 1050, 2715 ],
                    [ "F2P15", 1050, 2970 ],
                    [ "F2P16", 840, 3310 ],
                    [ "F2P17", 840, 3660 ],
                    [ "F2P18", 950, 3470 ],
                    [ "F2P19", 1140, 3290 ],
                    [ "F2P20", 1140, 3660 ],
                    [ "F2P21", 1430, 3240 ],
                    [ "F2P22", 1430, 3610 ],
                    [ "F2P23", 1580, 3430 ],
                    [ "F2P24", 1740, 3240 ],
                    [ "F2P25", 1740, 3610 ],
                    [ "F2P26", 2050, 3620 ],
                    [ "F2P27", 2370, 3620 ],
                    [ "F2P28", 2470, 3290 ]]
    position_data = pd.DataFrame(position_data, columns=["Point", "X", "Y"])

    cluster_models = [ knn_floor1_cluster_model_vic, knn_floor2_cluster_model_vic ]

    models_dict = {
        "C20" : knn_c20_position_model_vic,
        "C21" : knn_c21_position_model_vic,
        "C22" : knn_c22_position_model_vic,
        "C23" : knn_c23_position_model_vic,
        "C24" : knn_c24_position_model_vic,
        "C25" : knn_c25_position_model_vic,
        "C26" : knn_c26_position_model_vic,
        "E12E" : knn_e12e_position_model_vic
    }

    cluster_centers = {
        "C10" : [950, 2050],
        "C11" : [950, 2850],
        "C12" : [1300, 3470],
        "C13" : [2250, 3620],
        "E01E" : [2490, 3310],
        "E12E" : [2490, 3310]
    }

    floor = knn_floor_model_vic.predict(tab_.reshape(1, 56))[0]

    pred_cluster = cluster_models[floor - 1].predict(tab_.reshape(1, 56))[0]

    if floor == 2 or pred_cluster == "E12E":
        position = models_dict[pred_cluster].predict(tab_.reshape(1, 56))[0]
        x = position_data[position_data.Point == position].X.unique()[0]
        y = position_data[position_data.Point == position].Y.unique()[0]
    else:
        x, y = cluster_centers[pred_cluster]

    if pred_cluster == "E12E":
        cluster_nb = 7
    elif pred_cluster == "E01E":
        cluster_nb = 4
    else:
        cluster_nb = int(pred_cluster[-1])

    return {"floor": int(floor),
            "cluster": int(cluster_nb),
            "pos_x": int(x),
            "pos_y": int(y)}

@app.route('/bellecour', methods=['POST'])
def bellecour():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = content["tab"]
    if len(tab) != 85:
        return jsonify({'error': '85 values needed'})

    tab_ = np.array([tab])
    tab_[tab_ == 0] = -100

    position_data = [["F2P0_", 850, 1755],
                     ["F2P1_", 970, 1725],
                     ["F2P2_", 1110, 1725],
                     ["F2P3_", 1240, 1725],
                     ["F2P4_", 1330, 1690],
                     ["F2P5_", 1475, 1650],
                     ["F2P6_", 1430, 1600],
                     ["F2P7_", 1385, 1525],
                     ["F2P8_", 1410, 1420],
                     ["F2P9_", 1445, 1505],
                     ["F2P10", 1520, 1465],
                     ["F2P11", 1530, 1590],
                     ["F2P12", 1525, 1555],
                     ["F2P13", 1550, 1530],
                     ["F2P14", 2035, 1680],
                     ["F2P15", 2205, 1726],
                     ["F2P16", 2275, 1960],
                     ["F2P17", 2350, 2010],
                     ["F2P18", 2375, 1910],
                     ["F2P19", 2585, 1850],
                     ["F2P20", 3070, 2025],
                     ["F2P21", 3105, 1915],
                     ["F2P22", 3120, 1810],
                     ["F2P23", 3170, 2005],
                     ["F2P24", 3180, 1900],
                     ["F2P25", 3025, 1960],
                     ["F2P26", 3030, 1915],
                     ["F2P27", 3040, 1880],
                     ["F2P28", 3290, 1940],
                     ["F2P29", 3335, 1955],
                     ["F2P30", 3330, 1875],
                     ["F2P31", 1625, 1490],
                     ["F2P32", 1740, 1515],
                     ["F2P33", 1890, 1555],
                     ["F2P34", 1565, 1680],
                     ["F2P35", 1680, 1705],
                     ["F2P36", 1800, 1740],
                     ["F2P37", 1980, 1635],
                     ["F2P38", 1945, 1765],
                     ["F2P39", 2030, 1783],
                     ["F2P40", 2110, 1800],
                     ["F2P41", 2200, 1795],
                     ["F2P42", 2310, 1850],
                     ["F2P43", 2465, 1885],
                     ["F2P44", 2620, 1925],
                     ["F2P45", 2780, 1960],
                     ["F2P46", 2870, 1985],
                     ["F2P47", 2985, 2010],
                     ["F2P48", 2910, 1990],
                     ["F2P49", 3210, 1710],
                     ["F2P50", 3375, 1830],
                     ["F2P51", 3480, 1735],
                     ["F2P52", 3575, 1625],
                     ["F3P0_", 1670, 1625],
                     ["F3P1_", 1675, 1585],
                     ["F3P2_", 1685, 1560],
                     ["F3P3_", 1710, 1655],
                     ["F3P4_", 1735, 1570],
                     ["F3P5_", 1770, 1630],
                     ["F3P6_", 1805, 1675],
                     ["F3P7_", 1825, 1590],
                     ["F3P8_", 2125, 1705],
                     ["F3P9_", 2215, 1765],
                     ["F3P10", 2235, 1690],
                     ["F3P11", 2325, 1795],
                     ["F3P12", 2345, 1755],
                     ["F3P13", 2355, 1715],
                     ["F3P14", 2580, 1855],
                     ["F3P15", 2600, 1765],
                     ["F3P16", 2680, 1875],
                     ["F3P17", 2705, 1785],
                     ["F3P18", 2725, 1885],
                     ["F3P19", 2750, 1845],
                     ["F3P20", 2750, 1805],
                     ["F3P21", 1890, 1695],
                     ["F3P22", 1995, 1720],
                     ["F3P23", 2005, 1640],
                     ["F3P24", 1910, 1615],
                     ["F3P25", 2055, 1735],
                     ["F3P26", 2155, 1755],
                     ["F3P27", 2175, 1675],
                     ["F3P28", 2075, 1655],
                     ["F3P29", 2395, 1810],
                     ["F3P30", 2515, 1830],
                     ["F3P31", 2415, 1725],
                     ["F3P32", 2535, 1755],
                     ["F3P33", 2465, 1785]]

    position_data = pd.DataFrame(position_data, columns=["Point", "X", "Y"])
    cluster_models = { 2 : knn_floor2_cluster_model, 3 : knn_floor3_cluster_model }

    level2_models_dict = {
        "C20" : knn_c20_position_model,
        "C22" : knn_c22_position_model,
        "C23" : knn_c23_position_model,
        "C2B" : knn_c2b_position_model,
        "C2C" : knn_c2c_position_model,
        "C2D" : knn_c2d_position_model,
        "C2E" : knn_c2e_position_model,
        "C2F" : knn_c2f_position_model,
        "CE1" : knn_f2_ce1_position_model,
        "CE2" : knn_f2_ce2_position_model,
        "CE4" : knn_f2_ce4_position_model,
        "CE5" : knn_f2_ce5_position_model,
        "CS2" : knn_f2_cs2_position_model,
        "CS3" : knn_f2_cs3_position_model,
        "CS4" : knn_f2_cs4_position_model,
        "CS6" : knn_f2_cs6_position_model
    }

    level3_models_dict = {
        "C30" : knn_c30_position_model,
        "C31" : knn_c31_position_model,
        "C32" : knn_c32_position_model,
        "C34" : knn_c34_position_model,
        "C35" : knn_c35_position_model,
        "CE1" : knn_f3_ce1_position_model,
        "CE2" : knn_f3_ce2_position_model,
        "CE4" : knn_f3_ce4_position_model,
        "CE5" : knn_f3_ce5_position_model,
        "CS2" : knn_f3_cs2_position_model,
        "CS3" : knn_f3_cs3_position_model,
        "CS4" : knn_f3_cs4_position_model,
        "CS6" : knn_f3_cs6_position_model
    }

    cluster_2_convert = { "C20" : 0,
                            "C21" : 1,
                            "C22" : 2,
                            "C23" : 3,
                            "C24" : 4,
                            "C25" : 5,
                            "C26" : 6,
                            "C27" : 7,
                            "C28" : 8,
                            "C29" : 9,
                            "C2A" : 10,
                            "C2B" : 11,
                            "C2C" : 12,
                            "C2D" : 13,
                            "C2E" : 14,
                            "C2F" : 15,
                            "CS2" : 16,
                            "CE1" : 17,
                            "CS3" : 18,
                            "CS4" : 19,
                            "CE2" : 20,
                            "CA2" : 21,
                            "CE4" : 22,
                            "CS6" : 23,
                            "CE5" : 24 }

    cluster_3_convert = { "C30" : 0,
                        "C31" : 1,
                        "C32" : 2,
                        "C33" : 3,
                        "C34" : 4,
                        "C35" : 5,
                        "CS2" : 7,
                        "CE1" : 6,
                        "CS3" : 8,
                        "CS4" : 9,
                        "CE2" : 10,
                        "CA2" : 14,
                        "CE4" : 13,
                        "CS6" : 12,
                        "CE5" : 11 }

    floor = knn_floor_model.predict(tab_.reshape(1, 85))[0]

    pred_cluster = cluster_models[floor].predict(tab_.reshape(1, 85))[0]

    if floor == 2:
        position = level2_models_dict[pred_cluster].predict(tab_.reshape(1, 85))[0]
        x = position_data[position_data.Point == position].X.unique()[0]
        y = position_data[position_data.Point == position].Y.unique()[0]
        cluster = cluster_2_convert[pred_cluster]
    elif floor == 3:
        position = level3_models_dict[pred_cluster].predict(tab_.reshape(1, 85))[0]
        x = position_data[position_data.Point == position].X.unique()[0]
        y = position_data[position_data.Point == position].Y.unique()[0]
        cluster = cluster_3_convert[pred_cluster]

    if "floor" in content:
        floor_forced = content["floor"]

        if floor_forced == 2:
            cluster_forced = cluster_2_convert[cluster_models[floor_forced].predict(tab_.reshape(1, 85))[0]]

            return {"floor": int(floor),
                    "cluster": int(cluster),
                    "cluster_forced": int(cluster_forced),
                    "pos_x": int(x),
                    "pos_y": int(y)}
        elif floor_forced == 3:
            cluster_forced = cluster_3_convert[cluster_models[floor_forced].predict(tab_.reshape(1, 85))[0]]

            return {"floor": int(floor),
                    "cluster": int(cluster),
                    "cluster_forced": int(cluster_forced),
                    "pos_x": int(x),
                    "pos_y": int(y)}

    if "simu" in content:
        if content["simu"] == "true":
            if floor == 2:
                cluster_forced = cluster_3_convert[cluster_models[3].predict(tab_.reshape(1, 85))[0]]

                return {"floor": int(floor),
                        "cluster": int(cluster),
                        "cluster_forced": int(cluster_forced),
                        "pos_x": int(x),
                        "pos_y": int(y)}
            elif floor == 3:
                cluster_forced = cluster_2_convert[cluster_models[2].predict(tab_.reshape(1, 85))[0]]

                return {"floor": int(floor),
                        "cluster": int(cluster),
                        "cluster_forced": int(cluster_forced),
                        "pos_x": int(x),
                        "pos_y": int(y)}

    return {"floor": int(floor),
            "cluster": int(cluster),
            "pos_x": int(x),
            "pos_y": int(y)}

@app.route('/gare_de_lyon', methods=['POST'])
def gare_de_lyon():
    content = request.json

    if content is None:
        return jsonify({'error': 'missing data'})

    if not "tab" in content:
        return jsonify({'error': 'missing tab data'})

    tab = content["tab"]
    if len(tab) != 301:
        return jsonify({'error': '301 values needed'})

    tab_ = np.array([tab])
    tab_[tab_ == 0] = -100

    position_data = [[ "F1P0_", 85, 143 ],
                    [ "F1P1_", 85, 112 ],
                    [ "F1P2_", 145, 125 ],
                    [ "F1P3_", 177, 155 ],
                    [ "F1P4_", 177, 106 ],
                    [ "F1P5_", 177, 50 ],
                    [ "F1P6_", 235, 155 ],
                    [ "F1P7_", 235, 106 ],
                    [ "F1P8_", 235, 50 ],
                    [ "F1P9_", 286, 50 ],
                    [ "F1P10", 295, 319 ],
                    [ "F1P11", 295, 270 ],
                    [ "F1P12", 223, 225 ],
                    [ "F1P13", 259, 225 ],
                    [ "F1P14", 295, 225 ],
                    [ "F1P15", 223, 183 ],
                    [ "F1P16", 259, 183 ],
                    [ "F1P17", 295, 183 ],
                    [ "F1P18", 328, 166 ],
                    [ "F1P19", 352, 166 ],
                    [ "F1P20", 375, 166 ],
                    [ "F1P21", 328, 113 ],
                    [ "F1P22", 352, 113 ],
                    [ "F1P23", 375, 113 ],
                    [ "F1P24", 328, 60 ],
                    [ "F1P25", 352, 60 ],
                    [ "F1P26", 375, 60 ]]
    position_data = pd.DataFrame(position_data, columns=["Point", "X", "Y"])

    cluster_models = [ knn_cluster_model_gdl ]

    models_dict = {
        "C10" : knn_c10_position_model_gdl,
        "C11" : knn_c11_position_model_gdl,
        "C12" : knn_c12_position_model_gdl
    }

    floor = knn_floor_model_gdl.predict(tab_.reshape(1, 301))[0]

    pred_cluster = cluster_models[floor - 1].predict(tab_.reshape(1, 301))[0]

    position = models_dict[pred_cluster].predict(tab_.reshape(1, 301))[0]
    x = position_data[position_data.Point == position].X.unique()[0]
    y = position_data[position_data.Point == position].Y.unique()[0]

    cluster_nb = int(pred_cluster[-1])

    return {"floor": int(floor),
            "cluster": int(cluster_nb),
            "pos_x": int(x),
            "pos_y": int(y)}

@app.route('/pepiniere', methods=['POST'])
def pepiniere():
    content = request.json
    tab = content["tab"]
    floor = np.array([0])
    if len(tab) != 12:
        return 'Le tableau doit contenir 12 valeurs', 400
    tab_ = np.array([tab])
    tab_[tab_ == 0] = -90

    cluster = model_cluster_pep.predict(tab_.reshape(1, 12))
    pos_X = model_X_pep.predict(tab_.reshape(1, 12))
    pos_Y = model_Y_pep.predict(tab_.reshape(1, 12))

    return {"cluster": cluster[0].tolist(),
            "pos_x": pos_X[0].tolist(),
            "pos_y": pos_Y[0].tolist(),
            "floor": floor[0].tolist()}


@app.route('/stair_state', methods=['POST'])
def stair_state():
    content = request.json
    tab = content["tab"]
    if len(tab) != 15:
        return 'Le tableau doit contenir 15 valeurs', 400
    tab_ = np.array([tab])

    stair = model_stair.predict(tab_.reshape(1, 15))
    state = model_state.predict(tab_.reshape(1, 15))

    return {"state": state[0].tolist(),
            "stair": stair[0].tolist()}




if __name__ == '__main__':
    app.run(debug=True)
