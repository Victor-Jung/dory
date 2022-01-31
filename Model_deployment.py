# -*- coding: future_fstrings -*-     # should work even without -*-
# -*- coding: utf-8 -*-
# Model_deployment.py
# Alessio Burrello <alessio.burrello@unibo.it>
# Thorir Mar Ingolfsson <thoriri@iis.ee.ethz.ch>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from tiling import Tiling
import template as template
import os
import pandas as pd
from mako.template import Template
from collections import OrderedDict
import logging

class Model_deployment():
    """
    Used to manage the PULP graph. By now, supported Convolutions, Pooling, Linear Layers and Relu.
    """

    def __init__(self, platform, chip):
        self.platform = platform
        self.chip = chip

    def copy_files(self, optional, layer_mixed_list,version, sdk, dma_parallelization):
        ## copy backend and necessary files in the application folder
        os.system('rm -rf application')
        os.system('mkdir application')
        os.system('mkdir application/DORY_network')
        os.system('mkdir application/DORY_network/inc')
        os.system('mkdir application/DORY_network/src')
        tk = OrderedDict([])
        tk['sdk'] = sdk
        root = '/'.join(os.getcwd().split('/')[:-1])
        tmpl = Template(filename=root + "/templates/dory.h")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/inc/dory.h'
        with open(save_string, "w") as f:
            f.write(s)
        os.system('cp ../templates/mem_controller.c  ./application/DORY_network/src/')
        os.system('cp ../templates/mem_controller.h  ./application/DORY_network/inc/')
        tk = OrderedDict([])
        tk['sdk'] = sdk
        tk = OrderedDict([])
        tk['chip'] = self.chip
        tk['dma_parallelization'] = dma_parallelization
        tmpl = Template(filename=root+"/templates/dory.c")
        s = tmpl.render(**tk)
        save_string = './application/DORY_network/src/dory.c'
        with open(save_string, "w") as f:
            f.write(s)
        os.system('cp ../templates/test_template.c ./application/DORY_network/src/')
        os.system('cp ../templates/network.h ./application/DORY_network/inc/')
        if optional == "1D_Conv":
            os.system('cp ../pulp-nn-1d/' + version +'/include/*  ./application/DORY_network/inc/')
            os.system('cp ../pulp-nn-1d/' + version +'/src/* ./application/DORY_network/src/')
        elif optional == "8bit":
            os.system('cp ../pulp-nn/' + version +'/include/*  ./application/DORY_network/inc/')
            os.system('cp ../pulp-nn/' + version +'/src/* ./application/DORY_network/src/')
        elif optional == "mixed-sw":
            os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/include/*  ./application/DORY_network/inc/')
            for layer in layer_mixed_list:
                if layer.split('_')[2] == 'conv':
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/Convolution/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'depthwise':
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/Depthwise/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'matmul':
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/MatrixMultiplication/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'linear':
                    if layer.split('_')[4] == 'i32':
                        os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/LinearNoQuant/' + layer + ' ./application/DORY_network/src/')
                    else:
                        os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/LinearQuant/' + layer + ' ./application/DORY_network/src/')
                elif 'avgpool' in layer.split('_')[2]:
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/Pooling/AvgPool/' + layer + ' ./application/DORY_network/src/')
                elif 'maxpool' in layer.split('_')[2]:
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/Pooling/MaxPool/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'add':
                    os.system('cp ../pulp-nn-mixed/XpulpV2/' + version +'/src/Add/' + layer + ' ./application/DORY_network/src/')
        elif optional == "mixed-hw":
            os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/include/*  ./application/DORY_network/inc/')
            for layer in layer_mixed_list:
                if layer.split('_')[2] == 'conv':
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/Convolution/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'depthwise':
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/Depthwise/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'matmul':
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/MatrixMultiplication/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'linear':
                    if layer.split('_')[4] == 'i32':
                        os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/LinearNoQuant/' + layer + ' ./application/DORY_network/src/')
                    else:
                        os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/LinearQuant/' + layer + ' ./application/DORY_network/src/')
                elif 'avgpool' in layer.split('_')[2]:
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/Pooling/AvgPool/' + layer + ' ./application/DORY_network/src/')
                elif 'maxpool' in layer.split('_')[2]:
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/Pooling/MaxPool/' + layer + ' ./application/DORY_network/src/')
                elif layer.split('_')[2] == 'add':
                    os.system('cp ../pulp-nn-mixed/XpulpNN/' + version +'/src/Add/' + layer + ' ./application/DORY_network/src/')

    def copy_backend(self, optional, BitIn, BitW, BitOut, BitActivation, PULP_Nodes_Graph, number_of_deployed_layers, precision_dict_act, precision_dict_weights, sdk, dma_parallelization):
        layer_mixed_list = []
        ####################################################################################
        ###### SECTION 1: BACKEND FILE SELECTING. SELECTING CORRECT KERNELS TO IMPORT ######
        ####################################################################################
        if 'mixed-sw' in optional:
            for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
                BitIn = BitOut
                if nodes_to_deploy.outshift != 'empty':
                    BitOut = precision_dict_act[i]
                BitW = precision_dict_weights[i]
                if nodes_to_deploy.groups > 1:
                    layer_mixed_list.append(f'pulp_nn_depthwise_u{BitIn}_u{BitOut}_i{BitW}.c')
                else:
                    layer_mixed_list.append(f'pulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
                layer_mixed_list.append(f'pulp_nn_matmul_u{BitOut}_i{BitW}.c')
                if i == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                    BitOut = 32
                if 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                    layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_i{BitOut}_i{BitW}.c')
            layer_mixed_list.append('pulp_nn_add_u8_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u8.c')
            layer_mixed_list.append('pulp_nn_maxpool_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u4.c')
            layer_mixed_list.append('pulp_nn_maxpool_u4.c')
            layer_mixed_list.append('pulp_nn_avgpool_u2.c')
            layer_mixed_list.append('pulp_nn_maxpool_u2.c')
        if 'mixed-hw' in optional:
            for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
                BitIn = BitOut
                if nodes_to_deploy.outshift != 'empty':
                    BitOut = precision_dict_act[i]
                BitW = precision_dict_weights[i]
                if nodes_to_deploy.groups > 1:
                    layer_mixed_list.append(f'xpulp_nn_depthwise_u{BitIn}_u{BitOut}_i{BitW}.c')
                else:
                    layer_mixed_list.append(f'xpulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
                layer_mixed_list.append(f'xpulp_nn_matmul_u{BitIn}_u{BitOut}_i{BitW}.c')
                if i == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                    BitOut = 32
                if 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                    layer_mixed_list.append(f'pulp_nn_linear_u{BitIn}_i{BitOut}_i{BitW}.c')
            layer_mixed_list.append('pulp_nn_add_u8_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u8.c')
            layer_mixed_list.append('pulp_nn_maxpool_u8.c')
            layer_mixed_list.append('pulp_nn_avgpool_u4.c')
            layer_mixed_list.append('pulp_nn_maxpool_u4.c')
            layer_mixed_list.append('pulp_nn_avgpool_u2.c')
            layer_mixed_list.append('pulp_nn_maxpool_u2.c')
        version = str(BitActivation) + 'bit'
        self.copy_files(optional, layer_mixed_list, version, sdk, dma_parallelization)

    def create_weights_files(self, PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, precision_dict_weights):
        ####################################################################################
        ###### SECTION 2: WEIGHTS FILES CREATION. CREATING .HEX FILES FOR EACH LAYER  ######
        ####################################################################################
        file_list_w = []
        # Fetching weights,biases, k, and lambda for each node_iterating
        # 32 bits and 64 bits for Bn and Relu weights are used
        weights_to_write = []
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if str(nodes_to_deploy.weights) != 'empty':
                nodes_to_deploy.weights = nodes_to_deploy.weights.flatten().tolist()
                for i_w, _ in enumerate(nodes_to_deploy.weights):
                    nodes_to_deploy.weights[i_w] = np.uint8(nodes_to_deploy.weights[i_w])
                if precision_dict_weights[i] == 4:
                    temp = []
                    z = 0
                    for _, i_x in enumerate(nodes_to_deploy.weights):
                        if (z % 2) == 0:
                            temp.append(nodes_to_deploy.weights[i_w]& 0x0F)
                        else:
                            temp[-1] += i_x << 4
                        z += 1
                    nodes_to_deploy.weights = temp
                elif precision_dict_weights[i] == 2:
                    temp = []
                    z = 0
                    for _, i_x in enumerate(nodes_to_deploy.weights):
                        if (z % 4) == 0:
                            temp.append(nodes_to_deploy.weights[i_w]& 0x03)
                        else:
                            temp[-1] += i_x << 2 * (z % 4)
                        z += 1
                    nodes_to_deploy.weights = temp
                weights = nodes_to_deploy.weights
            if str(nodes_to_deploy.bias) != 'empty':
                nodes_to_deploy.bias = nodes_to_deploy.bias.flatten().tolist()
                for i_w, _ in enumerate(nodes_to_deploy.bias):
                    nodes_to_deploy.bias[i_w] = np.uint8(nodes_to_deploy.bias[i_w])
                weights = np.concatenate((weights, nodes_to_deploy.bias))
            if str(nodes_to_deploy.k) != 'empty':
                if str(nodes_to_deploy.outmul) != 'empty':
                    out_mult = np.int32(nodes_to_deploy.outmul)
                k_byte = []
                for i_k, _ in enumerate(nodes_to_deploy.k.flatten()):
                    if BitActivation == 64:
                        val = np.int64(nodes_to_deploy.k.flatten()[i_k])*out_mult
                    else:
                        val = np.int32(nodes_to_deploy.k.flatten()[i_k])*out_mult
                    if BitActivation == 32:
                        k_byte.append(np.uint8(val         & 0x000000FF))
                        k_byte.append(np.uint8((val >> 8)  & 0x000000FF))
                        k_byte.append(np.uint8((val >> 16) & 0x000000FF))
                        k_byte.append(np.uint8((val >> 24) & 0x000000FF))
                    if BitActivation == 64:
                        k_byte.append(np.uint8(val         & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 8)  & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                        k_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))
                nodes_to_deploy.k = k_byte

                weights = np.concatenate((weights, nodes_to_deploy.k))
            if str(nodes_to_deploy.lambd) != 'empty':
                lambd = np.float64(nodes_to_deploy.lambd.flatten()) * out_mult
                try:
                    lambd.shape[0]
                except:
                    lambd = np.asarray([np.float64(nodes_to_deploy.lambd.flatten()) * out_mult])
                lambd_byte = []
                for i_l, _ in enumerate(nodes_to_deploy.lambd.flatten()):
                    if BitActivation == 64:
                        val = np.int64(lambd[i_l])
                    else:
                        val = np.int32(lambd[i_l])
                    if BitActivation == 32:
                        lambd_byte.append(np.uint8(val &         0x000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x000000FF))
                    if BitActivation == 64:
                        lambd_byte.append(np.uint8(val &         0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))
                nodes_to_deploy.lambd = lambd_byte
                weights = np.concatenate((weights, nodes_to_deploy.lambd))
                if str(nodes_to_deploy.outmul) != 'empty':
                    PULP_Nodes_Graph[i].outmul = 1
            if str(nodes_to_deploy.weights) != 'empty':
                while len(weights) % 4 != 0:
                    weights = np.concatenate((weights, np.asarray([0])))
                weights = np.asarray(weights)
                weights_to_write.append(weights)
                string_layer = nodes_to_deploy.name + str(i) + "_weights.hex"
                file_list_w.append(string_layer)
                save_s = './application/DORY_network/' + string_layer
                with open(save_s, 'wb') as f:
                    for l in weights.astype('uint8').flatten():
                        f.write(bytes((l,)))
        return PULP_Nodes_Graph, file_list_w, weights_to_write

    def create_layers_tiling(self, PULP_Nodes_Graph, 
                            number_of_deployed_layers, 
                            L1_dimension,
                            l2_buffer_size, 
                            BitActivation, 
                            optional, 
                            performance_single_layer, 
                            BitIn,
                            BitW,
                            BitOut,
                            precision_dict_act,
                            precision_dict_weights,
                            sdk,
                            dma_parallelization):
        ####################################################################################
        ###### SECTION 3: PARSING OF EACH LAYER INDEPENDENT. TILING + LAYER CREATION  ######
        ####################################################################################
        name_list = []
        layer_list = []
        stringa_features = []
        name_layer_list = []
        name_layer_list_internal = []       
        MAC_total = 0
        BitOut = BitOut
        Layers_L3_input_act = 0
        Layers_L3_output_act = 0
        Layers_L3_weights = 0
        L2_memory_occupation = 0
        factor_h_out = 1
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if('Conv1D' in nodes_to_deploy.name):
                layer = 'Conv1D'
            elif('Conv' in nodes_to_deploy.name or 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                layer = 'Conv'
            elif('Pool' in nodes_to_deploy.name):
                layer = 'Pool'
            elif('Add' in nodes_to_deploy.name):
                layer = 'Add'

            name_layer = "layer" + nodes_to_deploy.name + str(i)
            ######################## NEED A  FIX ####################################################
            #### OTHERWISE ONLY WEIGHT < L2/2 GO in L2 --> much more L3 tiling not needed############
            #########################################################################################
            tile_factor = 2
            if (i < len(PULP_Nodes_Graph)-1) and ('Conv' in PULP_Nodes_Graph[i+1].name or 'Gemm' in PULP_Nodes_Graph[i+1].name or 'MatMul' in PULP_Nodes_Graph[i+1].name):
                if PULP_Nodes_Graph[i+1].input_channels*PULP_Nodes_Graph[i+1].output_channels*PULP_Nodes_Graph[i+1].filter_size_h*PULP_Nodes_Graph[i+1].filter_size_w > int(l2_buffer_size/tile_factor):
                    weight_overhead = int(l2_buffer_size/tile_factor)
                else:
                    weight_overhead = PULP_Nodes_Graph[i+1].input_channels*PULP_Nodes_Graph[i+1].output_channels*PULP_Nodes_Graph[i+1].filter_size_h*PULP_Nodes_Graph[i+1].filter_size_w +int(PULP_Nodes_Graph[i+1].output_channels*BitActivation/8*2)
            else:
                weight_overhead = 0
            if(optional != '8bit' and optional != '1D_Conv'):
                BitIn = BitOut
                if nodes_to_deploy.outshift != 'empty':
                    BitOut = precision_dict_act[i]
                BitW = precision_dict_weights[i]
            if i == len(PULP_Nodes_Graph)-1:
                name_layer = name_layer + '_last'
                BitOut = 32
            if(performance_single_layer == 'Yes'):
                test_location = 'L3+performance'
            else:
                test_location = 'L3'
            tile_gen = Tiling(layer,
                              nodes_to_deploy.output_channels,
                              [nodes_to_deploy.filter_size_h, nodes_to_deploy.filter_size_w],
                              nodes_to_deploy.stride,
                              [nodes_to_deploy.padding_top,nodes_to_deploy.padding_left,nodes_to_deploy.padding_bottom,nodes_to_deploy.padding_right],
                              nodes_to_deploy.groups,
                              [nodes_to_deploy.input_channels * nodes_to_deploy.groups,
                              nodes_to_deploy.input_h, nodes_to_deploy.input_w],
                              L1_dimension,
                              l2_buffer_size-weight_overhead,
                              self.platform,
                              self.chip,
                              test_location=test_location,
                              BitIn=BitIn,
                              BitW=BitW,
                              BitOut=BitOut,
                              BitActivation = BitActivation,
                              optional_type=optional,
                              sdk = sdk,
                              dma_parallelization = dma_parallelization)
            if(nodes_to_deploy.conv_1d == 0):
                str_l = 'ch_in' + str(nodes_to_deploy.input_channels) + 'ch_out' + str(nodes_to_deploy.output_channels) + 'groups' + str(
                    nodes_to_deploy.groups) + 'dim_image' + str(nodes_to_deploy.input_h,) + str(nodes_to_deploy.input_w,) + 'stride' + str(nodes_to_deploy.stride) + 'kernel'+ str(
                    nodes_to_deploy.filter_size_h) + str(nodes_to_deploy.filter_size_w) + 'kernel' + str(nodes_to_deploy.filter_size_h) + str(nodes_to_deploy.filter_size_w) + 'BitIn' + str(BitIn) + 'BitOut' + str(BitOut) + 'BitW' + str(BitW)
            else:
                str_l = 'ch_in' + str(nodes_to_deploy.input_channels) + 'ch_out' + str(nodes_to_deploy.output_channels) + 'groups' + str(
                    nodes_to_deploy.groups) + 'dim_image' + str(nodes_to_deploy.input_w,) + 'stride' + str(nodes_to_deploy.stride) + 'kernel'+ str(
                    nodes_to_deploy.filter_size_h) + 'kernel' + str(nodes_to_deploy.filter_size_w) + 'BitIn' + str(BitIn) + 'BitOut' + str(BitOut) + 'BitW' + str(
                        BitW) + 'Dilation' + str(nodes_to_deploy.dilation)
            name = nodes_to_deploy.name
            for scan_i, _ in enumerate(stringa_features):
                if(str_l == stringa_features[scan_i] and str(layer) == str(layer_list[scan_i])):
                    name_layer = name_layer_list[scan_i]
                    name = name_layer_list_internal[scan_i]
            stringa_features.append(str_l)
            layer_list.append(layer)
            name_layer_list.append(name_layer)
            name_layer_list_internal.append(name)
            relu = 0
            BN = 0
            DW = 0
            input_dim_constraint = 0
            output_weights_dim_constraint = 0
            if(i == 0):
                weight_constraint = 0
            if(i == 0):
                input_L3 = 0
            elif(factor_h_out > 1):
                input_L3 = 1
                input_dim_constraint = out_dim2
                output_weights_dim_constraint = l2_buffer_size - weight_overhead - out_dim2_old
                if(output_weights_dim_constraint < 0):
                    print("Problems with current implementation on L3 tiling. Prediction of weights of next layer not accurate. Exiting...")
                    os._exit(0)
            else:
                input_L3 = 0
            if('Relu' in nodes_to_deploy.name):
                relu = 1
            if('BN' in nodes_to_deploy.name):
                BN = 1
            if('DW' in nodes_to_deploy.name):
                DW = 1
            if('Conv1D' in nodes_to_deploy.name):
                if nodes_to_deploy.bias == 'empty':
                    h_b = 0
                else:
                    h_b = 1
                in_dim2, out_dim2, weights_dim, l1_dim2 = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                            relu=relu, BN=BN,
                                                                            dilation=nodes_to_deploy.dilation,
                                                                            has_bias=h_b,
                                                                            out_mul=nodes_to_deploy.outmul,
                                                                            out_shift=nodes_to_deploy.outshift,
                                                                            name=name_layer)
                if(i == 0):
                    out_dim2_old = in_dim2
                out_dim2_old = out_dim2
                L3_tiling = 0
                factor_ch_out = 1
            elif('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                if nodes_to_deploy.bias == 'empty':
                    h_b = 0
                else:
                    h_b = 1
                in_dim2, out_dim2, weights_dim, l1_dim2, L3_tiling, factor_ch_out, factor_h_out, factor_h_in = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                            relu=relu, BN=BN, DW=DW,
                                                                            has_bias=h_b,
                                                                            out_mul=nodes_to_deploy.outmul,
                                                                            out_shift=nodes_to_deploy.outshift,
                                                                            name=name_layer,
                                                                            input_L3 = input_L3,
                                                                            input_dim_constraint = input_dim_constraint,
                                                                            output_weights_dim_constraint = output_weights_dim_constraint,
                                                                            weight_constraint = weight_constraint)
                if(factor_ch_out > 1):
                    PULP_Nodes_Graph[i].L3_allocation = 1
                else:
                    PULP_Nodes_Graph[i].L3_allocation = 0
                Layers_L3_input_act += int(factor_h_in > 1)
                Layers_L3_output_act += int(factor_h_out > 1)
                Layers_L3_weights += int(factor_ch_out > 1)
                PULP_Nodes_Graph[i].L3_input = int(factor_h_in > 1)
                PULP_Nodes_Graph[i].L3_output = int(factor_h_out > 1)
                PULP_Nodes_Graph[i].L3_weights = int(factor_ch_out > 1)
                if(i == 0):
                    out_dim2_old = in_dim2
                if(factor_h_out > 1):
                    out_dim2 = l2_buffer_size - weight_overhead - out_dim2_old - weights_dim
                out_dim2_old = out_dim2
            elif('Pool' in nodes_to_deploy.name):
                in_dim2, out_dim2, l1_dim2, L3_tiling, factor_h_out, factor_h_in = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                 relu=relu, BN = BN,
                                                                 out_mul=nodes_to_deploy.outmul,
                                                                 out_shift=nodes_to_deploy.outshift,
                                                                 name=name_layer,
                                                                 input_L3 = input_L3,
                                                                 input_dim_constraint = input_dim_constraint,
                                                                 output_weights_dim_constraint = output_weights_dim_constraint,
                                                                 type=name)
                Layers_L3_input_act += int(factor_h_in > 1)
                Layers_L3_output_act += int(factor_h_out > 1)
                if(i == 0):
                    out_dim2_old = in_dim2
                if(factor_h_out > 1):
                    out_dim2 = l2_buffer_size - weight_overhead - out_dim2_old - weights_dim
                out_dim2_old = out_dim2
            elif('Add' in nodes_to_deploy.name):
                in_dim2, out_dim2, l1_dim2 = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                 relu=relu,
                                                                 out_mul1=nodes_to_deploy.inmul1,
                                                                 out_mul2=nodes_to_deploy.inmul2,
                                                                 out_shift=nodes_to_deploy.outshift,
                                                                 name=name_layer,
                                                                 type=name)
                L3_tiling = 0

            while weights_dim % 4 != 0:
                weights_dim += 1
            if(weight_overhead == int(l2_buffer_size/2)):
                weight_constraint = int(l2_buffer_size/2)
            else:
                weight_constraint = 0
            if(L3_tiling == 1):
                name_layer = name_layer + 'L3'
                PULP_Nodes_Graph[i].input_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].input_h * PULP_Nodes_Graph[i].input_w * PULP_Nodes_Graph[i].input_channels*BitIn/8)
                PULP_Nodes_Graph[i].output_activation_dimensions_L3 = int(PULP_Nodes_Graph[i].output_h * PULP_Nodes_Graph[i].output_w * PULP_Nodes_Graph[i].output_channels*BitOut/8)
            name_list.append(name_layer)
            if('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                if(i > 0):
                    PULP_Nodes_Graph[i].weights_dimension = PULP_Nodes_Graph[i-1].weights_dimension + weights_dim
                else:
                    PULP_Nodes_Graph[i].weights_dimension = weights_dim
            else:
                PULP_Nodes_Graph[i].weights_dimension = PULP_Nodes_Graph[i-1].weights_dimension
            if('Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name):
                if(factor_ch_out == 1):
                    if(i > 0):
                        PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3 + weights_dim
                    else:
                        PULP_Nodes_Graph[i].weights_dimension_L3 = weights_dim
                else:
                    if(i > 0):
                        PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3 + int(weights_dim*factor_ch_out/2)
                    else:
                        PULP_Nodes_Graph[i].weights_dimension_L3 = int(weights_dim*factor_ch_out/2)                    
            else:
                PULP_Nodes_Graph[i].weights_dimension_L3 = PULP_Nodes_Graph[i-1].weights_dimension_L3
            PULP_Nodes_Graph[i].input_activation_dimensions = int(in_dim2*BitIn/8)
            PULP_Nodes_Graph[i].output_activation_dimensions = int(out_dim2*BitOut/8)
            if(i > 0):
                if(PULP_Nodes_Graph[i].input_activation_dimensions != PULP_Nodes_Graph[i-1].output_activation_dimensions):
                    PULP_Nodes_Graph[i].input_activation_dimensions = PULP_Nodes_Graph[i-1].output_activation_dimensions
            PULP_Nodes_Graph[i].l1_dimensions = l1_dim2
            if('Pool' not in nodes_to_deploy.name):
                MAC_total += nodes_to_deploy.MACs
        return PULP_Nodes_Graph, Layers_L3_input_act, Layers_L3_output_act, Layers_L3_weights, name_layer_list, name_list, MAC_total

    def generate_intermediate_activations(self, PULP_Nodes_Graph, 
                                        load_dir, 
                                        number_of_deployed_layers, 
                                        check_layer,
                                        weights_to_write,
                                        BitIn,
                                        BitW,
                                        BitOut,
                                        optional,
                                        precision_dict):
        ######################################################################################
        ###### SECTION 4: GENERATE CHECKSUM BY USING WEIGHT AND OUT_LAYER{i}.TXT FILES  ######
        ######################################################################################
        x_in = None
        x_in = pd.read_csv(load_dir + 'input.txt')
        x_in = x_in.values[:, 0].astype(int)
        for i, _ in enumerate(x_in):
            x_in[i] = np.uint8(x_in[i])
        BitOut = 8
        PULP_Nodes_Graph[0].check_sum_in = sum(x_in)
        string_layer = "inputs.hex"
        save_s = './application/DORY_network/' + string_layer
        with open(save_s, 'wb') as f:
            for i in x_in.astype('uint8').flatten():
                f.write(bytes((i,)))
        f_w = 0
        for f, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            X_in = pd.read_csv(load_dir + 'out_layer' + str(f) + '.txt')
            X_in = X_in.values[:, 0].astype(int)
            if f == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                class_out = np.where(X_in == np.max(X_in))[0][0]
            for i, _ in enumerate(X_in):
                X_in[i] = np.uint8(X_in[i])
            if(optional != '8bit' and optional != '1D_Conv'):
                BitIn = BitOut
                if nodes_to_deploy.outshift != 'empty':
                    BitOut = precision_dict[f]
                if f == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                    BitOut = 32
            Input_compressed = []
            z = 0
            import copy
            Loop_over = copy.deepcopy(X_in)
            if f != len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                for _, i_x in enumerate(Loop_over):
                    if (z % int(8 / BitOut)) == 0:
                        Input_compressed.append(int(i_x.item()))
                    else:
                        Input_compressed[-1] += int(i_x.item()) << (BitOut * (z % int(8 / BitOut)))
                    z += 1
            if check_layer == f:
                act_compare = Input_compressed
            PULP_Nodes_Graph[f].check_sum_out = sum(Input_compressed)
            if f == len(PULP_Nodes_Graph) - 1:
                ww = np.asarray(nodes_to_deploy.weights).reshape(nodes_to_deploy.output_channels,nodes_to_deploy.input_channels ).astype(np.int8).astype(int)
                X_in = pd.read_csv(load_dir + 'out_layer' + str(f-1) + '.txt')
                X_out = pd.read_csv(load_dir + 'out_layer' + str(f) + '.txt')
                X_in = X_in.values[:, 0].astype(int).reshape(X_in.shape[0],1)
                try:
                    PULP_Nodes_Graph[f].check_sum_out = sum(sum(np.matmul(ww,X_in)))
                except:
                    PULP_Nodes_Graph[f].check_sum_out = 0
            if f != len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                PULP_Nodes_Graph[f + 1].check_sum_in = sum(Input_compressed)
            if 'Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
                PULP_Nodes_Graph[f].check_sum_w = sum(weights_to_write[f_w])
                f_w += 1
        return PULP_Nodes_Graph, class_out

    def print_model_network(self, PULP_Nodes_Graph,
                            number_of_deployed_layers=29,
                            load_dir='./mnistNet/',
                            check_layer=0,
                            verbose_level='None',
                            performance_single_layer='Yes',
                            L1_dimension = 35000,
                            master_stack = 4096,
                            slave_stack = 3072,
                            l2_buffer_size = 400000,
                            fc_frequency = 100000000,
                            cl_frequency = 100000000,
                            BitIn=8,
                            BitW=8,
                            BitOut=8,
                            BitActivation = 32,
                            sdk='gap_sdk', 
                            dma_parallelization='8-cores',
                            optional='8bit',
                            precision_dict_act = 'None',
                            precision_dict_weights = 'None'):
        # Function used to create all the files for the application
        # copy backend is used to copy all the files of the backend
        self.copy_backend(optional, BitIn, BitW, BitOut, BitActivation, PULP_Nodes_Graph, number_of_deployed_layers, precision_dict_act, precision_dict_weights, sdk, dma_parallelization)
        # create L3 files for weights. These files are .hex which are copied in hyperflash then
        PULP_Nodes_Graph, weights_files_list, weights_to_write = self.create_weights_files(PULP_Nodes_Graph, number_of_deployed_layers, BitActivation, precision_dict_weights)
        fileh = logging.FileHandler('logs/Tiling_profiling.log', 'a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fileh.setFormatter(formatter)
        fileh.setLevel(logging.DEBUG)
        log = logging.getLogger() 
        for hdlr in log.handlers[:]:
            log.removeHandler(hdlr)
        log.addHandler(fileh)
        print("Creating tiling profiling in Tiling_profling.log")
        # tiling of all the layers. Both tiling and layer generation
        PULP_Nodes_Graph, num_L3_input_tile, num_L3_output_tile, num_L3_weight_tile, name_layer_list, name_list, MAC_total = self.create_layers_tiling(PULP_Nodes_Graph, 
            number_of_deployed_layers, 
            L1_dimension, 
            l2_buffer_size, 
            BitActivation, 
            optional, 
            performance_single_layer,
            BitIn,
            BitW,
            BitOut,
            precision_dict_act,
            precision_dict_weights,
            sdk,
            dma_parallelization)

        logging.debug("  ")
        logging.debug("  Layers with L3 input activation: " + str(num_L3_input_tile))
        logging.debug("  Layers with L3 output activation: " + str(num_L3_output_tile))
        logging.debug("  Layers with L3 weights: " + str(num_L3_weight_tile))

        name_layer_list_unique = list(set(name_layer_list))
        for i, _ in enumerate(name_layer_list_unique):
            name_layer_list_unique[i] = name_layer_list_unique[i] + ".c"
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            if nodes_to_deploy.L3_allocation == 1:
                name_layer_list_unique.append(name_layer_list[i] + "L3" + ".c")
        # compute the checksums for intermediate activations checking
        if 'Check' in verbose_level or 'Last' in verbose_level:
            PULP_Nodes_Graph, class_out = self.generate_intermediate_activations(PULP_Nodes_Graph, 
                load_dir, 
                number_of_deployed_layers, 
                check_layer,
                weights_to_write,
                BitIn,
                BitW,
                BitOut,
                optional,
                precision_dict_act)
        else:
            x_in = torch.Tensor(1, PULP_Nodes_Graph[0].input_channels, PULP_Nodes_Graph[0].input_h, PULP_Nodes_Graph[0].input_w).uniform_(0, (2**(9)))
            x_in[x_in > (2**8 - 1)] = 0
            x_in = torch.round(x_in)
            x_in = x_in.flatten().numpy().astype(int)
            for i, _ in enumerate(x_in):
                x_in[i] = np.uint8(x_in[i])
            BitOut = 8
            class_out = 0
            PULP_Nodes_Graph[0].check_sum_in = sum(x_in)
            string_layer = "inputs.hex"
            save_s = './application/DORY_network/' + string_layer
            with open(save_s, 'wb') as f:
                for i in x_in.astype('uint8').flatten():
                    f.write(bytes((i,)))

        if check_layer == 100:
            act_compare = np.asarray([0, 0])
            act_size = [0, 0, 0]
        else:
            act_size = [PULP_Nodes_Graph[check_layer].output_h, PULP_Nodes_Graph[check_layer].output_w, PULP_Nodes_Graph[check_layer].output_channels]
        ## printf the network file. It calls all the layer functions
        template.print_template_network(
            weights_files_list,
            PULP_Nodes_Graph[:number_of_deployed_layers],
            'char',
            name=name_list,
            test=True,
            has_bias=True,
            verbose_level=verbose_level,
            performance_single_layer = performance_single_layer,
            check_layer=check_layer,
            act_compare=act_compare,
            act_size=act_size,
            class_out=class_out,
            l1_buffer=L1_dimension,
            master_stack = master_stack,
            slave_stack = slave_stack,
            l2_buffer_size = l2_buffer_size,
            fc_frequency = fc_frequency,
            cl_frequency = cl_frequency,
            MACs=MAC_total,
            platform=self.platform,
            BitIn=BitIn,
            BitW=BitW,
            BitOut=BitOut,
            sdk = sdk,
            dma_parallelization = dma_parallelization,
            optional_type = optional)
        # create the Makefile for the application
        template.print_template_Makefile(weights_files_list, self.platform, sdk)
