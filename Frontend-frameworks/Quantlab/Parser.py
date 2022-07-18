     # should work even without -*-
# -*- coding: utf-8 -*-
#!/bin/bash
# ONNX_management.py
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

# Libraries
import json
import os

# DORY modules
from Parsers.Parser_ONNX_to_DORY import Parser_ONNX_to_DORY
from .Pattern_rewriter import Pattern_rewriter

# Directory
file_path = "/".join(os.path.realpath(__file__).split("/")[:-1])


class onnx_manager(Parser_ONNX_to_DORY):
    # Used to manage the ONNX files. By now, supported Convolutions (PW and DW), Pooling, Fully Connected and Relu.

    def __init__(self, onnx, config_file):
        layers_accepted = ['Conv', 'Pad', 'Mul', 'Add', 'Div', 'Constant', 'AveragePool', 'GlobalAveragePool', 'MaxPool', 'Cast', 'Clip', 'Floor', 'Flatten', 'Gemm', 'MatMul', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax', 'Max', 'Min']
        layers_neglected = ['Cast', 'Floor', 'Flatten', 'Shape', 'Gather', 'Unsqueeze', 'Concat', 'Reshape', 'Sigmoid', 'LogSoftmax', 'Min']
        layers_to_node = ['AveragePool', 'MaxPool', 'Conv', 'Gemm', 'MatMul', 'GlobalAveragePool']
        f = open(os.path.join(file_path, "rules.json"))
        rules = json.load(f)
        self.BNRelu_bits = config_file["BNRelu_bits"]
        super().__init__(onnx, rules, layers_accepted, layers_neglected, layers_to_node)

    def frontend_mapping_to_DORY_nodes(self):
        print("\nQuantlab Frontend: Matching patterns from generated ONNX to DORY.")
        for i, node in enumerate(self.DORY_Graph):
            string_matching, indexes = self.pattern_matching(node, i)
            if isinstance(string_matching, str):
                self.DORY_Graph = Pattern_rewriter(self.DORY_Graph).execute(string_matching, indexes)
        print("\nQuantlab Frontend: Updating Add nodes with constants.")
        for i, node in enumerate(self.DORY_Graph):
            if "Addition" in node.name:
                ## output parameters 
                node.outshift = {}
                node.outshift["value"] = node.out_shift
                node.outshift["layout"] = ""
                node.constant_names.append("outshift")
                delattr(node, 'out_shift')
                node.outmul = {}
                node.outmul["value"] = node.out_mul
                node.outmul["layout"] = ""
                node.constant_names.append("outmul")
                delattr(node, 'out_mul')
                node.outadd = {}
                node.outadd["value"] = node.out_add
                node.outadd["layout"] = ""
                node.constant_names.append("outadd")
                delattr(node, 'out_add')
                # input 1 parameters
                #### Look at the order of inputs in Onnx. If the lowest index is not the first argument, revert the order inmul1 and inmul2
                if int(node.input_indexes[0]) > int(node.input_indexes[1]):
                    temp_shift = node.in1_shift
                    temp_mul   = node.in1_mul
                    temp_add   = node.in1_add
                    node.in1_shift = node.in2_shift
                    node.in1_mul   = node.in2_mul
                    node.in1_add   = node.in2_add
                    node.in2_shift = temp_shift
                    node.in2_mul   = temp_mul
                    node.in2_add   = temp_add
                node.inshift1 = {}
                node.inshift1["value"] = node.in1_shift
                node.inshift1["layout"] = ""
                node.constant_names.append("inshift1")
                delattr(node, 'in1_shift')
                node.inmul1 = {}
                node.inmul1["value"] = node.in1_mul
                node.inmul1["layout"] = ""
                node.constant_names.append("inmul1")
                delattr(node, 'in1_mul')
                node.inadd1 = {}
                node.inadd1["value"] = node.in1_add
                node.inadd1["layout"] = ""
                node.constant_names.append("inadd1")
                delattr(node, 'in1_add')
                # input 2 parameters
                node.inshift2 = {}
                node.inshift2["value"] = node.in2_shift
                node.inshift2["layout"] = ""
                node.constant_names.append("inshift2")
                delattr(node, 'in2_shift')
                node.inmul2 = {}
                node.inmul2["value"] = node.in2_mul
                node.inmul2["layout"] = ""
                node.constant_names.append("inmul2")
                delattr(node, 'in2_mul')
                node.inadd2 = {}
                node.inadd2["value"] = node.in2_add
                node.inadd2["layout"] = ""
                node.constant_names.append("inadd2")
                delattr(node, 'in2_add')
                # removing irrelevant parameters
                delattr(node, 'in2_n_levels')
                delattr(node, 'in2_rq')
                delattr(node, 'out_n_levels')
                delattr(node, 'out_rq')
                delattr(node, 'in1_n_levels')
                delattr(node, 'in1_rq')
        
    def add_nodes_precision(self):
        print("\nQuantlab Frontend: Adding Bit and Types to Nodes.")
        # Right now, the precision is fixed. We can extract it from either the original onnx graph or from a json.
        for i, node in enumerate(self.DORY_Graph):
            node.add_existing_parameter("weight_type", "int")
            node.add_existing_parameter("constant_type", "int")
            if i == 0:
                node.add_existing_parameter("input_activation_type", "uint")
                node.add_existing_parameter("input_activation_bits", 8)
            else:
                for previous_nodes in self.DORY_Graph:
                    if previous_nodes.output_index == self.DORY_Graph[i].input_indexes[0]:
                        node.add_existing_parameter("input_activation_type", previous_nodes.output_activation_type)
                        node.add_existing_parameter("input_activation_bits", previous_nodes.output_activation_bits)
                if len(self.DORY_Graph[i].input_indexes) == 2:
                    for previous_nodes in self.DORY_Graph:
                        if previous_nodes.output_index == self.DORY_Graph[i].input_indexes[1]:
                            node.second_input_activation_type = previous_nodes.output_activation_type
                            node.second_input_activation_bits = previous_nodes.output_activation_bits
            if node.name in ["Addition"]:
                node.add_existing_parameter("output_activation_bits", node.add_bits)
                delattr(node, "add_bits")
                node.add_existing_parameter("output_activation_type", self.DORY_Graph[i].input_activation_type)
            if node.name in ["Convolution", "FullyConnected"]:
                node.add_existing_parameter("output_activation_bits", 32)
            if node.name in ["QAddition", "Relu", "BNRelu", "Clip", "Mul", "Add", "Div", "Shift"]:
                node.add_existing_parameter("constant_bits", self.BNRelu_bits)
            if node.name in ["Pooling"]:
                node.add_existing_parameter("output_activation_bits", self.DORY_Graph[i].input_activation_bits)
                node.add_existing_parameter("output_activation_type", self.DORY_Graph[i].input_activation_type)
        node.add_existing_parameter("output_activation_type", "int") # last node is always int

    def add_data_layout(self):
        print("\nQuantlab Frontend: Adding Data Layout.")
        for i, node in enumerate(self.DORY_Graph):
            for name in node.constant_names:
                if name not in ["l","k","outshift","outmul","outadd", "inmul1", "inmul2", "inshift1", "inshift2", "inadd1", "inadd2"]:
                    if "bias" not in name:
                        weights_name = name
            if weights_name in node.__dict__:
                if node.name in "FullyConnected":
                    if node.__dict__[weights_name]["value"].shape[0] == node.input_channels:
                        node.__dict__[weights_name]["layout"] = "CinCout"
                    else:
                        node.__dict__[weights_name]["layout"] = "CoutCin"
                else:
                    node.__dict__[weights_name]["layout"] = "CoutCinK"
            node.layout = "CHW"



