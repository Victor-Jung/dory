/*
 * layer_template_nnx.c
 * Francesco Conti <f.conti@unibo.it>
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2018-2022 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */

#include "${func_name}.h"
#include "pulp_nnx.h"
#ifdef GVSOC_LOGGING
#include "pulp_nnx_util.h"
#endif GVSOC_LOGGING
% if ULTRA_VERBOSE:
// #define VERBOSE_PRINT(...) printf(__VA_ARGS__)
#define VERBOSE_PRINT(...)
% endif

void ${func_name}(
  void *args
) {
  //////////////////////////////////////////////////////////////////////////
  // arguments assigning: keeping same interface between L2 and L3 memory //
  //////////////////////////////////////////////////////////////////////////
  unsigned int *real_arg = (unsigned int *) args;
  unsigned int l3_x =(unsigned int)  real_arg[0];
  unsigned int l3_y =(unsigned int)  real_arg[1];
  unsigned int l3_W =(unsigned int)  real_arg[2];
  unsigned int l2_x =(unsigned int)  real_arg[3];
  unsigned int l2_x_2 =(unsigned int)  real_arg[4];
  unsigned int l2_y =(unsigned int)  real_arg[5];
  unsigned int l2_W =(unsigned int)  real_arg[6];
  unsigned int l1_buffer =(unsigned int)  real_arg[7];
  unsigned int hyperram =(unsigned int)  real_arg[8];
  unsigned int inmul1 = (unsigned int) real_arg[10];
  unsigned int inmul2 = (unsigned int) real_arg[11];
  unsigned int out_shift_in = (unsigned int) real_arg[12];

  /////////////////////
  // DMA declaration //
  /////////////////////
  uint32_t dory_dma_channel = dory_dma_allocate();
  DMA_copy DMA_copy_k, DMA_copy_lambda;
  DMA_copy DMA_copy_W, DMA_copy_x, DMA_copy_y;

% if has_bias == 1:
  DMA_copy DMA_copy_bias;
  DMA_copy_bias.hwc_to_chw = 0;
  DMA_copy_bias.stride_2d = 0;
  DMA_copy_bias.stride_1d = 0;
  DMA_copy_bias.dir = 1;
  DMA_copy_bias.dma_channel = dory_dma_channel;
% endif

  DMA_copy_k.hwc_to_chw = 0;
  DMA_copy_k.stride_2d = 0;
  DMA_copy_k.stride_1d = 0;
  DMA_copy_k.dir = 1;
  DMA_copy_k.dma_channel = dory_dma_channel;

  DMA_copy_lambda.hwc_to_chw = 0;
  DMA_copy_lambda.stride_2d = 0;
  DMA_copy_lambda.stride_1d = 0;
  DMA_copy_lambda.dir = 1;
  DMA_copy_lambda.dma_channel = dory_dma_channel;
  
% if flag_DW == 1:
  DMA_copy_x.hwc_to_chw = 1;
% else:
  DMA_copy_x.hwc_to_chw = 0;
% endif  

  DMA_copy_x.stride_2d = ${x_stride_w_byte};
  DMA_copy_x.stride_1d = ${x_stride_c_byte};
  DMA_copy_x.dir = 1;
  DMA_copy_x.dma_channel = dory_dma_channel;
  
  DMA_copy_W.hwc_to_chw = 0;
  DMA_copy_W.stride_2d = ${W_stride_nof_byte};
  DMA_copy_W.stride_1d = ${W_stride_hw_byte};
  DMA_copy_W.dir = 1;
  DMA_copy_W.dma_channel = dory_dma_channel;
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ${y_stride_w_byte};
  DMA_copy_y.stride_1d = ${y_stride_c_byte};
  DMA_copy_y.dir = 0;
  DMA_copy_y.dma_channel = dory_dma_channel;

  int p_r, p_l, p_t, p_b;

% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
  unsigned short x_length_nif_byte;
  int pad_offset_h, pad_offset_w;
% endif  

  int y_tile_size_h;
  int y_tile_size_w;
  int y_length_nof_byte;
  unsigned short  x_tile_size_h;
  unsigned short  x_tile_size_w;
  unsigned short  W_tile_size_nof;
  unsigned short  W_tile_size_nif;
  unsigned short  W_tile_size_byte;
  unsigned short W_length_nif_byte;
  ${type} *x, *W, *y, *b;

% if FLAG_BATCHNORM == 1:
  % if act_dim_bit == 32:
  int32_t *k;
  int32_t *lambda;
  % else:
  int64_t *k;
  int64_t *lambda;
  % endif
% endif

  int db_x = 0;
  int db_W = 0;
  int db_act = 0;
  int db_y = 0;
  int exec_db_x = 0;
  int exec_db_W = 0;
  int exec_db_act = 0;
  int store_db_y = 0;

  // double buffering state
  int db_state_x=0;
  int db_state_W=0;
  int db_state_y=0;
  // last-tile flags
  int iter;
  // tile loop indeces
  int _i_nof_load=0, _i_nif_load=0, _i_h_load=0, _i_w_load=0;
  int _i_nof_exec=0, _i_nif_exec=0, _i_h_exec=0, _i_w_exec=0;

% if has_bias == 1:
  int has_bias = 1;
% endif

% if FLAG_RELU == 1:
  uint16_t out_shift = out_shift_in;
% endif

  nnx_task_t nnx_task, nnx_task_remainder;
  // init accelerated task
  nnx_soft_clear();
  nnx_task_init(&nnx_task);

  const int l1_buffer_x = l1_buffer + ${l1_x_offset};
  const int l1_buffer_y = l1_buffer + ${l1_y_offset};
  const int l1_buffer_w = l1_buffer + ${l1_W_offset};
  const int l1_buffer_k = l1_buffer + ${l1_k_offset};
  const int l1_buffer_lambda = l1_buffer + ${l1_lambda_offset};

  nnx_weights_t nnx_weights = {
    .data = l1_buffer_w,
    .height = ${fs1},
    .width = ${fs2},
    .depth = ${x_tile_size_nif},
    .n_weights = ${y_tile_size_nof},
    .bitwidth = 8,
    .offset_factor = -128,
    .offset_mode = weightOffsetModeLayerWise
  };

  nnx_feature_t nnx_input = {
    .data = l1_buffer_x,
    .height = ${x_tile_size_h},
    .width = ${x_tile_size_w},
    .depth = ${x_tile_size_nif},
    .bitwidth = featureBitwidth8Bit
  };

  nnx_feature_t nnx_output = {
    .data = l1_buffer_y,
    .height = ${y_tile_size_h},
    .width = ${y_tile_size_w},
    .depth = ${y_tile_size_nof},
    .bitwidth = featureBitwidth8Bit
  };

  nnx_conv_${fs1}x${fs2}(&nnx_task, nnx_weights, nnx_input, nnx_output);

  // PULP-NN like defaults
  nnx_norm_t norm = {
    .mode  = normMode32Bit,
    .scale = l1_buffer_k,
    .bias  = l1_buffer_lambda,
    .shift = NE16_NULL
  };

  nnx_quant_t quant = {
    .shift_amount = out_shift,
    .mode = quantMode8Bit,
    .function = quantFunctionRelu,
    .use_rounding = 0
  };

  nnx_norm_quant(&nnx_task, norm, quant);

#ifdef GVSOC_LOGGING
  nnx_activate_gvsoc_logging(1);
#endif

  VERBOSE_PRINT("Acquire iter=PRE\n");
  int id = nnx_acquire();
  nnx_offload(&nnx_task);

% if tile_dim_nof * tile_dim_h * tile_dim_w * tile_dim_nif != 1:
  nnx_commit();
% endif

  VERBOSE_PRINT("  Job_id=%d\n", id);

  ////////////////////////////
  // First tile transfering //
  ////////////////////////////
% if has_bias == 1:
  DMA_copy_bias.ext = (uint32_t) l2_W+${l2_off_bias};
  DMA_copy_bias.loc = (uint32_t) (l1_buffer + ${l1_b_offset});
  DMA_copy_bias.number_of_2d_copies = 1;
  DMA_copy_bias.number_of_1d_copies = 1;
  DMA_copy_bias.length_1d_copy = (uint16_t) ${b_size_byte};
  dory_dma_memcpy_async(DMA_copy_bias);
% endif

% if FLAG_BATCHNORM == 1:
  DMA_copy_k.ext = (uint32_t) l2_W+${l2_off_k};
  DMA_copy_k.loc = (uint32_t) l1_buffer_k;
  DMA_copy_k.number_of_2d_copies = 1;
  DMA_copy_k.number_of_1d_copies = 1;
  DMA_copy_k.length_1d_copy = (uint16_t) ${k_tile_size_byte_transfer};
  dory_dma_memcpy_async(DMA_copy_k);

  DMA_copy_lambda.ext = (uint32_t) l2_W+${l2_off_lambda};
  DMA_copy_lambda.loc = (uint32_t) l1_buffer_lambda;
  DMA_copy_lambda.number_of_2d_copies = 1;
  DMA_copy_lambda.number_of_1d_copies = 1;
  DMA_copy_lambda.length_1d_copy = (uint16_t) ${lambda_tile_size_byte_transfer};
  dory_dma_memcpy_async(DMA_copy_lambda);
% endif

  DMA_copy_W.ext = l2_W;
  DMA_copy_W.loc = l1_buffer_w;
  DMA_copy_W.number_of_2d_copies = 1;

// TODO: I don't understand this part. Why does tile dimension change this? What is 1d_copies?
%if tile_dim_nof == 1:
  DMA_copy_W.number_of_1d_copies = 1;
  DMA_copy_W.length_1d_copy = ${W_tile_size_nof * (W_tile_nif_byte//16) * W_data_size_byte * fs1 * fs2 * 2};
%else:
  DMA_copy_W.number_of_1d_copies = ${W_tile_size_nof};
  DMA_copy_W.length_1d_copy = ${(W_tile_nif_byte//16) * W_data_size_byte * fs1 * fs2 * 2};
%endif

  dory_dma_memcpy_async(DMA_copy_W);

  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = l1_buffer_x;
  DMA_copy_x.number_of_2d_copies = ${x_tile_size_h};
  DMA_copy_x.number_of_1d_copies = ${x_tile_size_w};
  DMA_copy_x.length_1d_copy = ${x_tile_size_nif_byte};
  dory_dma_memcpy_async(DMA_copy_x);

  // ######## #### ##       ########       ##        #######   #######  ########  
  //    ##     ##  ##       ##             ##       ##     ## ##     ## ##     ## 
  //    ##     ##  ##       ##             ##       ##     ## ##     ## ##     ## 
  //    ##     ##  ##       ######         ##       ##     ## ##     ## ########  
  //    ##     ##  ##       ##             ##       ##     ## ##     ## ##        
  //    ##     ##  ##       ##             ##       ##     ## ##     ## ##        
  //    ##    #### ######## ########       ########  #######   #######  ##        

% if flag_DW == 0:
  int total_tiles = ${tile_dim_nof * tile_dim_nif * tile_dim_h * tile_dim_w};
% else:
  int total_tiles = ${tile_dim_nof * tile_dim_h * tile_dim_w};
% endif

  // tile loop nest
  for(iter=0; iter < total_tiles; iter++) {

% if tile_dim_nif != 1 and flag_DW == 0:
    // loop nest is nof,h,w,nif
    _i_nif_load += 1;
    if(_i_nif_load==${tile_dim_nif}) 
    {
      _i_nif_load = 0;
% endif

      _i_w_load += 1;
      if(_i_w_load==${tile_dim_w}) 
      {
        _i_w_load = 0;
        _i_h_load += 1;
        if(_i_h_load==${tile_dim_h}) 
        {
          _i_h_load = 0;

% if flag_DW == 1:
        _i_nif_load += 1;
% endif

          _i_nof_load += 1;
        }
      }

% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif

% if tile_dim_nof * tile_dim_h * tile_dim_w * tile_dim_nif != 1:
    // compute double buffering offsets and update db state

  % if tile_dim_nif != 1:
    exec_db_W = !db_state_W ? ${W_tile_size_byte} : 0;
  % elif tile_dim_nof != 1:
    if (_i_nof_load != _i_nof_exec)
      exec_db_W = !db_state_W ? ${W_tile_size_byte} : 0;
  % endif

    db_W = !db_state_W ? ${W_tile_size_byte} : 0;

    db_y = !db_state_y ? ${y_tile_size_byte} : 0;
    store_db_y = db_state_y ? ${y_tile_size_byte} : 0;

  % if FLAG_BATCHNORM == 1:
    if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec)
      db_act = !db_state_W ? ${k_tile_size_byte_transfer} : 0;
    else
      db_act =  db_state_W ? ${k_tile_size_byte_transfer} : 0;

    exec_db_act = db_state_W ? ${k_tile_size_byte_transfer} : 0;
  % endif
% endif

% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
    db_x = !db_state_x ? ${x_tile_size_byte} : 0;
    exec_db_x = db_x;
% endif

    //switch all double buffering offset and y only after that all n_input_features have been analyzed: we need to pass all n_in to produce a single fil

    // double buffered reads

    // ##        #######     ###    ########  
    // ##       ##     ##   ## ##   ##     ## 
    // ##       ##     ##  ##   ##  ##     ## 
    // ##       ##     ## ##     ## ##     ## 
    // ##       ##     ## ######### ##     ## 
    // ##       ##     ## ##     ## ##     ## 
    // ########  #######  ##     ## ########  

    if(iter < total_tiles-1) {
      asm volatile("": : :"memory");

% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
      x_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif})   ? ${x_tile_size_nif_byte_last} : ${x_tile_size_nif_byte};
      // additionally overlap by padding for the first tile after a border one
      //this because in the first tile we use less pixels from x_buffer, since we have the ones of padding
      pad_offset_h=0, pad_offset_w=0;
      if(_i_h_load > 0)
        pad_offset_h = ${padding_top};
      if(_i_w_load > 0)
        pad_offset_w = ${padding_left};
% endif

      x_tile_size_h   = (_i_h_load+1 == ${tile_dim_h})   ? ${x_tile_size_h_last} : ${x_tile_size_h};
      x_tile_size_w   = (_i_w_load+1 == ${tile_dim_w})   ? ${x_tile_size_w_last} : ${x_tile_size_w};
      y_tile_size_h   = (_i_h_load+1 == ${tile_dim_h})   ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w   = (_i_w_load+1 == ${tile_dim_w})   ? ${y_tile_size_w_last} : ${y_tile_size_w};
      W_tile_size_nof = (_i_nof_load+1 == ${tile_dim_nof}) ? ${W_tile_size_nof_last} : ${W_tile_size_nof};
      W_tile_size_nif = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_last} : ${W_tile_size_nif};

% if flag_DW == 1:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${fs1}*${fs2};
% else:
      W_tile_size_byte = W_tile_size_nof*W_tile_size_nif*${W_data_size_byte}*${fs1}*${fs2}/8;
% endif

      W_length_nif_byte = (_i_nif_load+1 == ${tile_dim_nif}) ? ${W_tile_size_nif_byte_last} : ${W_tile_nif_byte};
      // transfer of next input tile in double buffering

% if tile_dim_nif * tile_dim_h * tile_dim_w != 1:
      DMA_copy_x.ext = dory_get_tile_3d(l2_x, _i_h_load, _i_w_load, _i_nif_load, ${x_tile_size_h}, ${x_tile_size_w}, ${x_tile_size_nif}, ${x_w}, ${nif*g},  ${conv_overlap1}, ${conv_overlap2},0, pad_offset_h, pad_offset_w, 0, ${x_data_size_byte});
      DMA_copy_x.loc = l1_buffer_x + db_x;
      DMA_copy_x.number_of_2d_copies = x_tile_size_h;
      DMA_copy_x.number_of_1d_copies = x_tile_size_w;
      DMA_copy_x.length_1d_copy = x_length_nif_byte;
      dory_dma_memcpy_async(DMA_copy_x);
% endif

      // transfer of next weight tile if changed input or output channels
      if (_i_nif_load!=_i_nif_exec || _i_nof_load!=_i_nof_exec) {
% if flag_DW == 0:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, _i_nif_load, ${W_tile_size_nof}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
% else:
        DMA_copy_W.ext = dory_get_tile_3d(l2_W, _i_nof_load, 0, 0, ${W_tile_size_nof*8/W_data_size_byte}, ${fs1}*${fs2}, ${W_tile_size_nif}, ${fs1}*${fs2}, ${nif}, 0,0,0,0,0,0, ${W_data_size_byte});
% endif

        DMA_copy_W.loc = l1_buffer_w + db_W;

% if tile_dim_nof == 1:
        DMA_copy_W.number_of_1d_copies = 1;
        DMA_copy_W.length_1d_copy = W_tile_size_nof * W_length_nif_byte;
% else:
        DMA_copy_W.number_of_1d_copies = W_tile_size_nof;
        DMA_copy_W.length_1d_copy = W_length_nif_byte;
% endif

        dory_dma_memcpy_async(DMA_copy_W);

% if FLAG_BATCHNORM == 1:
        DMA_copy_k.ext = (uint32_t) l2_W+${l2_off_k} + ${k_tile_size_byte_transfer}*_i_nof_load;
        DMA_copy_k.loc = (uint32_t) l1_buffer_k + db_act;
        DMA_copy_k.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(DMA_copy_k);

        DMA_copy_lambda.ext = (uint32_t) l2_W+${l2_off_lambda} + ${lambda_tile_size_byte_transfer}*_i_nof_load;
        DMA_copy_lambda.loc = (uint32_t) l1_buffer_lambda + db_act;
        DMA_copy_lambda.length_1d_copy = (uint16_t) W_tile_size_nof * ${int(act_dim_bit/8)};
        dory_dma_memcpy_async(DMA_copy_lambda);
% endif
      }
    }

    // program NE in LOAD stage to take advantage of multi-context
    if(iter < total_tiles-1) {
      int is_border_tile = _i_nif_load+1 == ${tile_dim_nif} || _i_h_load+1 == ${tile_dim_h} || _i_w_load+1 == ${tile_dim_w} || _i_nof_load+1 == ${tile_dim_nof};
      if (is_border_tile) {
        // reinit task data structure
        nnx_task_init(&nnx_task_remainder);

        nnx_weights.data      = l1_buffer_w + exec_db_W;
        nnx_weights.depth     = W_tile_size_nif;
        nnx_weights.n_weights = W_tile_size_nof;

        nnx_input.data      = l1_buffer_x + exec_db_x;
        nnx_input.height    = x_tile_size_h;
        nnx_input.width     = x_tile_size_w;
        nnx_input.depth     = W_tile_size_nif;

        nnx_output.data     = l1_buffer_y + db_y;
        nnx_output.height   = y_tile_size_h;
        nnx_output.width    = y_tile_size_w;
        nnx_output.depth    = W_tile_size_nof;
        nnx_conv_${fs1}x${fs2}(&nnx_task_remainder, nnx_weights, nnx_input, nnx_output);

        norm.scale = l1_buffer_k + db_act;
        norm.bias  = l1_buffer_lambda + db_act;
        nnx_norm_quant(&nnx_task_remainder, norm, quant);
      }
      else {
        // do not reinit -- simply update the pointers
        nnx_task.weights_ptr     = l1_buffer_w + exec_db_W;
        nnx_task.infeat_ptr      = l1_buffer_x + db_x;
        nnx_task.outfeat_ptr     = l1_buffer_y + db_y;
        nnx_task.scale_ptr       = l1_buffer_k + db_act;
        nnx_task.scale_bias_ptr  = l1_buffer_lambda + db_act;
      }

      VERBOSE_PRINT("Acquire iter=%d total=%d bool=%d\n", iter, total_tiles, iter<total_tiles-1);
      int id = nnx_acquire();
      VERBOSE_PRINT("  Job_id=%d\n", id);    
      nnx_offload(is_border_tile ? &nnx_task_remainder : &nnx_task);
    }

    // ######## ##     ## ########  ######  
    // ##        ##   ##  ##       ##    ## 
    // ##         ## ##   ##       ##       
    // ######      ###    ######   ##       
    // ##         ## ##   ##       ##       
    // ##        ##   ##  ##       ##    ## 
    // ######## ##     ## ########  ######

    if(iter == 0) {

% if FLAG_BATCHNORM == 1:    
      dory_dma_barrier(DMA_copy_k);
      dory_dma_barrier(DMA_copy_lambda);
% endif      

      dory_dma_barrier(DMA_copy_x);
      dory_dma_barrier(DMA_copy_W);
    }

    // run the layer on NE (non-blocking)
    if (iter == 0 || iter < total_tiles-1) {
      nnx_run_async();
    }

    //  ######  ########  #######  ########  ######## 
    // ##    ##    ##    ##     ## ##     ## ##       
    // ##          ##    ##     ## ##     ## ##       
    //  ######     ##    ##     ## ########  ######   
    //       ##    ##    ##     ## ##   ##   ##       
    // ##    ##    ##    ##     ## ##    ##  ##       
    //  ######     ##     #######  ##     ## ######## 
    
% if tile_dim_nif != 1 and flag_DW == 0:
    if(_i_nif_load == 0) {
% endif
      // busy-wait until the next job is started
      if(iter != total_tiles-1)
        nnx_wait_on_id(iter);

      // in the last tile, wait for the end of the job
      if(iter == total_tiles-1)
        nnx_wait();

      y_tile_size_h = (_i_h_exec + 1 == ${tile_dim_h}) ? ${y_tile_size_h_last} : ${y_tile_size_h};
      y_tile_size_w = (_i_w_exec + 1 == ${tile_dim_w}) ? ${y_tile_size_w_last} : ${y_tile_size_w};
      y_length_nof_byte = (_i_nof_exec + 1 == ${tile_dim_nof}) ? ${y_length_nof_byte_last} : ${y_tile_size_nof_byte};

      DMA_copy_y.ext = dory_get_tile_3d(l2_y, _i_h_exec, _i_w_exec, _i_nof_exec, ${y_tile_size_h}, ${y_tile_size_w}, ${y_tile_size_nof}, ${y_w}, ${int(nof*factor)}, 0, 0, 0, 0, 0, 0, ${y_data_size_byte});
      DMA_copy_y.loc = l1_buffer_y + store_db_y;
      DMA_copy_y.number_of_2d_copies = y_tile_size_h;
      DMA_copy_y.number_of_1d_copies = y_tile_size_w;
      DMA_copy_y.length_1d_copy = y_length_nof_byte;
      dory_dma_memcpy_async(DMA_copy_y);   
% if tile_dim_nif != 1 and flag_DW == 0:
    }
% endif
    // update prev iterators
    db_state_y = !db_state_y; 
    db_state_x = !db_state_x;
    if (_i_nif_load != _i_nif_exec || _i_nof_load != _i_nof_exec)
      db_state_W = !db_state_W;
    _i_nof_exec = _i_nof_load;
    _i_nif_exec = _i_nif_load;
    _i_h_exec = _i_h_load;
    _i_w_exec = _i_w_load;
  }

% if not TEST:
  // wait for final write
  dory_dma_barrier(DMA_copy_y);
  dory_dma_deallocate(dory_dma_channel);
% endif

  // clear NNX for cleanup
  nnx_soft_clear();
}
