/*
 * pulp_nn_kernels.h
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
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

%if PULPNNEXT == 'XpulpV2':
#ifndef __PULPNN_KERNELS__
#define __PULPNN_KERNELS__
%elif PULPNNEXT == 'XpulpNN':
#ifndef __XPULPNN_KERNELS__
#define __XPULPNN_KERNELS__
%endif

${PULPNNAPI}

#endif