ит
с│
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8И▓
З
QNetwork/dense/kernelVarHandleOp*
shape:	А*&
shared_nameQNetwork/dense/kernel*
_output_shapes
: *
dtype0
А
)QNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense/kernel*
dtype0*
_output_shapes
:	А
~
QNetwork/dense/biasVarHandleOp*
shape:*
dtype0*
_output_shapes
: *$
shared_nameQNetwork/dense/bias
w
'QNetwork/dense/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense/bias*
dtype0*
_output_shapes
:
╨
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernelVarHandleOp*
dtype0*
shape: *
_output_shapes
: *G
shared_name86QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel
╔
JQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel/Read/ReadVariableOpReadVariableOp6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel*
dtype0*&
_output_shapes
: 
└
4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/biasVarHandleOp*
shape: *E
shared_name64QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias*
_output_shapes
: *
dtype0
╣
HQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias/Read/ReadVariableOpReadVariableOp4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias*
dtype0*
_output_shapes
: 
╘
8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1VarHandleOp*
shape: @*I
shared_name:8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1*
_output_shapes
: *
dtype0
═
LQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1/Read/ReadVariableOpReadVariableOp8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1*&
_output_shapes
: @*
dtype0
─
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1VarHandleOp*
shape:@*G
shared_name86QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1*
_output_shapes
: *
dtype0
╜
JQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1/Read/ReadVariableOpReadVariableOp6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1*
_output_shapes
:@*
dtype0
╘
8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2VarHandleOp*I
shared_name:8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2*
_output_shapes
: *
dtype0*
shape:@@
═
LQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2/Read/ReadVariableOpReadVariableOp8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2*&
_output_shapes
:@@*
dtype0
─
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2VarHandleOp*G
shared_name86QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2*
_output_shapes
: *
dtype0*
shape:@
╜
JQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2/Read/ReadVariableOpReadVariableOp6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2*
_output_shapes
:@*
dtype0
╚
5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernelVarHandleOp*F
shared_name75QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel*
_output_shapes
: *
dtype0*
shape:
└А
┴
IQNetwork/EncodingNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel* 
_output_shapes
:
└А*
dtype0
┐
3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasVarHandleOp*D
shared_name53QNetwork/EncodingNetwork/EncodingNetwork/dense/bias*
_output_shapes
: *
dtype0*
shape:А
╕
GQNetwork/EncodingNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3QNetwork/EncodingNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:А*
dtype0
P
ConstConst*
valueB :
         *
_output_shapes
: *
dtype0

NoOpNoOp
ъ$
Const_1Const"/device:CPU:0*г$
valueЩ$BЦ$ BП$
%
_wrapped_policy

signatures


_q_network
 
t
_encoder
_q_value_layer
trainable_variables
	variables
regularization_losses
		keras_api
О

_flat_preprocessing_layers
_postprocessing_layers
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
F
0
1
2
3
4
5
6
7
8
9
F
0
1
2
3
4
5
6
7
8
9
 
Ъ
layer_regularization_losses

layers
trainable_variables
	variables
regularization_losses
 non_trainable_variables
!metrics

"0
#
#0
$1
%2
&3
'4
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
 
Ъ
(layer_regularization_losses

)layers
trainable_variables
	variables
regularization_losses
*non_trainable_variables
+metrics
vt
VARIABLE_VALUEQNetwork/dense/kernelK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEQNetwork/dense/biasI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
,layer_regularization_losses

-layers
trainable_variables
	variables
regularization_losses
.non_trainable_variables
/metrics
ШХ
VARIABLE_VALUE6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernelK_wrapped_policy/_q_network/trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/biasK_wrapped_policy/_q_network/trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1K_wrapped_policy/_q_network/trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1K_wrapped_policy/_q_network/trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2K_wrapped_policy/_q_network/trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2K_wrapped_policy/_q_network/trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ЧФ
VARIABLE_VALUE5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernelK_wrapped_policy/_q_network/trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ХТ
VARIABLE_VALUE3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasK_wrapped_policy/_q_network/trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
 
 
й
0	arguments
1_variable_dict
2_trainable_weights
3_non_trainable_weights
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

kernel
bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h

kernel
bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
h

kernel
bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
R
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
h

kernel
bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
 
*
"0
#1
$2
%3
&4
'5
 
 
 
 
 
 
 
 
 
 
 
 
 
Ъ
Llayer_regularization_losses

Mlayers
4trainable_variables
5	variables
6regularization_losses
Nnon_trainable_variables
Ometrics

0
1

0
1
 
Ъ
Player_regularization_losses

Qlayers
8trainable_variables
9	variables
:regularization_losses
Rnon_trainable_variables
Smetrics

0
1

0
1
 
Ъ
Tlayer_regularization_losses

Ulayers
<trainable_variables
=	variables
>regularization_losses
Vnon_trainable_variables
Wmetrics

0
1

0
1
 
Ъ
Xlayer_regularization_losses

Ylayers
@trainable_variables
A	variables
Bregularization_losses
Znon_trainable_variables
[metrics
 
 
 
Ъ
\layer_regularization_losses

]layers
Dtrainable_variables
E	variables
Fregularization_losses
^non_trainable_variables
_metrics

0
1

0
1
 
Ъ
`layer_regularization_losses

alayers
Htrainable_variables
I	variables
Jregularization_losses
bnon_trainable_variables
cmetrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
Z
action_0/discountPlaceholder*
shape:*
dtype0*
_output_shapes
:
u
action_0/observationPlaceholder*
shape:TT*
dtype0*&
_output_shapes
:TT
X
action_0/rewardPlaceholder*
_output_shapes
:*
dtype0*
shape:
[
action_0/step_typePlaceholder*
shape:*
dtype0*
_output_shapes
:
╝
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_16QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_18QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_26QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_25QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasQNetwork/dense/kernelQNetwork/dense/bias*.
_gradient_op_typePartitionedCall-2228476*
Tin
2*
_output_shapes
:**
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_2228358*
Tout
2	
╝
PartitionedFunctionCallPartitionedCall*.
_gradient_op_typePartitionedCall-2228478*	
Tin
 *
config_proto	RRШ*.
f)R'
%__inference_signature_wrapper_2228363*

Tout
 
ъ
PartitionedCallPartitionedCallConst*.
_gradient_op_typePartitionedCall-2228479*
Tin
2*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*.
f)R'
%__inference_signature_wrapper_2228371*
Tout
2
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╞
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)QNetwork/dense/kernel/Read/ReadVariableOp'QNetwork/dense/bias/Read/ReadVariableOpJQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel/Read/ReadVariableOpHQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias/Read/ReadVariableOpLQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_1/Read/ReadVariableOpJQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_1/Read/ReadVariableOpLQNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_2/Read/ReadVariableOpJQNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_2/Read/ReadVariableOpIQNetwork/EncodingNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGQNetwork/EncodingNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpConst_1*)
f$R"
 __inference__traced_save_2228510*
Tout
2*.
_gradient_op_typePartitionedCall-2228511*
Tin
2*
_output_shapes
: **
config_proto

GPU 

CPU2J 8
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameQNetwork/dense/kernelQNetwork/dense/bias6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias8QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_16QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_18QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel_26QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias_25QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel3QNetwork/EncodingNetwork/EncodingNetwork/dense/bias*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-2228554*,
f'R%
#__inference__traced_restore_2228553Ь╘
┼
╣
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228421

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+                           @*
Tout
2*
Tin
2*.
_gradient_op_typePartitionedCall-2228416*\
fWRU
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228410Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
╝q
Ш
__inference_action_724769
	time_step
time_step_1
time_step_2
time_step_3R
Nqnetwork_encodingnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resourceS
Oqnetwork_encodingnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resourceT
Pqnetwork_encodingnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resourceU
Qqnetwork_encodingnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resourceT
Pqnetwork_encodingnetwork_encodingnetwork_conv2d_2_conv2d_readvariableop_resourceU
Qqnetwork_encodingnetwork_encodingnetwork_conv2d_2_biasadd_readvariableop_resourceQ
Mqnetwork_encodingnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nqnetwork_encodingnetwork_encodingnetwork_dense_biasadd_readvariableop_resource1
-qnetwork_dense_matmul_readvariableop_resource2
.qnetwork_dense_biasadd_readvariableop_resource
identity	ИвFQNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpвEQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpвHQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpвGQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpвHQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOpвGQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOpвEQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpвDQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpв%QNetwork/dense/BiasAdd/ReadVariableOpв$QNetwork/dense/MatMul/ReadVariableOpy
$QNetwork/EncodingNetwork/lambda/CastCasttime_step_3*

DstT0*&
_output_shapes
:TT*

SrcT0n
)QNetwork/EncodingNetwork/lambda/truediv/yConst*
valueB
 *  C*
dtype0*
_output_shapes
: ┴
'QNetwork/EncodingNetwork/lambda/truedivRealDiv(QNetwork/EncodingNetwork/lambda/Cast:y:02QNetwork/EncodingNetwork/lambda/truediv/y:output:0*
T0*&
_output_shapes
:TTК
EQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_encodingnetwork_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: Ц
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2DConv2D+QNetwork/EncodingNetwork/lambda/truediv:z:0MQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*&
_output_shapes
: А
FQNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpReadVariableOpOqnetwork_encodingnetwork_encodingnetwork_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: Д
7QNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAddBiasAdd?QNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D:output:0NQNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp:value:0*&
_output_shapes
: *
T0п
4QNetwork/EncodingNetwork/EncodingNetwork/conv2d/ReluRelu@QNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
: О
GQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpReadVariableOpPqnetwork_encodingnetwork_encodingnetwork_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0▒
8QNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2DConv2DBQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Relu:activations:0OQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp:value:0*
strides
*&
_output_shapes
:		@*
T0*
paddingVALIDД
HQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpQqnetwork_encodingnetwork_encodingnetwork_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0К
9QNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAddBiasAddAQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D:output:0PQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp:value:0*&
_output_shapes
:		@*
T0│
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/ReluReluBQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd:output:0*&
_output_shapes
:		@*
T0О
GQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOpReadVariableOpPqnetwork_encodingnetwork_encodingnetwork_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@@*
dtype0│
8QNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2DConv2DDQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Relu:activations:0OQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOp:value:0*
strides
*&
_output_shapes
:@*
T0*
paddingVALIDД
HQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpQqnetwork_encodingnetwork_encodingnetwork_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0К
9QNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAddBiasAddAQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D:output:0PQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@│
6QNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/ReluReluBQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@
.QNetwork/EncodingNetwork/flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"    @  ▄
(QNetwork/EncodingNetwork/flatten/ReshapeReshapeDQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Relu:activations:07QNetwork/EncodingNetwork/flatten/Reshape/shape:output:0*
T0*
_output_shapes
:	└В
DQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_encodingnetwork_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
└А*
dtype0ъ
5QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0LQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
_output_shapes
:	А*
T0 
EQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_encodingnetwork_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А√
6QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAddBiasAdd?QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul:product:0MQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
_output_shapes
:	А*
T0ж
3QNetwork/EncodingNetwork/EncodingNetwork/dense/ReluRelu?QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd:output:0*
_output_shapes
:	А*
T0┴
$QNetwork/dense/MatMul/ReadVariableOpReadVariableOp-qnetwork_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0╣
QNetwork/dense/MatMulMatMulAQNetwork/EncodingNetwork/EncodingNetwork/dense/Relu:activations:0,QNetwork/dense/MatMul/ReadVariableOp:value:0*
_output_shapes

:*
T0╛
%QNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp.qnetwork_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0Ъ
QNetwork/dense/BiasAddBiasAddQNetwork/dense/MatMul:product:0-QNetwork/dense/BiasAdd/ReadVariableOp:value:0*
_output_shapes

:*
T0u
*ShiftedCategorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         е
 ShiftedCategorical_1/mode/ArgMaxArgMaxQNetwork/dense/BiasAdd:output:03ShiftedCategorical_1/mode/ArgMax/dimension:output:0*
_output_shapes
:*
T0G
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R l
addAddV2)ShiftedCategorical_1/mode/ArgMax:output:0add/y:output:0*
_output_shapes
:*
T0	T
Deterministic/atolConst*
value	B	 R *
dtype0	*
_output_shapes
: T
Deterministic/rtolConst*
value	B	 R *
dtype0	*
_output_shapes
: h
%Deterministic_1/sample/sample_shape/xConst*
valueB *
dtype0*
_output_shapes
: Н
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

SrcT0*
_output_shapes
: *

DstT0f
Deterministic_1/sample/ShapeConst*
valueB:*
dtype0*
_output_shapes
:a
Deterministic_1/sample/Shape_1Const*
valueB *
dtype0*
_output_shapes
: a
Deterministic_1/sample/Shape_2Const*
valueB *
dtype0*
_output_shapes
: б
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: е
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:_
Deterministic_1/sample/ConstConst*
valueB *
dtype0*
_output_shapes
: p
&Deterministic_1/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:d
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Й
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
_output_shapes
:*
T0*
NЛ
"Deterministic_1/sample/BroadcastToBroadcastToadd:z:0&Deterministic_1/sample/concat:output:0*
_output_shapes

:*
T0	o
Deterministic_1/sample/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      t
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic_1/sample/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:┬
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
T0*
Index0*
_output_shapes
:*
end_maskf
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : р
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
T0*
_output_shapes
:*
Nе
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*
_output_shapes
:Y
clip_by_value/Minimum/yConst*
dtype0	*
_output_shapes
: *
value	B	 RР
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*
_output_shapes
:Q
clip_by_value/yConst*
dtype0	*
_output_shapes
: *
value	B	 R r
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*
_output_shapes
:х
IdentityIdentityclip_by_value:z:0G^QNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpF^QNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpI^QNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpH^QNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpI^QNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOpH^QNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOpF^QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp&^QNetwork/dense/BiasAdd/ReadVariableOp%^QNetwork/dense/MatMul/ReadVariableOp*
T0	*
_output_shapes
:"
identityIdentity:output:0*_
_input_shapesN
L::::TT::::::::::2О
EQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2N
%QNetwork/dense/BiasAdd/ReadVariableOp%QNetwork/dense/BiasAdd/ReadVariableOp2Ф
HQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOpHQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/BiasAdd/ReadVariableOp2О
EQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOpEQNetwork/EncodingNetwork/EncodingNetwork/conv2d/Conv2D/ReadVariableOp2Р
FQNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOpFQNetwork/EncodingNetwork/EncodingNetwork/conv2d/BiasAdd/ReadVariableOp2Т
GQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOpGQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/Conv2D/ReadVariableOp2L
$QNetwork/dense/MatMul/ReadVariableOp$QNetwork/dense/MatMul/ReadVariableOp2М
DQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2Т
GQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOpGQNetwork/EncodingNetwork/EncodingNetwork/conv2d_1/Conv2D/ReadVariableOp2Ф
HQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOpHQNetwork/EncodingNetwork/EncodingNetwork/conv2d_2/BiasAdd/ReadVariableOp: : :)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step: :	 : : : :) %
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step: : :
 
У
ь
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228435

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
:@@*
dtype0м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+                           @*
strides
*
T0*
paddingVALIDа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @е
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ў
√
(__inference_polymorphic_action_fn_724783
	step_type

reward
discount
observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity	ИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13**
config_proto

GPU 

CPU2J 8*
Tout
2	*"
fR
__inference_action_724769*
_output_shapes
:*
Tin
2*-
_gradient_op_typePartitionedCall-724770u
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
_output_shapes
:*
T0	"
identityIdentity:output:0*_
_input_shapesN
L::::TT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :&"
 
_user_specified_namereward:+'
%
_user_specified_nameobservation: :	 : : : :) %
#
_user_specified_name	step_type:($
"
_user_specified_name
discount: : :
 
Р
¤
*__inference_function_with_signature_727250
	step_type

reward
discount
observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity	ИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13**
config_proto

GPU 

CPU2J 8*
Tout
2	*1
f,R*
(__inference_polymorphic_action_fn_727236*
_output_shapes
:*
Tin
2*-
_gradient_op_typePartitionedCall-727237u
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
_output_shapes
:*
T0	"
identityIdentity:output:0*_
_input_shapesN
L::::TT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :($
"
_user_specified_name
0/reward:-)
'
_user_specified_name0/observation: :	 : : : :+ '
%
_user_specified_name0/step_type:*&
$
_user_specified_name
0/discount: : :
 
я
г
(__inference_polymorphic_action_fn_727398
time_step_step_type
time_step_reward
time_step_discount
time_step_observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity	ИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCalltime_step_step_typetime_step_rewardtime_step_discounttime_step_observationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
_output_shapes
:*
Tin
2*
Tout
2	**
config_proto

GPU 

CPU2J 8*"
fR
__inference_action_724769*-
_gradient_op_typePartitionedCall-724770u
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
_output_shapes
:*
T0	"
identityIdentity:output:0*_
_input_shapesN
L::::TT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :0,
*
_user_specified_nametime_step/reward:51
/
_user_specified_nametime_step/observation: :	 : : : :3 /
-
_user_specified_nametime_step/step_type:2.
,
_user_specified_nametime_step/discount: : :
 
·'
е
 __inference__traced_save_2228510
file_prefix4
0savev2_qnetwork_dense_kernel_read_readvariableop2
.savev2_qnetwork_dense_bias_read_readvariableopU
Qsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_read_readvariableopS
Osavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_read_readvariableopW
Ssavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_1_read_readvariableopU
Qsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_1_read_readvariableopW
Ssavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_2_read_readvariableopU
Qsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_2_read_readvariableopT
Psavev2_qnetwork_encodingnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_bias_read_readvariableop
savev2_1_const_1

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *<
value3B1 B+_temp_9184d335ee034e89b45a21b695809401/part*
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
_output_shapes
: *
NL

num_shardsConst*
_output_shapes
: *
value	B :*
dtype0f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
value	B : *
dtype0У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ь
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*Х
valueЛBИ
BK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEBI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
dtype0Б
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*'
valueB
B B B B B B B B B B *
dtype0г
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_qnetwork_dense_kernel_read_readvariableop.savev2_qnetwork_dense_bias_read_readvariableopQsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_read_readvariableopOsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_read_readvariableopSsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_1_read_readvariableopQsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_1_read_readvariableopSsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_2_read_readvariableopQsavev2_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_2_read_readvariableopPsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_bias_read_readvariableop"/device:CPU:0*
dtypes
2
*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0┼
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
NЦ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*Г
_input_shapesr
p: :	А:: : : @:@:@@:@:
└А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : :
 
Н
°
%__inference_signature_wrapper_2228358
discount
observation

reward
	step_type"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity	ИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*3
f.R,
*__inference_function_with_signature_727250*
Tout
2	**
config_proto

GPU 

CPU2J 8*
_output_shapes
:*
Tin
2*-
_gradient_op_typePartitionedCall-727259u
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
:"
identityIdentity:output:0*_
_input_shapesN
L::TT::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :-)
'
_user_specified_name0/observation:+'
%
_user_specified_name0/step_type: :	 : : : :* &
$
_user_specified_name
0/discount:($
"
_user_specified_name
0/reward: : :
 
Х
'
%__inference_signature_wrapper_2228363╓
PartitionedFunctionCallPartitionedCall*	
Tin
 *-
_gradient_op_typePartitionedCall-727283*3
f.R,
*__inference_function_with_signature_727278*
config_proto	RRШ*

Tout
 *
_output_shapes
 *
_input_shapes 
Л
,
*__inference_function_with_signature_727278╟
PartitionedFunctionCallPartitionedCall*	
Tin
 *-
_gradient_op_typePartitionedCall-727277*$
fR
__inference_function_724676*
config_proto	RRШ*

Tout
 *
_output_shapes
 *
_input_shapes 
╢
8
__inference_<lambda>_724684
unknown
identity>
IdentityIdentityunknown*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
: :  
У
ь
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: @*
dtype0м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
T0*
paddingVALID*A
_output_shapes/
-:+                           @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:@*
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @е
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
т
Q
%__inference_signature_wrapper_2228371
partitionedcall_args_0
identity 
PartitionedCallPartitionedCallpartitionedcall_args_0**
config_proto

GPU 

CPU2J 8*
Tout
2*
_output_shapes
: *-
_gradient_op_typePartitionedCall-727298*
Tin
2*3
f.R,
*__inference_function_with_signature_727293O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
: :  
Й
Г
(__inference_polymorphic_action_fn_727236
	time_step
time_step_1
time_step_2
time_step_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13
identity	ИвStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13*
Tin
2*-
_gradient_op_typePartitionedCall-724770**
config_proto

GPU 

CPU2J 8*"
fR
__inference_action_724769*
_output_shapes
:*
Tout
2	u
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
_output_shapes
:*
T0	"
identityIdentity:output:0*_
_input_shapesN
L::::TT::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step: :	 : : : :) %
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step: : :
 
╒2
В
#__inference__traced_restore_2228553
file_prefix*
&assignvariableop_qnetwork_dense_kernel*
&assignvariableop_1_qnetwork_dense_biasM
Iassignvariableop_2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernelK
Gassignvariableop_3_qnetwork_encodingnetwork_encodingnetwork_conv2d_biasO
Kassignvariableop_4_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_1M
Iassignvariableop_5_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_1O
Kassignvariableop_6_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_2M
Iassignvariableop_7_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_2L
Hassignvariableop_8_qnetwork_encodingnetwork_encodingnetwork_dense_kernelJ
Fassignvariableop_9_qnetwork_encodingnetwork_encodingnetwork_dense_bias
identity_11ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1я
RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*Х
valueЛBИ
BK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEBI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEBK_wrapped_policy/_q_network/trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEД
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:
*'
valueB
B B B B B B B B B B ╨
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:В
AssignVariableOpAssignVariableOp&assignvariableop_qnetwork_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0Ж
AssignVariableOp_1AssignVariableOp&assignvariableop_1_qnetwork_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:й
AssignVariableOp_2AssignVariableOpIassignvariableop_2_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernelIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:з
AssignVariableOp_3AssignVariableOpGassignvariableop_3_qnetwork_encodingnetwork_encodingnetwork_conv2d_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:л
AssignVariableOp_4AssignVariableOpKassignvariableop_4_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_1Identity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0й
AssignVariableOp_5AssignVariableOpIassignvariableop_5_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_1Identity_5:output:0*
_output_shapes
 *
dtype0N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:л
AssignVariableOp_6AssignVariableOpKassignvariableop_6_qnetwork_encodingnetwork_encodingnetwork_conv2d_kernel_2Identity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0й
AssignVariableOp_7AssignVariableOpIassignvariableop_7_qnetwork_encodingnetwork_encodingnetwork_conv2d_bias_2Identity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
_output_shapes
:*
T0и
AssignVariableOp_8AssignVariableOpHassignvariableop_8_qnetwork_encodingnetwork_encodingnetwork_dense_kernelIdentity_8:output:0*
_output_shapes
 *
dtype0N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0ж
AssignVariableOp_9AssignVariableOpFassignvariableop_9_qnetwork_encodingnetwork_encodingnetwork_dense_biasIdentity_9:output:0*
_output_shapes
 *
dtype0М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B ╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 л
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0╕
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6: : : : : :+ '
%
_user_specified_namefile_prefix: : :	 : :
 
┼
╣
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228396

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*\
fWRU
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228385*.
_gradient_op_typePartitionedCall-2228391*A
_output_shapes/
-:+                            **
config_proto

GPU 

CPU2J 8*
Tout
2Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
2

__inference_function_724676*
_input_shapes 
┼
╣
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228446

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*A
_output_shapes/
-:+                           @*
Tin
2*
Tout
2*\
fWRU
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228435*.
_gradient_op_typePartitionedCall-2228441Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
У
ь
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228385

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*&
_output_shapes
: *
dtype0м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+                            а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            е
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*H
_input_shapes7
5:+                           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
╪
V
*__inference_function_with_signature_727293
partitionedcall_args_0
identityЁ
PartitionedCallPartitionedCallpartitionedcall_args_0*
_output_shapes
: *
Tout
2*$
fR
__inference_<lambda>_724684*
Tin
2*-
_gradient_op_typePartitionedCall-727289**
config_proto

GPU 

CPU2J 8O
IdentityIdentityPartitionedCall:output:0*
_output_shapes
: *
T0"
identityIdentity:output:0*
_input_shapes
: :  "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*P
get_train_step> 
int32
PartitionedCall:0 tensorflow/serving/predict*Ъ
actionП
-
0/step_type
action_0/step_type:0
=
0/observation,
action_0/observation:0TT
'
0/reward
action_0/reward:0
+

0/discount
action_0/discount:0-
action#
StatefulPartitionedCall:0	tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*1
get_initial_statetensorflow/serving/predict:уе
v
_wrapped_policy

signatures

daction
eget_initial_state
f
train_step"
_generic_user_object
.

_q_network"
_generic_user_object
N

gaction
hget_initial_state
iget_train_step"
signature_map
▄
_encoder
_q_value_layer
trainable_variables
	variables
regularization_losses
		keras_api
*j&call_and_return_all_conditional_losses
k__call__"л
_tf_keras_networkП{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null}
Г

_flat_preprocessing_layers
_postprocessing_layers
trainable_variables
	variables
regularization_losses
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"╕
_tf_keras_networkЬ{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null}
╚

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"г
_tf_keras_layerЙ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
layer_regularization_losses

layers
trainable_variables
	variables
regularization_losses
 non_trainable_variables
!metrics
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
'
"0"
trackable_list_wrapper
C
#0
$1
%2
&3
'4"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
(layer_regularization_losses

)layers
trainable_variables
	variables
regularization_losses
*non_trainable_variables
+metrics
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
(:&	А2QNetwork/dense/kernel
!:2QNetwork/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
,layer_regularization_losses

-layers
trainable_variables
	variables
regularization_losses
.non_trainable_variables
/metrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
P:N 26QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel
B:@ 24QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias
P:N @26QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel
B:@@24QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias
P:N@@26QNetwork/EncodingNetwork/EncodingNetwork/conv2d/kernel
B:@@24QNetwork/EncodingNetwork/EncodingNetwork/conv2d/bias
I:G
└А25QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel
B:@А23QNetwork/EncodingNetwork/EncodingNetwork/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ў
0	arguments
1_variable_dict
2_trainable_weights
3_non_trainable_weights
4trainable_variables
5	variables
6regularization_losses
7	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Р
_tf_keras_layerЎ{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": ["4wEAAAAAAAAAAQAAAAQAAABDAAAAcxAAAAB0AKABfAB0AqECZAEbAFMAKQJO5wAAAAAA4G9AKQPa\nAnRm2gRjYXN02gVmbG9hdCkB2gNvYnOpAHIGAAAA+khDOi9Vc2Vycy90YW91ZmlrLmVsa2hpcmFv\ndWkvUHljaGFybVByb2plY3RzL2JyZWFrb3V0X3JsL3J1bkV4cGVyaWVuY2UucHnaCDxsYW1iZGE+\nIwAAAPMAAAAA\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
т

kernel
bias
8trainable_variables
9	variables
:regularization_losses
;	keras_api
*r&call_and_return_all_conditional_losses
s__call__"╜
_tf_keras_layerг{"class_name": "Conv2D", "name": "EncodingNetwork/conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "EncodingNetwork/conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [8, 8], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}}}
у

kernel
bias
<trainable_variables
=	variables
>regularization_losses
?	keras_api
*t&call_and_return_all_conditional_losses
u__call__"╛
_tf_keras_layerд{"class_name": "Conv2D", "name": "EncodingNetwork/conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "EncodingNetwork/conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [2, 2], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
у

kernel
bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
*v&call_and_return_all_conditional_losses
w__call__"╛
_tf_keras_layerд{"class_name": "Conv2D", "name": "EncodingNetwork/conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "EncodingNetwork/conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
м
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
*x&call_and_return_all_conditional_losses
y__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ъ

kernel
bias
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
*z&call_and_return_all_conditional_losses
{__call__"┼
_tf_keras_layerл{"class_name": "Dense", "name": "EncodingNetwork/dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "EncodingNetwork/dense", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}}
 "
trackable_list_wrapper
J
"0
#1
$2
%3
&4
'5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Llayer_regularization_losses

Mlayers
4trainable_variables
5	variables
6regularization_losses
Nnon_trainable_variables
Ometrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Player_regularization_losses

Qlayers
8trainable_variables
9	variables
:regularization_losses
Rnon_trainable_variables
Smetrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Tlayer_regularization_losses

Ulayers
<trainable_variables
=	variables
>regularization_losses
Vnon_trainable_variables
Wmetrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
Xlayer_regularization_losses

Ylayers
@trainable_variables
A	variables
Bregularization_losses
Znon_trainable_variables
[metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
\layer_regularization_losses

]layers
Dtrainable_variables
E	variables
Fregularization_losses
^non_trainable_variables
_metrics
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
`layer_regularization_losses

alayers
Htrainable_variables
I	variables
Jregularization_losses
bnon_trainable_variables
cmetrics
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Л2И
(__inference_polymorphic_action_fn_727398
(__inference_polymorphic_action_fn_724783▒
к▓ж
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsв
в 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
║2╖
__inference_function_724676Ч
Р▓М
FullArgSpec
argsЪ
jself
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
B
__inference_<lambda>_724684
[BY
%__inference_signature_wrapper_2228358
0/discount0/observation0/reward0/step_type
)B'
%__inference_signature_wrapper_2228363
)B'
%__inference_signature_wrapper_2228371
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
х2т▀
╓▓╥
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ц2ур
╫▓╙
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
в 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2ур
╫▓╙
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
в 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▓2п
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228385╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Ч2Ф
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228396╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
▓2п
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228410╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Ч2Ф
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228421╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
▓2п
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228435╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Ч2Ф
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228446╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
	J
ConstЩ
%__inference_signature_wrapper_2228358я
╝в╕
в 
░км
!
0/rewardК
0/reward
7
0/observation&К#
0/observationTT
%

0/discountК

0/discount
'
0/step_typeК
0/step_type""к

actionК
action	└
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228396ГIвF
?в<
:К7
inputs+                           
к "2К/+                            3
__inference_function_724676в

в 
к "в ш
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228435РIвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ Y
%__inference_signature_wrapper_22283710|в

в 
к "к

int32К
int32 ╔
(__inference_polymorphic_action_fn_724783Ь
┬в╛
╢в▓
к▓ж
TimeStep#
	step_typeК
	step_type
rewardК
reward!
discountК
discount3
observation$К!
observationTT
в 
к "I▓F

PolicyStep
actionК
action	
stateв 
infoв ё
(__inference_polymorphic_action_fn_727398─
ъвц
▐в┌
╥▓╬
TimeStep-
	step_type К
time_step/step_type'
rewardК
time_step/reward+
discountК
time_step/discount=
observation.К+
time_step/observationTT
в 
к "I▓F

PolicyStep
actionК
action	
stateв 
infoв ш
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228385РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ └
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228446ГIвF
?в<
:К7
inputs+                           @
к "2К/+                           @=
%__inference_signature_wrapper_2228363в

в 
к "к ш
S__inference_EncodingNetwork/conv2d_layer_call_and_return_conditional_losses_2228410РIвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ :
__inference_<lambda>_724684|в

в 
к "К └
8__inference_EncodingNetwork/conv2d_layer_call_fn_2228421ГIвF
?в<
:К7
inputs+                            
к "2К/+                           @