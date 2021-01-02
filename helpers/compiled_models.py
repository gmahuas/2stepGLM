import numpy as np
import tensorflow as tf
import custom_penalties as cust_pen

def compiled_LN_model(nx, ny, n_basis_stim, l1_reg_stim=0, l2_reg_stim=0, l2_lapl_reg_stim=0, lapl_axis=None):    
    # Inputs
    inputs = tf.keras.Input(shape=(nx*ny*n_basis_stim), dtype='float32', name='input')

    # Layers
    filters = tf.keras.layers.Dense(1, activation=tf.keras.activations.exponential, 
                                    use_bias=True, bias_initializer=tf.keras.initializers.constant(0), 
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg_stim, l2=l2_reg_stim),
                                    name='stimulus_filter')
    
    # Build the model
    outputs = filters(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # L2 penalty on the spatial laplacian of the stimulus filter
    if l2_lapl_reg_stim!=0 and lapl_axis!=None:
        model.add_loss(lambda: l2_lapl_reg_stim*cust_pen.lapl_pen_stim(nx, ny, n_basis_stim, lapl_axis, filters))
    
    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Poisson())
    
    return model

def compiled_GLM_model(nx, ny, n_cells, n_basis_stim, n_basis_coupl, n_basis_self,
                       l1_reg_stim=0, l2_reg_stim=0, l2_lapl_reg=0, lapl_axis=None,
                       l1_reg_coupl=0, l2_reg_coupl=0, l1_reg_self=0, l2_reg_self=0):    
    # Inputs
    input_stim= tf.keras.Input(shape=(nx*ny*n_basis_stim), dtype='float32', name='stimulus_input')
    input_coupl= tf.keras.Input(shape=((n_cells-1)*n_basis_coupl), dtype='float32', name='couplings_input')
    input_self= tf.keras.Input(shape=(n_basis_self), dtype='float32', name='self_input')
    input_refr = tf.keras.Input(shape=(1), dtype='float32', name='refractory_input')
    refr_weight = tf.constant(-1e3)
    
    # Layers
    stim_filters = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg_stim, l2=l2_reg_stim), use_bias=True, name='stimulus_filter')
    coupl_filters = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg_coupl, l2=l2_reg_coupl), use_bias=False, name='coupling_filters')
    self_filter = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg_self, l2=l2_reg_self), use_bias=False, name='self_filter')
    output_layer = tf.keras.layers.Activation(activation=tf.keras.activations.exponential, name='output')
    
    # Model
    x_stim = stim_filters(input_stim)
    x_coupl = coupl_filters(input_coupl)
    x_self = self_filter(input_self)
    
    x = tf.keras.layers.concatenate([x_stim, x_coupl, x_self, refr_weight*input_refr])
    x = tf.reduce_sum(x, axis=1)
    outputs = output_layer(x)
    
    model = tf.keras.Model(inputs=[input_stim, input_coupl, input_self, input_refr], outputs=outputs)
    
    # L2 penalty on the spatial laplacian of the stimulus filter
    if l2_lapl_reg!=0 and lapl_axis!=None:
        model.add_loss(lambda: l2_lapl_reg*cust_pen.lapl_pen_stim(nx, ny, n_basis_stim, lapl_axis, stim_filters))
    
    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Poisson())
    
    return model

def compiled_interaction_model(nt_train, n_cells, n_basis_coupl, n_basis_self, l2_stim_field=0, l2_lapl_reg=0,
                               l1_coupl=0, l2_coupl=0, l1_self=0, l2_self=0):
    # Declare the layers
    stim_field = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(l2_stim_field),
                                       use_bias=True, name='stim_field')
    coupl_filters = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_coupl, l2=l2_coupl),
                                          use_bias=False, name='coupling_filters')
    self_filter = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_self, l2=l2_self),
                                        use_bias=False, name='self_filter')
    output_layer = tf.keras.layers.Activation(activation=tf.keras.activations.exponential)

    # Build the model
    input_stim= tf.keras.Input(shape=(1), dtype='int32')
    input_coupl= tf.keras.Input(shape=((n_cells-1)*n_basis_coupl), dtype='float32')
    input_self= tf.keras.Input(shape=(n_basis_self), dtype='float32')
    input_refr = tf.keras.Input(shape=(1), dtype='float32')
    refr_weight = tf.constant(-1e3)

    x_stim = tf.squeeze(tf.one_hot(input_stim, depth=nt_train), axis=1)
    x_stim = stim_field(x_stim)
    x_coupl = coupl_filters(input_coupl)
    x_self = self_filter(input_self)

    x = tf.keras.layers.concatenate([x_stim, x_coupl, x_self, refr_weight*input_refr])
    x = tf.reduce_sum(x, axis=1)
    outputs = output_layer(x)

    model = tf.keras.Model(inputs=[input_stim, input_coupl, input_self, input_refr], outputs=outputs)

    # L2 penalty on the spatial laplacian of the stimulus field
    if l2_lapl_reg!=0:
        model.add_loss(lambda: l2_lapl_reg*cust_pen.lapl_pen_weights(stim_field))
    
    # Compile
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Poisson())

    return model