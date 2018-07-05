import tensorflow as tf


def group_norm(self, input, num_group=32, epsilon=1e-05, name='group_norm'):
    # We here assume the channel-last ordering (NHWC)
    with tf.variable_scope(name):

        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        
        NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NHWCG)
        
        mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        
        # gamma and beta
        gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))
    
        output = tf.reshape(output, tf.shape(input)) * gamma + beta
    
    return output