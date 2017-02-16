def buildMinibatchDiscriminator(features, numFeatures, kernels, kernelDim=5, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        # TODO: no undefined dimensions until 1.0 release
        batchTensor = tf.get_variable('disc_minibatch',
                       shape=[numFeatures, kernels, kernelDim],
                       initializer=tf.truncated_normal_initializer(stddev=0.1),
                       regularizer=slim.l2_regularizer(0.05))
        

        flatFeatures = slim.flatten(features)
        multFeatures = tf.einsum('ij,jkl->ikl',flatFeatures, batchTensor)
        multFeaturesExpanded1 = tf.expand_dims(multFeatures,[1])

        fn = lambda x: x - multFeatures

        multFeaturesDiff = tf.exp(
            -tf.reduce_sum(
                tf.abs(
                    tf.map_fn(fn, multFeaturesExpanded1)
                ),
            axis=[3])
        )

        output = tf.reduce_sum(multFeaturesDiff, axis=[1]) - 1 # differs from paper, but convergence seems better with -1 in my experiments
    
    return output
