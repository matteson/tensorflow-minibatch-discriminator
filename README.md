# tensorflow-minibatch-discriminator
A function that can be used to build a minibatch discriminator for use in Generative Adversarial Neural-Networks, or other applications where in-batch discrimination of similarity can be applied. From section 3.2 "Minibatch discrimination" of https://arxiv.org/pdf/1606.03498.pdf

## Example use
    def buildDiscriminator(images, kernel_size=[5,5], reuse=False):
       '''Toy discriminator model'''
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # Convolutional discriminator
            discriminator = slim.conv2d(images, 96, kernel_size,  stride=2, scope='disc_conv_1')
            discriminator = slim.conv2d(discriminator, 96, kernel_size,  stride=2, normalizer_fn=slim.batch_norm, scope='disc_conv_2')
            discriminator = slim.conv2d(discriminator, 64, kernel_size,  stride=2, normalizer_fn=slim.batch_norm, scope='disc_conv_3')
            discriminatorConvEnd = slim.conv2d(discriminator, 64, kernel_size,  stride=2, normalizer_fn=slim.batch_norm, scope='disc_conv_4')

            numFeatures = height/16*width/16*64
            miniBatchDisc = buildMinibatchDiscriminator(discriminatorConvEnd, numFeatures, kernels=100, reuse=reuse)

            miniBatchSummary = slim.fully_connected(miniBatchDisc, 1, activation_fn = None, scope='disc_full_mini')

            convBottleNeck = slim.flatten(discriminatorConvEnd, scope='bottleneck_flatten')
            convSummary = slim.fully_connected(convBottleNeck, 1, activation_fn = None, scope='disc_full_conv')

            discriminatorBottleneck = tf.concat(1, [convSummary, miniBatchSummary])

            discriminator_logits = slim.fully_connected(discriminatorBottleneck, 1, activation_fn = None, scope='disc_full_final')
            discriminator = tf.nn.sigmoid(discriminator_logits)

        return discriminator, discriminator_logits, discriminatorBottleneck
