import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_data(x, y, figure, columns):
    fig = plt.figure(figure)
    fig.clf()
    fig.gca().cla()
    gs = fig.add_gridspec(3,2)

    for i in range(len(x)):
        ax = fig.add_subplot(gs[i,0])
        ax.pcolormesh(x[i].reshape(10,10))
        ax.set_title("{}:{}".format(columns[0], i))
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[i,1])
        ax.pcolormesh(y[i].reshape(10,10))
        ax.set_title("{}:{}".format(columns[1], i))
        ax.set_xticks([])
        ax.set_yticks([])

def create_random_image(image_size):

    # make a x y grid
    x_grid = ( np.arange(0,image_size[0]) - (image_size[0]/2) ).reshape(1,-1)
    y_grid = ( np.arange(0,image_size[1]) - (image_size[1]/2) ).reshape(-1,1)

    z = np.zeros(tuple(image_size))

    for _ in range(np.random.randint(2,9)):
        width = 1
        x_pos = (int( np.random.rand()>0.5 ) or -1)*np.random.rand()*3
        y_pos = (int( np.random.rand()>0.5 ) or -1)*np.random.rand()*3
        z += np.exp((-(x_grid-x_pos)**2)/width)*np.exp((-(y_grid-y_pos)**2)/width)

    # normalize z
    z = z / np.max(z)
    return z.reshape(image_size[0],image_size[1],1)

def create_data_set(image_size):

    # create 10 random samples
    samples_x, samples_y = [], []
    for _ in range(3):

        # create image
        sample_x = create_random_image(image_size)
        sample_y = create_random_image(image_size)

        samples_x.append(sample_x)
        samples_y.append(sample_y)

    samples_y = np.array(samples_y)
    samples_x = np.array(samples_x)

    return samples_x, samples_y


if __name__ == "__main__":

    # # # # # # # # # # # # # # #
    # create a tensorflow graph #
    # # # # # # # # # # # # # # #

    # create an input image
    placeholder_x = tf.placeholder(dtype=tf.float32, shape=[None,10,10,1])

    # flatten the input layer
    x_flat = tf.layers.flatten(placeholder_x)

    # create a single dense layer with 100 neurons
    layer_1 = tf.layers.dense(inputs=x_flat, units=200)

    layer_2 = tf.layers.dense(inputs=layer_1, units=100)

    # output layer
    network_output = tf.reshape(layer_2, [-1,10,10,1])

    # create the label to calculate loss from the output
    y_label = tf.placeholder(dtype=tf.float32, shape=[None,10,10,1])

    # # # # # # # # # # # # # # # # #
    # create data to input to graph #
    # # # # # # # # # # # # # # # # #
    data_x, data_y = create_data_set(image_size=[10,10])

    # visualize the data
    plot_data(data_x, data_y, figure=1, columns=["data_x", "data_y"])

    # # # # # # # # # # # # # # # #
    # setup training for network  #
    # # # # # # # # # # # # # # # #

    # define loss function
    loss = tf.losses.mean_squared_error(network_output, y_label)
    # define optimizer
    optimizer = tf.train.AdamOptimizer()
    # define training
    run_train = optimizer.minimize(loss)

    initialize = tf.global_variables_initializer()

    # matplotlib interacive plotting mode
    plt.ion()
    with tf.Session() as sess:

        # initialize tensorflow variables
        sess.run(initialize)

        for epoch in range(100):

            print("running training iteration {}".format(epoch))
            sess.run(run_train, feed_dict={placeholder_x:data_x, y_label:data_y})

            # view the output of the network
            output = sess.run(network_output, feed_dict={placeholder_x:data_x})

            plot_data(data_x, output, figure=2, columns=["data_x", "output"])
            plt.pause(0.5)

    plt.ioff()
    plt.show()











