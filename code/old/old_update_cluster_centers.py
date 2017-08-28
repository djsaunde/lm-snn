def update_cluster_centers(cluster_centers, im, fig):
	'''
	Update the plot of the cluster centers (input to excitatory weights).
	'''
	centers_sqrt = int(math.sqrt(len(cluster_centers)))
	to_show = np.zeros((conv_size * centers_sqrt, conv_size * centers_sqrt))
	for i in xrange(centers_sqrt):
		for j in xrange(centers_sqrt):
			to_show[i * conv_size : (i + 1) * conv_size, j * conv_size : (j + 1) * conv_size] = cluster_centers[i * centers_sqrt + j].reshape((conv_size, conv_size)).T
	im.set_array(to_show)
	fig.canvas.draw()
	return im
