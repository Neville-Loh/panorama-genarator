# # some simplevisualizations
    #
    # fig1, axs1 = pyplot.subplots(1, 2)
    #
    # axs1[0].set_title('Harris response left overlaid on orig image')
    # axs1[1].set_title('Harris response right overlaid on orig image')
    # axs1[0].imshow(px_array_left, cmap='gray')
    # axs1[1].imshow(px_array_right, cmap='gray')
    #
    # # plot a red point in the center of each image
    # circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    # axs1[0].add_patch(circle)
    #
    # circle = Circle((image_width/2, image_height/2), 3.5, color='r')
    # axs1[1].add_patch(circle)
    #
    # pyplot.show()
    #
    # # a combined image including a red matching line as a connection patch artist (from matplotlib)
    #
    # matchingImage = prepareMatchingImage(px_array_left, px_array_right, image_width, image_height)
    #
    # pyplot.imshow(matchingImage, cmap='gray')
    # ax = pyplot.gca()
    # ax.set_title("Matching image")
    #
    # pointA = (image_width/2, image_height/2)
    # pointB = (3*image_width/2, image_height/2)
    # connection = ConnectionPatch(pointA, pointB, "data", edgecolor='r', linewidth=1)
    # ax.add_artist(connection)
    #
    # pyplot.show()

    # plot_histogram(pixelArrayToSingleList(px_array_left)).show()
    # plot_histogram(pixelArrayToSingleList(px_array_right)).show()