from matplotlib import pyplot


def generate_distance_distributions(unfiltered_distance, filtered_distance):
    fig1, axs1 = pyplot.subplots(2, sharex=True, tight_layout=True)
    axs1[0].hist(x=unfiltered_distance, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs1[0].grid(axis='y', alpha=0.75)
    axs1[0].set_xlabel("Distance Between Paired Points")
    axs1[0].set_ylabel("Frequency")
    axs1[0].set_title('Distance Between Paired Points (Before outlier rejection)')
    axs1[1].hist(x=filtered_distance, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
    axs1[1].grid(axis='y', alpha=0.75)
    axs1[1].set_xlabel("Distance Between Paired Points")
    axs1[1].set_ylabel("Frequency")
    axs1[1].set_title('Distance Between Paired Points (After outlier rejection)')
    pyplot.show()