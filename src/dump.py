if plot:
    time_x = np.arange(0, len(signal) / float(sampling_rate), 1.0 / sampling_rate)

    plt.subplot(2, 1, 1)
    plt.plot(time_x, np.array(signal))
    for s_lim in seg_limits:
        plt.axvline(x=s_lim[0], color='red')
        plt.axvline(x=s_lim[1], color='red')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, prob_on_set.shape[0] * st_step, st_step),
             prob_on_set)
    plt.title('Signal')
    for s_lim in seg_limits:
        plt.axvline(x=s_lim[0], color='red')
        plt.axvline(x=s_lim[1], color='red')
    plt.title('svm Probability')
    plt.show()
