import scipy.signal


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


if __name__ == "__main__":
    x = [1, 2, 3, 4, 5]
    discount = 0.9

    print(discounted_cumulative_sums(x, discount))
