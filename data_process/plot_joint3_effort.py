#!/usr/bin/env python3
"""Plot recorded joint3 effort from a text file."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to effort log file")
    args = ap.parse_args()

    data = np.loadtxt(args.path)
    t = np.arange(len(data))
    plt.plot(t, data)
    plt.xlabel("sample")
    plt.ylabel("joint3 effort")
    plt.title("Joint3 Effort During Pull")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
