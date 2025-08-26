from multilayer_tb_kubo import example_band_structure
import matplotlib.pyplot as plt

xs, bands = example_band_structure()

plt.figure(figsize=(6,4))
for i in range(bands.shape[1]):
    plt.plot(xs, bands[:, i], 'k-', lw=0.5)
plt.xlabel("k-path index")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("examples/figures/band_structure.png")
plt.show()
