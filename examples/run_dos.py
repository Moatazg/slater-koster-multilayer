from multilayer_tb_kubo import example_compute_dos_and_sigma
import matplotlib.pyplot as plt

Es, dos, sxx = example_compute_dos_and_sigma()

plt.figure(figsize=(6,4))
plt.plot(Es, dos, label="DOS")
plt.plot(Es, sxx, label="σxx")
plt.xlabel("Energy")
plt.legend()
plt.tight_layout()
plt.savefig("examples/figures/dos_sigma.png")
plt.show()
