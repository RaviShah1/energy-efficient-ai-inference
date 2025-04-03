import os
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetTotalEnergyConsumption, NVMLError
)

class EnergyTracker:
    def __init__(self, gpu_index=0):
        # GPU setup
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(gpu_index)

        # Initial energy readings
        self.gpu_start = None
        self.cpu_start = None

    def _get_gpu_energy(self):
        try:
            energy_mJ = nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return energy_mJ / 1000.0  # Convert to joules
        except NVMLError as e:
            print(f"Error reading GPU energy: {e}")
            return 0.0

    def _get_cpu_energy(self):
        try:
            with open('/sys/class/powercap/intel-rapl:0/energy_uj', 'r') as file:
                return float(file.read().strip()) / 1_000_000  # Convert µJ to J
        except Exception as e:
            print(f"Error reading CPU energy: {e}")
            return 0.0

    def start(self):
        self.gpu_start = self._get_gpu_energy()
        self.cpu_start = self._get_cpu_energy()

    def end(self):
        gpu_end = self._get_gpu_energy()
        cpu_end = self._get_cpu_energy()

        gpu_delta = gpu_end - self.gpu_start if self.gpu_start is not None else 0.0
        cpu_delta = cpu_end - self.cpu_start if self.cpu_start is not None else 0.0

        return {
            "gpu_energy_joules": gpu_delta,
            "cpu_energy_joules": cpu_delta
        }

    def shutdown(self):
        nvmlShutdown()
