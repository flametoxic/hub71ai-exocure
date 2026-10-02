"""Мастер-ТЗ блок B, чего не было в присланных документах: B1a тепловая зона 1R1C (теплопритоки, шаг
Эйлера, калибровка с ограничениями и 72 ч истории), B1b PINN/FNO/HNN за исследовательским гейтом,
B2 Trust(a), B3 корреляция / частная корреляция / лаг внутри физических границ F26 и ансамбль,
B5 Beta-априор ребра и размер выборки микроэксперимента. Всё — кандидаты и функции, прав на действие нет."""
from .assertion_trust import assertion_trust
from .discovery_stats import best_lag, ccf, ensemble_admission, partial_correlation, pearson, pair_statistics
from .experiments import edge_beta, experiment_gate, microexperiment_sample_size, sample_edge_probability
from .neural_physics import conservation_test, fno_spectral_layer, hamiltonian_field, leapfrog, pinn_loss, research_gate
from .thermal_1r1c import ThermalSample, calibrate_zone, euler_step, occupancy_gain, solar_gain

__all__ = ["ThermalSample", "assertion_trust", "best_lag", "calibrate_zone", "ccf", "conservation_test", "edge_beta",
           "ensemble_admission", "euler_step", "experiment_gate", "fno_spectral_layer", "hamiltonian_field", "leapfrog",
           "microexperiment_sample_size", "occupancy_gain", "partial_correlation", "pearson", "pinn_loss", "research_gate",
           "sample_edge_probability", "solar_gain", "pair_statistics"]
