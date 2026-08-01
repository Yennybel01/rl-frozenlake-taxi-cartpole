# Actividad 06 — Aprendizaje por Refuerzo

Implementación de **Q-Learning** y **Policy Gradient (REINFORCE)** para resolver cuatro entornos clásicos de [Gymnasium](https://gymnasium.farama.org/): `FrozenLake`, `FrozenLake Slippery`, `Taxi-v3` y `CartPole`.

**Objetivo:** comprender el funcionamiento del aprendizaje por refuerzo implementando una Q-Table desde cero (algoritmo Q-Learning) en entornos deterministas y estocásticos, y comparar dicho enfoque tabular contra un método basado en gradiente de política cuando el espacio de estados es continuo.

## Contenido del notebook

El notebook [`Actividad_06_FrozenLake__FrozenLake_Slippery__Taxi_v3_y_CartPole.ipynb`](./Actividad_06_FrozenLake__FrozenLake_Slippery__Taxi_v3_y_CartPole.ipynb) está organizado en cuatro secciones independientes y autocontenidas (cada una puede ejecutarse por separado):

| # | Sección | Entorno | Algoritmo | Espacio de estados |
|---|---|---|---|---|
| 1 | Solución 1 | `FrozenLake-v1` (`is_slippery=False`) | Q-Learning (Q-Table) | 16 estados discretos, transiciones deterministas |
| 2 | Solución 2 | `FrozenLake-v1` (`is_slippery=True`) | Q-Learning (Q-Table) | 16 estados discretos, transiciones estocásticas |
| 3 | Solución 3 | `Taxi-v3` | Q-Learning (Q-Table) | 500 estados discretos, 6 acciones |
| 4 | Solución 4 | `CartPole-v1` | Q-Learning discretizado **vs.** Policy Gradient (REINFORCE) | Espacio de estados continuo (4 variables) |

Cada sección sigue la misma estructura:

1. Instalación e importación de librerías
2. Creación del entorno
3. Definición de hiperparámetros (α, γ, ε, decaimiento de ε)
4. Entrenamiento del agente con actualización de Bellman `Q(s,a) ← Q(s,a) + α[r + γ·maxₐ' Q(s',a') − Q(s,a)]`
5. Curva de aprendizaje (recompensa promedio / media móvil por episodio)
6. Evaluación de la política aprendida sin exploración (ε = 0)
7. Visualización de un episodio jugado por el agente entrenado
8. Conclusiones de la sección

## Resultados principales

| Entorno | Episodios de entrenamiento | Tasa de éxito / recompensa final (evaluación) |
|---|---|---|
| FrozenLake (determinista) | — | 100 % de éxito, 6 pasos promedio |
| FrozenLake Slippery | 25 000 | 75.8 % de éxito |
| Taxi-v3 | 8 000 | 100 % de éxito, recompensa promedio 8.18, 12.82 pasos promedio |
| CartPole — Q-Learning discretizado | 8 000 | Recompensa promedio ≈ 13.4 (máx. posible: 500) |
| CartPole — Policy Gradient (REINFORCE) | 800 | Recompensa promedio ≈ 460.7 (máx. posible: 500) |

En `FrozenLake Slippery`, la tasa de éxito no llega al 100 % incluso con una política óptima, ya que el entorno es estocástico (el agente puede deslizarse en una dirección distinta a la elegida). En `CartPole`, Policy Gradient supera ampliamente a Q-Learning discretizado usando 10 veces menos episodios, al trabajar directamente sobre el espacio de estados continuo en lugar de discretizarlo en bins.

## Requisitos

- Python ≥ 3.9
- [Gymnasium](https://gymnasium.farama.org/) (con extras `toy-text` para FrozenLake/Taxi)
- NumPy
- Matplotlib
- PyTorch (solo para la sección de Policy Gradient en CartPole)

Instalación rápida:

```bash
pip install "gymnasium[toy-text]" numpy matplotlib torch
```

El propio notebook instala las dependencias necesarias en la primera celda de cada sección (compatible con Google Colab).

## Cómo ejecutarlo

1. Clonar el repositorio y abrir el notebook con Jupyter o Google Colab.
2. Ejecutar las celdas en orden, sección por sección (cada sección reinstala/reimporta lo que necesita, por lo que no es obligatorio ejecutar todo el notebook de corrido).
3. Las semillas aleatorias están fijadas (`seed=42` / `seed=0`) para favorecer la reproducibilidad de los resultados, aunque pequeñas variaciones son esperables entre ejecuciones por la naturaleza estocástica del entrenamiento.

## Estructura sugerida del repositorio

```
.
├── Actividad_06_FrozenLake__FrozenLake_Slippery__Taxi_v3_y_CartPole.ipynb
└── README.md
```

## Fundamento teórico

- **Q-Learning**: algoritmo off-policy de diferencias temporales que aprende una tabla de valores Q(s,a) mediante la ecuación de Bellman, con política de exploración ε-greedy y decaimiento de ε durante el entrenamiento.
- **Policy Gradient (REINFORCE)**: método que aproxima directamente la política mediante una red neuronal, ajustando sus parámetros en la dirección que incrementa la probabilidad de las acciones asociadas a mayores retornos descontados. Es la alternativa natural cuando discretizar el espacio de estados (como en Q-Learning tabular) resulta poco práctico o pierde información relevante.

## Referencias

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
- [Gymnasium Documentation](https://gymnasium.farama.org/) — Farama Foundation.
