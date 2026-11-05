---
title: Gallery
description: Trajectories, attractors and spectra of the systems in edmkit.generate.
---

Each system in [`edmkit.generate`](../../reference/generate/) at the parameters used in `scripts/gallery.py`, which draws every figure below. For most systems there are three views: the time series, the attractor, and the spectrum. Initial transients are dropped. The spectra are Morlet scalograms from [`edmkit.wavelet.cwt`](../../reference/wavelet/#cwt), with the shaded cone of influence marking times that edge effects make unreliable.

## Lorenz

[`lorenz`](../../reference/generate/#lorenz)

![Lorenz time series](../../../assets/gallery/lorenz_traces.png)
![Lorenz attractor](../../../assets/gallery/lorenz_attractor.png)
![Lorenz spectrum](../../../assets/gallery/lorenz_spectrum.png)

## Rössler

[`rossler`](../../reference/generate/#rossler)

![Rössler time series](../../../assets/gallery/rossler_traces.png)
![Rössler attractor](../../../assets/gallery/rossler_attractor.png)
![Rössler spectrum](../../../assets/gallery/rossler_spectrum.png)

## Li three-scroll

[`three_scroll`](../../reference/generate/#three_scroll)

![Li three-scroll time series](../../../assets/gallery/three_scroll_traces.png)
![Li three-scroll attractor](../../../assets/gallery/three_scroll_attractor.png)
![Li three-scroll spectrum](../../../assets/gallery/three_scroll_spectrum.png)

## Chua

[`chua`](../../reference/generate/#chua)

![Chua time series](../../../assets/gallery/chua_traces.png)
![Chua attractor](../../../assets/gallery/chua_attractor.png)
![Chua spectrum](../../../assets/gallery/chua_spectrum.png)

## Duffing

[`duffing`](../../reference/generate/#duffing)

![Duffing time series](../../../assets/gallery/duffing_traces.png)
![Duffing attractor](../../../assets/gallery/duffing_attractor.png)
![Duffing spectrum](../../../assets/gallery/duffing_spectrum.png)

## Double pendulum

[`double_pendulum`](../../reference/generate/#double_pendulum)

![Path of the lower bob of a double pendulum](../../../assets/gallery/double_pendulum_attractor.png)

## Euler spiral

[`euler_spiral`](../../reference/generate/#euler_spiral)

![Euler spiral](../../../assets/gallery/euler_spiral_attractor.png)

## Hindmarsh–Rose

[`hindmarsh_rose`](../../reference/generate/#hindmarsh_rose)

![Hindmarsh–Rose time series](../../../assets/gallery/hindmarsh_rose_traces.png)
![Hindmarsh–Rose attractor](../../../assets/gallery/hindmarsh_rose_attractor.png)
![Hindmarsh–Rose spectrum](../../../assets/gallery/hindmarsh_rose_spectrum.png)

## FitzHugh–Nagumo

[`fitzhugh_nagumo`](../../reference/generate/#fitzhugh_nagumo)

![FitzHugh–Nagumo time series](../../../assets/gallery/fitzhugh_nagumo_traces.png)
![FitzHugh–Nagumo attractor](../../../assets/gallery/fitzhugh_nagumo_attractor.png)
![FitzHugh–Nagumo spectrum](../../../assets/gallery/fitzhugh_nagumo_spectrum.png)

## Morris–Lecar

[`morris_lecar`](../../reference/generate/#morris_lecar)

![Morris–Lecar time series](../../../assets/gallery/morris_lecar_traces.png)
![Morris–Lecar attractor](../../../assets/gallery/morris_lecar_attractor.png)
![Morris–Lecar spectrum](../../../assets/gallery/morris_lecar_spectrum.png)

## Rulkov map

[`rulkov_map`](../../reference/generate/#rulkov_map)

![Rulkov map time series](../../../assets/gallery/rulkov_map_traces.png)
![Rulkov map attractor](../../../assets/gallery/rulkov_map_attractor.png)
![Rulkov map spectrum](../../../assets/gallery/rulkov_map_spectrum.png)

## Pyloric circuit

[`pyloric`](../../reference/generate/#pyloric)

![Pyloric circuit membrane potentials](../../../assets/gallery/pyloric_traces.png)
![Pyloric circuit calcium attractor](../../../assets/gallery/pyloric_attractor.png)
![Pyloric circuit spectrum](../../../assets/gallery/pyloric_spectrum.png)

## Mackey–Glass

[`mackey_glass`](../../reference/generate/#mackey_glass)

![Mackey–Glass time series](../../../assets/gallery/mackey_glass_traces.png)
![Mackey–Glass delay embedding](../../../assets/gallery/mackey_glass_attractor.png)
![Mackey–Glass spectrum](../../../assets/gallery/mackey_glass_spectrum.png)

## Lorenz 96

[`lorenz96`](../../reference/generate/#lorenz96)

![Lorenz 96 heatmap of all variables](../../../assets/gallery/lorenz96_heatmap.png)
![Lorenz 96 spectrum](../../../assets/gallery/lorenz96_spectrum.png)

## Coupled chain

Lorenz, Rössler and Duffing systems coupled as A → B → C ← D, built from [`lorenz`](../../reference/generate/#lorenz), [`rossler`](../../reference/generate/#rossler) and [`duffing`](../../reference/generate/#duffing). Useful as ground truth for [CCM](../../reference/ccm/).

![Chain time series](../../../assets/gallery/chain_traces.png)
![Chain attractors](../../../assets/gallery/chain_attractor.png)
![Chain coupling between driver and driven nodes](../../../assets/gallery/chain_coupling.png)
![Chain spectrum](../../../assets/gallery/chain_spectrum.png)

## CTRNN

[`ctrnn`](../../reference/generate/#ctrnn)

![CTRNN connectivity matrix](../../../assets/gallery/ctrnn_connectivity.png)
![CTRNN activity heatmap](../../../assets/gallery/ctrnn_heatmap.png)
![CTRNN two-unit slices](../../../assets/gallery/ctrnn_slices.png)
![CTRNN spectrum](../../../assets/gallery/ctrnn_spectrum.png)

## Lotka–Volterra

[`lotka_volterra`](../../reference/generate/#lotka_volterra)

![Lotka–Volterra time series](../../../assets/gallery/lotka_volterra_traces.png)
![Lotka–Volterra attractor](../../../assets/gallery/lotka_volterra_attractor.png)
![Lotka–Volterra spectrum](../../../assets/gallery/lotka_volterra_spectrum.png)

## Kuramoto

[`kuramoto`](../../reference/generate/#kuramoto)

![Kuramoto phase heatmap](../../../assets/gallery/kuramoto_heatmap.png)
![Kuramoto mean against natural frequencies](../../../assets/gallery/kuramoto_spectrum.png)
