package gofm

import (
	"fmt"
	"math"
)

const epsilon = 1e-7

func BinaryCrossEntropy(targets []float64, preds []float64) float64 {
	if len(targets) != len(preds) {
		panic(fmt.Sprintf("targets must have the same length of preds. targets: %d, preds: %d", len(targets), len(preds)))
	}
	if len(targets) == 0 {
		return 0.0
	}

	totalValue := 0.0
	for i, target := range targets {
		validateProbability(target)
		pred := preds[i]
		validateProbability(pred)

		totalValue += -target*math.Log(pred+epsilon) - (1.0-target)*math.Log(1.0-pred+epsilon)
	}
	return totalValue / float64(len(targets))
}

func DiffBinaryCrossEntropy(targets []float64, preds []float64) float64 {
	if len(targets) != len(preds) {
		panic(fmt.Sprintf("targets must have the same length of preds. targets: %d, preds: %d", len(targets), len(preds)))
	}
	if len(targets) == 0 {
		return 0.0
	}

	totalValue := 0.0
	for i, target := range targets {
		totalValue += diffBinaryCrossEntropyElement(target, preds[i])
	}

	return totalValue / float64(len(targets))
}

func diffBinaryCrossEntropyElement(target float64, pred float64) float64 {
	validateProbability(target)
	validateProbability(pred)

	if pred == 1.0 {
		pred = 1.0 - epsilon
	}
	if pred == 0.0 {
		pred = 0.0 + epsilon
	}
	return target/pred - (1.0-target)/(1.0-pred)
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func validateProbability(p float64) {
	if p < 0.0 || p > 1.0 {
		panic("probability must be [0.0, 1.0]")
	}
}
